"""HMM map matching with Viterbi decoding.

Hidden state at observation i: one candidate arc c in C_i (real arc id of the trip network,
carried with its projection fraction t_c along the arc).

Emission:   log e_i(c) = lam * log p_i(c), p_i = classifier probability renormalised over C_i.
Transition: c (obs i-1) -> c' (obs i), with network route distance
              d_route = (1-t_c) L_c + SP(v_c, u_c') + t_c' L_c'        (different arcs)
              d_route = |t_c' - t_c| L_c  (+ backward distance as penalty)   (same arc)
            and great-circle step d_gc = |x_i - x_{i-1}|:
              log T = -|d_route - d_gc| / beta           if reachable within cutoff
              log T = unreachable_logp                    otherwise (soft; never -inf so the
                                                           chain cannot dead-end on disconnected roads)
            SP = bounded Dijkstra on the directed arc graph, cutoff = 2*d_gc + 150 m.
Gaps:       if the observation follows a time gap > 30 s or a jump > 500 m, transitions are
            uniform (log T = 0): the decoder restarts while still returning one arc per fix.
Recurrence: delta_i(c') = log e_i(c') + max_c [delta_{i-1}(c) + log T(c, c')]; back-pointers
            give the arg-max path. Observations with an empty candidate set are skipped (no
            prediction, -1) and the chain is restarted after them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .network import ProjectedNetwork

EPS = 1e-6


@dataclass
class HMMParams:
    beta: float = 20.0
    lam: float = 1.0
    unreachable_logp: float = -12.0
    backward_weight: float = 1.0


@dataclass
class TripLattice:
    obs: list[int]  # observation indices with >= 1 candidate
    arcs: list[np.ndarray]
    logp: list[np.ndarray]  # log normalised classifier probabilities
    dev: list[np.ndarray | None]  # |d_route - d_gc| (+backward) per step (None = restart)
    reach: list[np.ndarray | None]
    n_obs: int


def build_lattice(track: pd.DataFrame, pn: ProjectedNetwork, cand: pd.DataFrame, score_col: str = "score") -> TripLattice:
    xy = track[["x", "y"]].to_numpy()
    gap = track["gap_before"].to_numpy()
    groups = {o: g for o, g in cand.groupby("obs", sort=True)}
    obs_l, arcs_l, logp_l, dev_l, reach_l = [], [], [], [], []
    prev = None
    for o in range(len(track)):
        g = groups.get(o)
        if g is None:
            prev = None  # empty candidate set -> restart after it
            continue
        arcs = g.arc_id.to_numpy()
        t = g.proj_t.to_numpy()
        p = np.clip(g[score_col].to_numpy(), EPS, None)
        lp = np.log(p / p.sum())
        dev = reach = None
        if prev is not None and not gap[o] and prev[0] == o - 1:
            _, parcs, pt = prev
            dgc = float(np.hypot(*(xy[o] - xy[o - 1])))
            cutoff = 2 * dgc + 150.0
            dev = np.zeros((len(parcs), len(arcs)))
            reach = np.zeros((len(parcs), len(arcs)), bool)
            for i, (a, ta) in enumerate(zip(parcs.tolist(), pt.tolist())):
                La = pn.length[a]
                sp = None
                for j, (b, tb) in enumerate(zip(arcs.tolist(), t.tolist())):
                    if a == b:
                        fwd = (tb - ta) * La
                        dr = abs(fwd)
                        dev[i, j] = abs(dr - dgc) + (-fwd if fwd < 0 else 0.0)
                        reach[i, j] = True
                        continue
                    if sp is None:
                        sp = pn.shortest_from(int(pn.v[a]), cutoff)
                    s = sp.get(int(pn.u[b]))
                    if s is None:
                        continue
                    dr = (1 - ta) * La + s + tb * pn.length[b]
                    if dr > cutoff:
                        continue
                    dev[i, j] = abs(dr - dgc)
                    reach[i, j] = True
        obs_l.append(o); arcs_l.append(arcs); logp_l.append(lp); dev_l.append(dev); reach_l.append(reach)
        prev = (o, arcs, t)
    return TripLattice(obs_l, arcs_l, logp_l, dev_l, reach_l, len(track))


def viterbi(lat: TripLattice, prm: HMMParams) -> np.ndarray:
    """Return predicted arc per observation (-1 where no candidates)."""
    out = np.full(lat.n_obs, -1, dtype=np.int64)
    n = len(lat.obs)
    if n == 0:
        return out
    back: list[np.ndarray | None] = [None] * n
    delta = prm.lam * lat.logp[0]
    seg_end = []  # (step index, best state) where a segment ends
    for k in range(1, n):
        em = prm.lam * lat.logp[k]
        if lat.dev[k] is None:
            seg_end.append((k - 1, int(np.argmax(delta))))
            back[k] = None
            delta = em + delta.max()  # restart (uniform transition)
            continue
        logT = np.where(lat.reach[k], -lat.dev[k] / prm.beta, prm.unreachable_logp)
        sc = delta[:, None] + logT
        bp = sc.argmax(0)
        back[k] = bp
        delta = em + sc[bp, np.arange(sc.shape[1])]
        delta -= delta.max()  # numerical stability
    seg_end.append((n - 1, int(np.argmax(delta))))
    # backtrack each segment
    states = np.empty(n, dtype=np.int64)
    ends = {k: s for k, s in seg_end}
    k = n - 1
    s = ends[k]
    while k >= 0:
        states[k] = s
        if k == 0:
            break
        if back[k] is None:
            k -= 1
            s = ends[k]
        else:
            s = int(back[k][s])
            k -= 1
    for k in range(n):
        out[lat.obs[k]] = lat.arcs[k][states[k]]
    return out


def reconstruct_route(pn: ProjectedNetwork, pred: np.ndarray, xy: np.ndarray | None = None,
                      slack: float = 2.0, extra: float = 200.0, max_fill: float = 5000.0):
    """Collapse per-observation arcs into a connected arc path, filling missing connector arcs
    with network shortest paths. A connector is only accepted if its length is plausible given
    the GPS displacement between the two observations (<= slack * d_gps + extra); otherwise the
    route is left broken there instead of inventing a detour. Returns (arc list, n_breaks)."""
    idx = [i for i, a in enumerate(pred) if a >= 0]
    seq, first, last = [], [], []
    for i in idx:
        a = int(pred[i])
        if seq and seq[-1] == a:
            last[-1] = i
        else:
            seq.append(a); first.append(i); last.append(i)
    route, breaks = [], 0
    for k, a in enumerate(seq):
        if k > 0:
            prev = seq[k - 1]
            if pn.v[prev] != pn.u[a]:
                bound = max_fill
                if xy is not None:
                    bound = min(max_fill, slack * float(np.hypot(*(xy[first[k]] - xy[last[k - 1]]))) + extra)
                fill = pn.path_arcs(int(pn.v[prev]), int(pn.u[a]), cutoff=bound)
                if fill is None or sum(pn.length[f] for f in fill) > bound:
                    breaks += 1
                else:
                    route.extend(fill)
        route.append(a)
    return route, breaks
