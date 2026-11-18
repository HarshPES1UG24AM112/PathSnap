"""Candidate generation and (observation, candidate-arc) feature engineering.

Formulation: map matching is cast as *candidate ranking*. Every GPS observation i gets a
candidate set C_i of real arc ids from its own trip network; each row of the design matrix
is one (i, arc) pair, carrying ``trip_id``, ``obs`` and ``arc_id`` so that predictions
always map back to an actual road segment. The binary target is ``arc_id == reference arc``.
Arc ids are never used as features or classes (they are local to each trip network).
All features are deterministic functions of track + network (no fitted parameters).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff, point_segment
from .network import ProjectedNetwork
from .preprocessing import CleanTrack

RADIUS = 50.0  # m, primary candidate radius
K_MAX = 12  # keep at most K nearest arcs within RADIUS
FALLBACK_RADIUS = 250.0  # if nothing within RADIUS, take K_FALLBACK nearest within this
K_FALLBACK = 4
SIGMA = 10.0  # m, fixed Gaussian scale for the distance-share feature
NO_ADJ = 60.0  # value used when no connected neighbour candidate exists

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "gauss_share", "is_nearest", "n_cands",
    "heading_diff", "axis_diff", "heading_valid", "speed", "dt", "is_jump",
    "arc_len", "proj_t", "at_endpoint", "has_twin", "out_deg_end", "in_deg_start",
    "dist_prev", "dist_next", "ctx_mean_dist", "adj_prev_dist", "adj_next_dist",
]


def generate_candidates(ct: CleanTrack, pn: ProjectedNetwork, radius=RADIUS, k_max=K_MAX):
    xy = ct.df[["x", "y"]].to_numpy()
    near = pn.arcs_near(xy, radius)
    obs_l, arc_l, d_l, t_l = [], [], [], []
    for i, arcs in enumerate(near):
        d, t = (pn.dist_to_arcs(xy[i], arcs) if len(arcs) else (np.empty(0), np.empty(0)))
        sel = d <= radius
        if not sel.any():
            arcs2 = pn.arcs_near(xy[i:i + 1], FALLBACK_RADIUS)[0]
            if len(arcs2):
                d2, t2 = pn.dist_to_arcs(xy[i], arcs2)
                o = np.argsort(d2, kind="stable")[:K_FALLBACK]
                arcs, d, t = arcs2[o], d2[o], t2[o]
            else:
                continue  # empty candidate set -> handled downstream
        else:
            arcs, d, t = arcs[sel], d[sel], t[sel]
            o = np.argsort(d, kind="stable")[:k_max]
            arcs, d, t = arcs[o], d[o], t[o]
        obs_l.append(np.full(len(arcs), i))
        arc_l.append(arcs)
        d_l.append(d)
        t_l.append(t)
    if not obs_l:
        return pd.DataFrame(columns=["obs", "arc_id", "dist", "proj_t"])
    return pd.DataFrame({"obs": np.concatenate(obs_l), "arc_id": np.concatenate(arc_l),
                         "dist": np.concatenate(d_l), "proj_t": np.concatenate(t_l)})


def build_features(ct: CleanTrack, pn: ProjectedNetwork, cand: pd.DataFrame) -> pd.DataFrame:
    tr = ct.df
    n = len(tr)
    c = cand.copy()
    obs = c.obs.to_numpy()
    arc = c.arc_id.to_numpy()
    g = c.groupby("obs")["dist"]
    c["dist_rank"] = g.rank(method="first").to_numpy() - 1
    c["dist_minus_min"] = c.dist - g.transform("min")
    w = np.exp(-0.5 * (c.dist / SIGMA) ** 2)
    c["gauss_share"] = w / w.groupby(obs).transform("sum").clip(lower=1e-300)
    c["is_nearest"] = (c.dist_rank == 0).astype(int)
    c["n_cands"] = g.transform("size")
    head = tr.heading.to_numpy()[obs]
    hv = tr.heading_valid.to_numpy()[obs]
    brg = pn.bearing[arc]
    hd = angle_diff(head, brg)
    c["heading_diff"] = np.where(hv, hd, 90.0)
    c["axis_diff"] = np.where(hv, np.minimum(hd, 180 - hd), 45.0)
    c["heading_valid"] = hv.astype(int)
    c["speed"] = tr.speed.to_numpy()[obs]
    c["dt"] = np.minimum(tr.dt.to_numpy()[obs], 120.0)
    c["is_jump"] = tr.is_jump.to_numpy()[obs].astype(int)
    c["arc_len"] = pn.length[arc]
    c["at_endpoint"] = ((c.proj_t <= 1e-6) | (c.proj_t >= 1 - 1e-6)).astype(int)
    c["has_twin"] = pn.has_twin[arc].astype(int)
    c["out_deg_end"] = pn.out_deg[pn.v[arc]]
    c["in_deg_start"] = pn.in_deg[pn.u[arc]]
    xy = tr[["x", "y"]].to_numpy()
    ctx = []
    for off in (-2, -1, 1, 2):
        j = np.clip(obs + off, 0, n - 1)
        d, _ = point_segment(xy[j], pn.A[arc], pn.B[arc])
        ctx.append(d)
    c["dist_prev"], c["dist_next"] = ctx[1], ctx[2]
    c["ctx_mean_dist"] = (ctx[0] + ctx[1] + c.dist.to_numpy() + ctx[2] + ctx[3]) / 5
    # connectivity to neighbouring observations' candidates (label-free)
    by_obs = {}
    for o, a, d in zip(obs.tolist(), arc.tolist(), c.dist.tolist()):
        by_obs.setdefault(o, []).append((a, d))
    u, v = pn.u, pn.v
    adj_prev = np.full(len(c), NO_ADJ)
    adj_next = np.full(len(c), NO_ADJ)
    for r, (o, a) in enumerate(zip(obs.tolist(), arc.tolist())):
        ua, va = u[a], v[a]
        best = NO_ADJ
        for a2, d2 in by_obs.get(o - 1, ()):
            if (a2 == a or v[a2] == ua) and d2 < best:
                best = d2
        adj_prev[r] = best
        best = NO_ADJ
        for a2, d2 in by_obs.get(o + 1, ()):
            if (a2 == a or u[a2] == va) and d2 < best:
                best = d2
        adj_next[r] = best
    c["adj_prev_dist"], c["adj_next_dist"] = adj_prev, adj_next
    return c


def trip_design(ct: CleanTrack, pn: ProjectedNetwork) -> pd.DataFrame:
    cand = generate_candidates(ct, pn)
    return build_features(ct, pn, cand)
