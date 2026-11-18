"""Service layer: trip registry, prediction pipeline and server-side evaluation."""
from __future__ import annotations

import logging
import re
import sys
import threading
import time

import numpy as np

from . import config

sys.path.insert(0, str(config.ROOT / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.inference import MapMatcher, MatchResult  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import route_metrics  # noqa: E402
from pathsnap.visualization import arcs_polylines, network_arcs_in_view  # noqa: E402

log = logging.getLogger("pathsnap.api")
TRIP_RE = re.compile(r"^\d{1,12}$")


class TripNotFound(Exception):
    pass


class PathSnapService:
    def __init__(self):
        self.eval_dir = config.EVAL_DIR
        self.matcher = MapMatcher(config.MODEL_PATH, config.HMM_CONFIG)
        self._cache: dict[str, tuple[MatchResult, dict | None]] = {}
        self._lock = threading.Lock()
        log.info("model loaded: %s | HMM %s | eval dir: %s", config.MODEL_PATH.name, self.matcher.hmm, self.eval_dir)

    # ---- trips ------------------------------------------------------------------------
    def trip_ids(self) -> list[str]:
        if not self.eval_dir.exists():
            return []
        return sorted(d.name for d in self.eval_dir.iterdir() if d.is_dir() and TRIP_RE.match(d.name)
                      and (d / f"{d.name}.track").exists())

    def _check(self, trip_id: str):
        if not TRIP_RE.match(trip_id) or trip_id not in self.trip_ids():
            raise TripNotFound(trip_id)

    def _load_inputs(self, trip_id: str):
        root, part = self.eval_dir.parent, self.eval_dir.name
        return io.load_track(trip_id, part, root), io.load_network(trip_id, part, root)

    def trip_info(self, trip_id: str) -> dict:
        tr, net = self._load_inputs(trip_id)
        return dict(trip_id=trip_id, n_observations=len(tr), duration_s=float(tr.t.max() - tr.t.min()) if len(tr) else 0.0,
                    n_nodes=len(net.node_xy_ll), n_arcs=len(net.arcs))

    # ---- prediction ---------------------------------------------------------------------
    def predict(self, trip_id: str) -> tuple[MatchResult, dict | None]:
        self._check(trip_id)
        with self._lock:
            if trip_id in self._cache:
                log.info("trip %s: served from cache", trip_id)
                return self._cache[trip_id]
        t0 = time.perf_counter()
        track, net = self._load_inputs(trip_id)  # NOTE: no reference route is loaded here
        res = self.matcher.match(track, net, trip_id)
        metrics = self._evaluate(trip_id, res)  # reference used only after prediction is final
        log.info("trip %s: %d obs, %d candidates, route %d arcs, %.0f ms | top1 hmm=%s clf=%s", trip_id,
                 len(res.prepared.ct.df), len(res.prepared.X), len(res.route), (time.perf_counter() - t0) * 1000,
                 None if metrics is None else round(metrics["hmm_top1_accuracy"], 4),
                 None if metrics is None else round(metrics["classifier_top1_accuracy"], 4))
        with self._lock:
            self._cache[trip_id] = (res, metrics)
        return res, metrics

    def _evaluate(self, trip_id: str, res: MatchResult) -> dict | None:
        """Server-side evaluation. Reference labels never leave this method."""
        root, part = self.eval_dir.parent, self.eval_dir.name
        rpath = io.trip_dir(trip_id, part, root) / f"{trip_id}.route"
        if not rpath.exists():
            return None
        route = io.load_route(trip_id, part, root)
        pt = res.prepared
        xy = pt.ct.df[["x", "y"]].to_numpy()
        _, ref, _ = align_route(xy, route, pt.pn.A, pt.pn.B)
        u, v = pt.pn.u, pt.pn.v
        mh = route_metrics(res.hmm_arcs, ref, route, u, v)
        mc = route_metrics(res.clf_arcs, ref, route, u, v)
        s, r = set(res.route), set(route.tolist())
        inter = len(s & r)
        rp, rr = inter / max(len(s), 1), inter / max(len(r), 1)
        cand = pt.X.groupby("obs").arc_id.apply(set)
        miss = np.mean([ref[o] not in cand.get(o, set()) for o in range(len(ref))])
        return dict(hmm_top1_accuracy=mh["top1"], classifier_top1_accuracy=mc["top1"], hmm_arc_macro_f1=mh["arc_macro_f1"],
                    classifier_arc_macro_f1=mc["arc_macro_f1"], hmm_route_continuity=mh["continuity"],
                    classifier_route_continuity=mc["continuity"], hmm_route_precision=rp, hmm_route_recall=rr,
                    hmm_route_f1=2 * rp * rr / max(rp + rr, 1e-9), candidate_miss_rate=float(miss),
                    n_evaluated_observations=len(ref))

    def network(self, trip_id: str) -> dict:
        res, _ = self.predict(trip_id)
        pn = res.prepared.pn
        ids = network_arcs_in_view(pn, res.prepared.ct.df[["x", "y"]].to_numpy(), margin=150, max_arcs=config.MAX_NETWORK_ARCS)
        return dict(trip_id=trip_id, n_arcs_total=int(pn.n_arcs), n_arcs_returned=len(ids),
                    arcs=[[list(map(lambda z: round(z, 6), p)) for p in seg] for seg in arcs_polylines(pn, ids)])

    @staticmethod
    def polylines(res: MatchResult, arcs: list[int]):
        return [[[round(p[0], 6), round(p[1], 6)] for p in seg] for seg in arcs_polylines(res.prepared.pn, arcs)]
