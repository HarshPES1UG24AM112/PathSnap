"""End-to-end label-free inference: raw track + network -> per-observation arcs + route."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import io
from .evaluation import obs_predictions
from .features import FEATURES
from .hmm import HMMParams, build_lattice, reconstruct_route, viterbi
from .pipeline import PreparedTrip, prepare_inputs


@dataclass
class MatchResult:
    prepared: PreparedTrip
    hmm_arcs: np.ndarray
    clf_arcs: np.ndarray
    route: list[int]
    clf_route: list[int]
    breaks: int
    timings_ms: dict


class MapMatcher:
    def __init__(self, model_path: Path, hmm_config: Path):
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        cfg = json.loads(Path(hmm_config).read_text())
        self.hmm = HMMParams(**cfg["params"])

    def match(self, track: pd.DataFrame, net: io.Network, trip_id: str = "") -> MatchResult:
        t = {}
        t0 = time.perf_counter()
        pt = prepare_inputs(track, net, trip_id)
        t["preprocess_candidates_features"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        X = pt.X
        X["score"] = self.model.predict_proba(X[FEATURES].to_numpy())[:, 1] if len(X) else np.empty(0)
        t["classifier"] = (time.perf_counter() - t0) * 1000
        O = pd.DataFrame({"trip_id": trip_id, "obs": np.arange(len(pt.ct.df))})
        clf = obs_predictions(X, O, "score").to_numpy() if len(X) else np.full(len(O), -1)
        t0 = time.perf_counter()
        lat = build_lattice(pt.ct.df, pt.pn, X)
        hmm = viterbi(lat, self.hmm)
        t["hmm_viterbi"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        xy = pt.ct.df[["x", "y"]].to_numpy()
        route, breaks = reconstruct_route(pt.pn, hmm, xy)
        clf_route, _ = reconstruct_route(pt.pn, clf, xy)
        t["route_reconstruction"] = (time.perf_counter() - t0) * 1000
        return MatchResult(pt, hmm, clf, route, clf_route, breaks, t)
