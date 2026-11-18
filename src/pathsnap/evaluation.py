"""Dataset loading for experiments and a single evaluation entry point."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io
from .features import FEATURES
from .metrics import aggregate_route, candidate_metrics, route_metrics

PROC = io.ROOT / "data" / "processed"
UNRELIABLE_MEDIAN_M = 50.0  # trips whose median GPS->reference distance exceeds this are not used for fitting


def unreliable_train_trips() -> list[str]:
    inv = pd.read_csv(io.ROOT / "results" / "eda" / "trip_inventory.csv", dtype={"trip_id": str})
    return inv[(inv.partition == "train") & (inv.gps_to_route_med > UNRELIABLE_MEDIAN_M)].trip_id.tolist()


def load_partition(part: str):
    X = pd.read_parquet(PROC / f"{part}_cands.parquet")
    O = pd.read_parquet(PROC / f"{part}_obs.parquet")
    return X, O


_net_cache: dict = {}


def trip_meta(tid: str, part: str):
    k = (tid, part)
    if k not in _net_cache:
        net = io.load_network(tid, part)
        _net_cache[k] = (net.arcs[:, 0], net.arcs[:, 1], io.load_route(tid, part))
    return _net_cache[k]


def obs_predictions(X: pd.DataFrame, O: pd.DataFrame, score_col: str) -> pd.Series:
    """Arg-max arc for every cleaned observation in O (-1 where no candidates)."""
    top = X.loc[X.groupby(["trip_id", "obs"])[score_col].idxmax(), ["trip_id", "obs", "arc_id"]]
    m = O[["trip_id", "obs"]].merge(top, on=["trip_id", "obs"], how="left")
    return m.arc_id.fillna(-1).astype(np.int64)


def evaluate(X: pd.DataFrame, O: pd.DataFrame, score_col: str, part: str, pred_obs: pd.Series | None = None) -> dict:
    """Candidate-level + observation/route-level metrics. pred_obs overrides arg-max (e.g. HMM)."""
    res = candidate_metrics(X, score_col) if score_col else {}
    pred = obs_predictions(X, O, score_col) if pred_obs is None else pred_obs
    O = O.assign(pred=pred.to_numpy())
    rows = []
    for tid, g in O.groupby("trip_id", sort=False):
        u, v, route = trip_meta(tid, part)
        rows.append(route_metrics(g.pred.to_numpy(), g.ref_arc.to_numpy(), route, u, v))
    res.update(aggregate_route(rows))
    res["cand_miss_rate"] = float(1 - O.ref_in_cands.mean())
    return res


__all__ = ["FEATURES", "load_partition", "evaluate", "unreliable_train_trips", "obs_predictions"]
