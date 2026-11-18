"""Evaluation metrics for candidate ranking and route reconstruction.

Decision rule for the binary (obs, candidate) task: exactly one candidate per observation is
predicted positive - the arg-max - because exactly one arc is correct. This makes precision,
recall, F1 and the confusion matrix threshold-free and consistent with Top-1 accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_recall_fscore_support

EPS = 1e-7


def argmax_flags(df: pd.DataFrame, score_col: str) -> np.ndarray:
    key = df["trip_id"].astype(str) + ":" + df["obs"].astype(str) if "trip_id" in df else df["obs"]
    idx = df.groupby(key.values, sort=False)[score_col].idxmax()
    flag = np.zeros(len(df), dtype=np.int8)
    flag[df.index.get_indexer(idx.values)] = 1
    return flag


def per_obs_prediction(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """One row per (trip_id, obs): predicted arc and its normalised probability."""
    d = df[["trip_id", "obs", "arc_id", score_col]].copy()
    d["_norm"] = d[score_col] / d.groupby(["trip_id", "obs"])[score_col].transform("sum").clip(lower=EPS)
    top = d.loc[d.groupby(["trip_id", "obs"])[score_col].idxmax()]
    return top.rename(columns={"arc_id": "pred_arc", "_norm": "pred_prob"})[["trip_id", "obs", "pred_arc", "pred_prob"]]


def candidate_metrics(df: pd.DataFrame, score_col: str) -> dict:
    y = df["label"].to_numpy()
    p = np.clip(df[score_col].to_numpy(), EPS, 1 - EPS)
    yhat = argmax_flags(df.reset_index(drop=True), score_col)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average=None, labels=[0, 1], zero_division=0)
    norm = df[score_col] / df.groupby(["trip_id", "obs"])[score_col].transform("sum").clip(lower=EPS)
    pos = norm[df["label"] == 1].clip(lower=EPS)
    return {
        "cand_accuracy": float((y == yhat).mean()),
        "precision_pos": float(prec[1]), "recall_pos": float(rec[1]),
        "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "obs_nll": float(-np.log(pos).mean()) if len(pos) else np.nan,
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def route_metrics(pred_arc: np.ndarray, ref_arc: np.ndarray, route: np.ndarray, u: np.ndarray, v: np.ndarray) -> dict:
    """pred_arc/ref_arc: one per cleaned observation (pred -1 = no prediction)."""
    has = pred_arc >= 0
    top1 = float(np.mean(pred_arc == ref_arc))
    arc_f1 = float(f1_score(ref_arc, pred_arc, average="macro", zero_division=0))
    p = pred_arc[has]
    ch = np.nonzero(p[1:] != p[:-1])[0]
    if len(ch):
        a, b = p[ch], p[ch + 1]
        cont = float(np.mean(v[a] == u[b]))
    else:
        cont = 1.0
    collapsed = p[np.r_[True, p[1:] != p[:-1]]] if len(p) else p
    ps, rs = set(collapsed.tolist()), set(route.tolist())
    inter = len(ps & rs)
    rp = inter / max(len(ps), 1)
    rr = inter / max(len(rs), 1)
    return {"top1": top1, "arc_macro_f1": arc_f1, "continuity": cont, "n_breaks": int(len(ch) - (v[p[ch]] == u[p[ch + 1]]).sum()) if len(ch) else 0,
            "route_precision": rp, "route_recall": rr, "route_f1": 2 * rp * rr / max(rp + rr, EPS),
            "n_obs": int(len(pred_arc)), "coverage": float(has.mean())}


def aggregate_route(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    w = df["n_obs"]
    out = {"top1": float(np.average(df.top1, weights=w))}  # observation-weighted
    for k in ("arc_macro_f1", "continuity", "route_precision", "route_recall", "route_f1"):
        out[k] = float(df[k].mean())  # trip-averaged
    out["n_breaks"] = int(df.n_breaks.sum())
    return out
