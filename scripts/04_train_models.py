"""Train + compare all models with 5-fold GroupKFold over training trips (groups = trip_id),
then refit on all usable training trips and evaluate on the 10 student-testing trips.

Usage: python scripts/04_train_models.py [model ...]
Outputs: experiments/models/<name>.json, artifacts/models/<name>.joblib,
         data/processed/scores_<name>.parquet (OOF + student_test probabilities)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.evaluation import FEATURES, PROC, evaluate, load_partition, unreliable_train_trips  # noqa: E402
from pathsnap.models import MODEL_SPECS, SEED, build_model, train_rows_cap  # noqa: E402

EXP = io.ROOT / "experiments" / "models"
ART = io.ROOT / "artifacts" / "models"
EXP.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
N_FOLDS = 5


def fit_rows(X, O, cap, rng):
    """Training rows: observations whose reference arc is among candidates (others carry no positive)."""
    ok = O.loc[O.ref_in_cands == 1, ["trip_id", "obs"]]
    Xf = X.merge(ok, on=["trip_id", "obs"])
    if cap and len(Xf) > cap:
        # subsample whole observations to keep candidate groups intact
        keys = Xf[["trip_id", "obs"]].drop_duplicates()
        frac = cap / len(Xf)
        keys = keys.sample(frac=frac, random_state=SEED)
        Xf = Xf.merge(keys, on=["trip_id", "obs"])
    return Xf


def history(model):
    m = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(m, "loss_curve_"):
        return {"train_loss": list(map(float, m.loss_curve_)), "val_score": list(map(float, getattr(m, "validation_scores_", []) or []))}
    if hasattr(m, "train_score_") and len(getattr(m, "train_score_", [])):
        return {"train_loss": [-float(s) for s in m.train_score_], "val_loss": [-float(s) for s in m.validation_score_]}
    return None


def run(name, params=None, tag=None):
    tag = tag or name
    rng = np.random.default_rng(SEED)
    X, O = load_partition("train")
    bad = unreliable_train_trips()
    X, O = X[~X.trip_id.isin(bad)].reset_index(drop=True), O[~O.trip_id.isin(bad)].reset_index(drop=True)
    Xs, Os = load_partition("student_test")
    cap = train_rows_cap(name, params)
    trips = np.array(sorted(O.trip_id.unique()))
    folds = []
    oof = np.full(len(X), np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tr_i, va_i) in enumerate(gkf.split(trips, groups=trips)):
        tr_t, va_t = set(trips[tr_i]), set(trips[va_i])
        Xtr = fit_rows(X[X.trip_id.isin(tr_t)], O[O.trip_id.isin(tr_t)], cap, rng)
        mdl = build_model(name, params)
        t0 = time.time(); mdl.fit(Xtr[FEATURES].to_numpy(), Xtr.label.to_numpy()); t_fit = time.time() - t0
        vm = X.trip_id.isin(va_t).to_numpy()
        t0 = time.time(); oof[vm] = mdl.predict_proba(X.loc[vm, FEATURES].to_numpy())[:, 1]; t_inf = time.time() - t0
        Xva = X[vm].assign(score=oof[vm])
        r_va = evaluate(Xva, O[O.trip_id.isin(va_t)], "score", "train")
        tr_eval_t = sorted(tr_t)[:: max(1, len(tr_t) // 12)]  # training-fold fit check on a trip subset
        tm = X.trip_id.isin(tr_eval_t).to_numpy()
        Xtt = X[tm].assign(score=mdl.predict_proba(X.loc[tm, FEATURES].to_numpy())[:, 1])
        r_tr = evaluate(Xtt, O[O.trip_id.isin(tr_eval_t)], "score", "train")
        folds.append({"fold": k, "n_train_rows": len(Xtr), "val_trips": sorted(va_t), "fit_s": t_fit, "infer_s": t_inf,
                      "val": {kk: vv for kk, vv in r_va.items() if kk != "confusion"},
                      "train_subset": {kk: r_tr[kk] for kk in ("top1", "macro_f1", "log_loss")}})
        print(f"[{tag}] fold {k} top1={r_va['top1']:.4f} f1={r_va['macro_f1']:.4f} ll={r_va['log_loss']:.4f} "
              f"train_top1={r_tr['top1']:.4f} fit={t_fit:.1f}s", flush=True)
    # refit on all usable training trips
    Xall = fit_rows(X, O, cap, rng)
    mdl = build_model(name, params)
    t0 = time.time(); mdl.fit(Xall[FEATURES].to_numpy(), Xall.label.to_numpy()); t_fit = time.time() - t0
    t0 = time.time(); ps = mdl.predict_proba(Xs[FEATURES].to_numpy())[:, 1]; t_inf = time.time() - t0
    r_st = evaluate(Xs.assign(score=ps), Os, "score", "student_test")
    cv = pd.DataFrame([f["val"] for f in folds])
    out = {
        "model": tag, "base_model": name, "family": MODEL_SPECS[name][0],
        "params": {**MODEL_SPECS[name][1], **(params or {})}, "seed": SEED, "features": FEATURES,
        "excluded_train_trips": bad, "n_folds": N_FOLDS, "folds": folds,
        "cv_mean": cv.mean(numeric_only=True).to_dict(), "cv_std": cv.std(numeric_only=True).to_dict(),
        "train_subset_mean": pd.DataFrame([f["train_subset"] for f in folds]).mean().to_dict(),
        "full_fit_s": t_fit, "n_full_train_rows": len(Xall),
        "student_test": r_st, "student_test_infer_s": t_inf,
        "student_test_infer_ms_per_obs": 1000 * t_inf / len(Os), "history": history(mdl),
    }
    (EXP / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    joblib.dump(mdl, ART / f"{tag}.joblib", compress=3)
    pd.DataFrame({"oof": oof}).to_parquet(PROC / f"oof_{tag}.parquet", index=False)
    pd.DataFrame({"score": ps}).to_parquet(PROC / f"st_{tag}.parquet", index=False)
    print(f"[{tag}] CV top1={out['cv_mean']['top1']:.4f}±{out['cv_std']['top1']:.4f} | ST top1={r_st['top1']:.4f} "
          f"f1={r_st['macro_f1']:.4f} ll={r_st['log_loss']:.4f} cont={r_st['continuity']:.3f} fit={t_fit:.1f}s inf={t_inf:.2f}s", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODEL_SPECS)
    for n in names:
        run(n)
