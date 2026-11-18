"""Bayesian hyper-parameter optimisation (Optuna TPE) of the selected model on training trips only.

Objective: maximise mean observation-level Top-1 accuracy over the same 5-fold GroupKFold
(groups = trips) used in the model comparison. Evaluator data is never touched; student-test
trips are NOT used inside the objective (only to report tuned vs untuned afterwards).
Usage: python scripts/06_optuna.py [n_trials] [model]
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.evaluation import FEATURES, load_partition, unreliable_train_trips  # noqa: E402
from pathsnap.models import SEED, build_model  # noqa: E402

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODEL = sys.argv[2] if len(sys.argv) > 2 else "hist_gbm"
OUT = io.ROOT / "experiments" / "optuna"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("tm", io.ROOT / "scripts" / "04_train_models.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)


def space(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 1000, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
        max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
    )


X, O = load_partition("train")
bad = unreliable_train_trips()
X, O = X[~X.trip_id.isin(bad)].reset_index(drop=True), O[~O.trip_id.isin(bad)].reset_index(drop=True)
trips = np.array(sorted(O.trip_id.unique()))
splits = list(GroupKFold(n_splits=5).split(trips, groups=trips))
fold_data = []
for tr_i, va_i in splits:
    tr_t, va_t = set(trips[tr_i]), set(trips[va_i])
    Xtr = tm.fit_rows(X[X.trip_id.isin(tr_t)], O[O.trip_id.isin(tr_t)], None, None)
    vm = X.trip_id.isin(va_t).to_numpy()
    fold_data.append((Xtr[FEATURES].to_numpy(), Xtr.label.to_numpy(), X.loc[vm].reset_index(drop=True), int(O.trip_id.isin(va_t).sum())))


def top1(Xva, p, n_obs):
    d = Xva[["trip_id", "obs", "label"]].assign(p=p)
    best = d.loc[d.groupby(["trip_id", "obs"]).p.idxmax()]
    return best.label.sum() / n_obs  # observations without candidates count as errors


def objective(trial):
    params = space(trial)
    scores = []
    for k, (Xt, yt, Xva, n_obs) in enumerate(fold_data):
        m = build_model(MODEL, params)
        m.fit(Xt, yt)
        scores.append(top1(Xva, m.predict_proba(Xva[FEATURES].to_numpy())[:, 1], n_obs))
        trial.report(float(np.mean(scores)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("fold_top1", scores)
    return float(np.mean(scores))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize", study_name=f"{MODEL}_top1",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                                storage=f"sqlite:///{(OUT / 'study.db').as_posix()}", load_if_exists=False)
    # enqueue the untuned default configuration as a reference trial
    study.enqueue_trial(dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1e-4, max_features=1.0, max_iter=300))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    wall = time.time() - t0
    df = study.trials_dataframe()
    df.to_csv(OUT / "trials.csv", index=False)
    best = {"model": MODEL, "best_params": study.best_params, "best_cv_top1": study.best_value, "n_trials": len(study.trials),
            "n_pruned": int((df.state == "PRUNED").sum()), "n_complete": int((df.state == "COMPLETE").sum()),
            "wall_time_s": wall, "objective": "mean 5-fold GroupKFold observation Top-1 accuracy", "sampler": "TPE(seed=42)",
            "pruner": "MedianPruner(n_startup_trials=8, warmup=1 fold)"}
    (OUT / "best_params.json").write_text(json.dumps(best, indent=1))
    print(json.dumps(best, indent=1))
