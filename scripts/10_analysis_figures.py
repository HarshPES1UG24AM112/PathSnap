"""Figures/tables for optimisation, HMM comparison and error analysis (student-test only)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.evaluation import PROC, load_partition, trip_meta  # noqa: E402

FIG = ROOT / "results" / "figures"
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# ---------------- Optuna ----------------
tr = pd.read_csv(ROOT / "experiments/optuna/trials.csv")
comp = tr[tr.state == "COMPLETE"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].scatter(comp.number, comp.value, s=18, color="#2f6db5", label="completed trial")
pr = tr[tr.state == "PRUNED"]
axs[0].scatter(pr.number, np.full(len(pr), comp.value.min()), marker="x", color="#999", label="pruned (shown at floor)")
axs[0].plot(comp.number, comp.value.cummax(), color="#e5322d", label="best so far")
axs[0].axhline(comp.value.iloc[0] if comp.number.iloc[0] == 0 else np.nan, ls="--", color="k", lw=1, label="trial 0 = untuned defaults")
axs[0].set_xlabel("trial"); axs[0].set_ylabel("CV Top-1 (5-fold GroupKFold)"); axs[0].legend(fontsize=7); axs[0].set_title("Optuna TPE optimisation history")
pcols = [c for c in tr.columns if c.startswith("params_")]
cor = {c.replace("params_", ""): np.corrcoef(np.log(comp[c].astype(float)) if c.split("_", 1)[1] in ("learning_rate", "l2_regularization", "min_samples_leaf", "max_leaf_nodes") else comp[c].astype(float), comp.value)[0, 1] for c in pcols}
axs[1].barh(list(cor), list(cor.values()), color="#e08a1e"); axs[1].set_xlabel("Pearson r with CV Top-1 (log scale for log params)"); axs[1].set_title("Hyper-parameter sensitivity (completed trials)")
fig.tight_layout(); fig.savefig(FIG / "models" / "optuna_history.png"); plt.close(fig)

# ---------------- HMM tuning ----------------
ht = pd.read_csv(ROOT / "results/hmm_tuning.csv")
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, ul in zip(axs, sorted(ht.unreachable_logp.unique())):
    piv = ht[ht.unreachable_logp == ul].pivot(index="lam", columns="beta", values="top1")
    im = ax.imshow(piv.values, cmap="viridis", vmin=ht.top1.min(), vmax=ht.top1.max(), aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel("β (m)"); ax.set_title(f"unreachable log-p = {ul}", fontsize=9); ax.grid(False)
axs[0].set_ylabel("λ (emission weight)")
fig.suptitle("HMM decoder grid search – Top-1 on 39 training trips (out-of-fold probabilities)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "hmm_tuning.png"); plt.close(fig)

# ---------------- HMM per trip ----------------
per = pd.read_csv(ROOT / "results/hmm_student_test_per_trip.csv", dtype={"trip_id": str})
x = np.arange(len(per)); w = 0.38
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].bar(x - w / 2, per.clf_top1, w, label="classifier only", color="#f59e0b")
axs[0].bar(x + w / 2, per.hmm_top1, w, label="classifier + HMM", color="#e5322d")
axs[0].set_xticks(x, per.trip_id.str[-2:]); axs[0].set_xlabel("student-test trip (last 2 digits)"); axs[0].set_ylabel("Top-1"); axs[0].set_ylim(0.6, 1); axs[0].legend(fontsize=8)
axs[0].set_title("Observation Top-1 per trip")
axs[1].bar(x - w / 2, per.clf_continuity, w, label="classifier only", color="#f59e0b")
axs[1].bar(x + w / 2, per.hmm_continuity, w, label="classifier + HMM", color="#e5322d")
axs[1].set_xticks(x, per.trip_id.str[-2:]); axs[1].set_ylabel("route continuity"); axs[1].set_ylim(0, 1.05); axs[1].legend(fontsize=8)
axs[1].set_title("Fraction of arc changes that are network-connected")
fig.tight_layout(); fig.savefig(FIG / "hmm_per_trip.png"); plt.close(fig)

# ---------------- error analysis ----------------
_, Os = load_partition("student_test")
P = np.load(PROC / "st_hmm_preds.npz")
rows = []
for tid, g in Os.groupby("trip_id", sort=False):
    u, v, route = trip_meta(tid, "student_test")
    ref = g.ref_arc.to_numpy(); rset = set(route.tolist())
    for kind in ("clf", "hmm"):
        pr = P[f"{tid}_{kind}"]
        e = pr != ref
        twin = (u[pr] == v[ref]) & (v[pr] == u[ref])
        adj = ((v[pr] == u[ref]) | (u[pr] == v[ref])) & ~twin
        miss = g.ref_in_cands.to_numpy() == 0
        on_route = np.array([a in rset for a in pr])
        cats = {
            "reference not in candidates": e & miss,
            "reverse-direction twin arc": e & ~miss & twin,
            "adjacent arc (boundary)": e & ~miss & ~twin & adj,
            "other arc on reference route": e & ~miss & ~twin & ~adj & on_route,
            "off-route arc": e & ~miss & ~twin & ~adj & ~on_route,
        }
        for c, m in cats.items():
            rows.append(dict(trip_id=tid, kind=kind, category=c, n=int(m.sum())))
ea = pd.DataFrame(rows).groupby(["kind", "category"]).n.sum().unstack(0)
ea["share_clf"] = ea.clf / ea.clf.sum(); ea["share_hmm"] = ea.hmm / ea.hmm.sum()
ea.to_csv(ROOT / "results/error_analysis.csv")
print(ea)
fig, ax = plt.subplots(figsize=(8, 3.4))
y = np.arange(len(ea))
ax.barh(y + 0.2, ea.clf, 0.4, label="classifier only", color="#f59e0b"); ax.barh(y - 0.2, ea.hmm, 0.4, label="classifier + HMM", color="#e5322d")
ax.set_yticks(y, ea.index); ax.set_xlabel("erroneous observations (10 student-test trips)"); ax.legend(); ax.set_title("Error taxonomy")
fig.tight_layout(); fig.savefig(FIG / "error_analysis.png"); plt.close(fig)

# tuned vs untuned table
R = {n: json.loads((ROOT / f"experiments/models/{n}.json").read_text()) for n in ("hist_gbm", "hist_gbm_tuned")}
tt = pd.DataFrame({n: {"cv_top1": r["cv_mean"]["top1"], "cv_top1_std": r["cv_std"]["top1"], "cv_macro_f1": r["cv_mean"]["macro_f1"],
                       "cv_log_loss": r["cv_mean"]["log_loss"], "st_top1": r["student_test"]["top1"], "st_macro_f1": r["student_test"]["macro_f1"],
                       "st_log_loss": r["student_test"]["log_loss"], "st_continuity": r["student_test"]["continuity"],
                       "fit_s": r["full_fit_s"], "infer_ms_per_obs": r["student_test_infer_ms_per_obs"]} for n, r in R.items()})
tt.to_csv(ROOT / "results/tuned_vs_untuned.csv")
print(tt)
