"""Consolidate model experiments -> results/model_comparison.csv + comparison figures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io  # noqa: E402

EXP = io.ROOT / "experiments" / "models"
FIG = io.ROOT / "results" / "figures" / "models"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
ORDER = ["logreg", "gaussian_nb", "knn", "decision_tree", "random_forest", "hist_gbm", "mlp", "hist_gbm_tuned"]
NICE = {"logreg": "Logistic Regression", "gaussian_nb": "Gaussian NB", "knn": "KNN (k=50)", "decision_tree": "Decision Tree",
        "random_forest": "Random Forest", "hist_gbm": "HistGradientBoosting", "mlp": "MLP (64-32)", "hist_gbm_tuned": "HistGB (Optuna-tuned)"}

runs = {}
for f in EXP.glob("*.json"):
    r = json.loads(f.read_text())
    runs[r["model"]] = r
names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))

rows = []
for n in names:
    r = runs[n]
    cm, cs, st = r["cv_mean"], r["cv_std"], r["student_test"]
    rows.append({
        "model": n, "name": NICE.get(n, n), "family": r["family"],
        "config": "; ".join(f"{k}={v}" for k, v in r["params"].items()),
        "cv_macro_f1": cm["macro_f1"], "cv_macro_f1_std": cs["macro_f1"], "cv_top1": cm["top1"], "cv_top1_std": cs["top1"],
        "cv_log_loss": cm["log_loss"], "cv_continuity": cm["continuity"], "train_top1": r["train_subset_mean"]["top1"],
        "st_top1": st["top1"], "st_macro_f1": st["macro_f1"], "st_precision": st["precision_pos"], "st_recall": st["recall_pos"],
        "st_accuracy": st["cand_accuracy"], "st_log_loss": st["log_loss"], "st_obs_nll": st["obs_nll"],
        "st_arc_macro_f1": st["arc_macro_f1"], "st_continuity": st["continuity"], "st_route_f1": st["route_f1"],
        "st_cand_miss": st["cand_miss_rate"], "fit_s": r["full_fit_s"], "train_rows": r["n_full_train_rows"],
        "infer_ms_per_obs": r["student_test_infer_ms_per_obs"], "st_confusion": r["student_test"]["confusion"],
    })
df = pd.DataFrame(rows)
df.to_csv(io.ROOT / "results" / "model_comparison.csv", index=False)
print(df[["model", "cv_top1", "cv_top1_std", "cv_macro_f1", "cv_log_loss", "train_top1", "st_top1", "st_macro_f1", "st_log_loss",
          "st_continuity", "st_arc_macro_f1", "fit_s", "infer_ms_per_obs"]].round(4).to_string())

base = df[df.model != "hist_gbm_tuned"]
labels = [NICE.get(m, m) for m in base.model]
x = np.arange(len(base))

# CV macro F1 + per-fold spread
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(x, base.cv_macro_f1, yerr=base.cv_macro_f1_std, capsize=4, color="#2f6db5")
for i, n in enumerate(base.model):
    fv = [f["val"]["macro_f1"] for f in runs[n]["folds"]]
    ax.scatter(np.full(len(fv), i), fv, color="k", s=8, zorder=3)
ax.set_xticks(x, labels, rotation=20, ha="right"); ax.set_ylabel("Macro F1 (5-fold GroupKFold)"); ax.set_ylim(base.cv_macro_f1.min() - 0.05, 1)
ax.set_title("Cross-validation Macro F1 (dots = folds)")
fig.tight_layout(); fig.savefig(FIG / "cv_macro_f1.png"); plt.close(fig)

# student test top1 + macro f1
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.38
ax.bar(x - w / 2, base.st_top1, w, label="Top-1 accuracy (obs)", color="#2f6db5")
ax.bar(x + w / 2, base.st_macro_f1, w, label="Macro F1 (candidates)", color="#e08a1e")
ax.set_xticks(x, labels, rotation=20, ha="right"); ax.set_ylim(min(base.st_top1.min(), base.st_macro_f1.min()) - 0.05, 1); ax.legend(loc="lower right")
ax.set_title("Student-testing performance (10 trips)")
fig.tight_layout(); fig.savefig(FIG / "student_test_top1_f1.png"); plt.close(fig)

# log loss
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(x - w / 2, base.cv_log_loss, w, label="CV", color="#2f6db5")
ax.bar(x + w / 2, base.st_log_loss, w, label="student-test", color="#e08a1e")
ax.set_yscale("log"); ax.set_xticks(x, labels, rotation=20, ha="right"); ax.set_ylabel("binary log loss (log scale)"); ax.legend(); ax.set_title("Probability quality")
fig.tight_layout(); fig.savefig(FIG / "log_loss.png"); plt.close(fig)

# generalisation gap + efficiency
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].bar(x - w / 2, base.train_top1, w, label="training folds (fit data)", color="#999999")
axs[0].bar(x + w / 2, base.cv_top1, w, label="validation folds", color="#2f6db5")
axs[0].set_xticks(x, labels, rotation=25, ha="right"); axs[0].set_ylim(0.6, 1); axs[0].legend(fontsize=8); axs[0].set_title("Train vs validation Top-1 (over/underfitting)")
axs[1].scatter(base.infer_ms_per_obs, base.st_top1, s=40, color="#2f6db5")
for _, r in base.iterrows():
    axs[1].annotate(NICE.get(r.model, r.model), (r.infer_ms_per_obs, r.st_top1), fontsize=7, xytext=(3, 3), textcoords="offset points")
axs[1].set_xscale("log"); axs[1].set_xlabel("inference time (ms per GPS observation)"); axs[1].set_ylabel("student-test Top-1"); axs[1].set_title("Accuracy vs inference cost")
fig.tight_layout(); fig.savefig(FIG / "fit_gap_efficiency.png"); plt.close(fig)

# confusion matrices
fig, axs = plt.subplots(1, len(base), figsize=(2.3 * len(base), 2.6))
for ax, (_, r) in zip(np.atleast_1d(axs), base.iterrows()):
    cm = np.array(r.st_confusion)
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7, color="k")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7); ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=7)
    ax.set_title(NICE.get(r.model, r.model), fontsize=8); ax.grid(False)
fig.suptitle("Student-test candidate confusion matrices (arg-max rule)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png"); plt.close(fig)

# epoch-wise histories only where they exist (MLP, HistGB)
hist = [(n, runs[n]["history"]) for n in names if runs[n].get("history")]
if hist:
    fig, axs = plt.subplots(1, len(hist), figsize=(4.2 * len(hist), 3.2))
    for ax, (n, h) in zip(np.atleast_1d(axs), hist):
        ax.plot(h["train_loss"], label="training loss")
        if h.get("val_loss"):
            ax.plot(h["val_loss"], label="validation loss (10% hold-out)")
        if h.get("val_score"):
            ax2 = ax.twinx(); ax2.plot(h["val_score"], color="#e08a1e", label="validation accuracy"); ax2.set_ylabel("val. accuracy", fontsize=8); ax2.grid(False)
        ax.set_xlabel("epoch / boosting iteration"); ax.set_ylabel("log loss"); ax.set_title(NICE.get(n, n), fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "training_histories.png"); plt.close(fig)
print("figures ->", FIG)
