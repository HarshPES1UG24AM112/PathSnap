"""Model zoo: seven candidate scorers from distinct algorithm families.

All models are binary classifiers P(candidate arc is the reference arc | features); their
probabilities are later renormalised within each observation's candidate set and used as
HMM emission scores. Scaling is inside each pipeline so it is fitted on training folds only.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# name -> (family, params, max training rows (None = all), builder)
MODEL_SPECS = {
    "logreg": ("Linear (probabilistic)", dict(C=1.0, max_iter=500, solver="lbfgs")),
    "gaussian_nb": ("Probabilistic generative", dict(var_smoothing=1e-9)),
    "knn": ("Instance-based", dict(n_neighbors=50, weights="distance", train_rows=60000)),
    "decision_tree": ("Tree", dict(max_depth=12, min_samples_leaf=50)),
    "random_forest": ("Bagging ensemble", dict(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features="sqrt")),
    "hist_gbm": ("Boosting ensemble", dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=0.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)),
    "mlp": ("Neural network", dict(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                                   max_iter=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6)),
}


def build_model(name: str, params: dict | None = None):
    family, default = MODEL_SPECS[name]
    p = dict(default)
    p.update(params or {})
    p.pop("train_rows", None)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **p))
    if name == "gaussian_nb":
        # quantile->normal transform makes heavy-tailed distances closer to NB's Gaussian assumption
        return make_pipeline(QuantileTransformer(output_distribution="normal", n_quantiles=200, subsample=200000, random_state=SEED), GaussianNB(**p))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1, algorithm="brute", **p))
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=SEED, **p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **p)
    if name == "hist_gbm":
        return HistGradientBoostingClassifier(random_state=SEED, scoring="loss", **p)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=SEED, **p))
    raise KeyError(name)


def train_rows_cap(name: str, params: dict | None = None):
    p = dict(MODEL_SPECS[name][1])
    p.update(params or {})
    return p.get("train_rows")
