"""Shared helpers for the drowsiness-detection notebooks (03 and 04)."""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

LABEL_NAMES = ["alert", "low vigilant", "drowsy"]

# ---------------------------------------------------------------- data
def load_windows(csv="data/windows.csv", groups_json="data/feature_groups.json"):
    df = pd.read_csv(csv, dtype={"subject": str})
    with open(groups_json) as f:
        groups = json.load(f)
    return df, groups


def feature_list(groups, names):
    return [f for g in names for f in groups[g]]


# ---------------------------------------------------------------- models
DEFAULTS = {
    "rf": dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features="sqrt"),
    "xgb": dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8),
    "mlp": dict(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3),
}


def _pipe(name, params, quick, seed):
    p = {**DEFAULTS[name], **(params or {})}
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    if quick and "n_estimators" in p:
        p["n_estimators"] = min(int(p["n_estimators"]), 60)
    steps = [("imp", SimpleImputer(strategy="median"))]
    if name == "rf":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    elif name == "xgb":
        clf = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                            n_jobs=1, random_state=seed, **p)
    elif name == "mlp":
        steps.append(("sc", StandardScaler()))
        clf = MLPClassifier(max_iter=120 if quick else 400, early_stopping=True,
                            n_iter_no_change=15, random_state=seed, **p)
    else:
        raise ValueError(name)
    steps.append(("clf", clf))
    return Pipeline(steps)


def make_models(params=None, quick=False, seed=0):
    """Returns {'rf','xgb','mlp','ensemble'}. params = {'rf': {...}, 'xgb': {...}, 'mlp': {...}}."""
    params = params or {}
    base = {n: _pipe(n, params.get(n), quick, seed) for n in ("rf", "xgb", "mlp")}
    ens = VotingClassifier([(n, clone(m)) for n, m in base.items()], voting="soft")
    return {**base, "ensemble": ens}


# ---------------------------------------------------------------- evaluation
def cv_proba(model, X, y, groups, cv, n_jobs=1):
    """Out-of-fold class probabilities. `groups` decides what is never shared between train and test."""
    return cross_val_predict(clone(model), X, y, groups=groups, cv=cv,
                             method="predict_proba", n_jobs=n_jobs)


def window_metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), macro_precision=p, macro_recall=r, macro_f1=f)


def video_level(df, proba):
    """Video-level accuracy by (a) majority vote of window predictions, (b) mean class probability."""
    vid = df["video"].values
    y_vid = df.groupby("video")["y"].first()
    pred = pd.Series(proba.argmax(1), index=df.index)
    vote = pred.groupby(vid).agg(lambda s: s.value_counts().idxmax())
    mean_p = pd.DataFrame(proba, index=df.index).groupby(vid).mean().values.argmax(1)
    mean_p = pd.Series(mean_p, index=sorted(set(vid)))
    return dict(video_acc_vote=accuracy_score(y_vid.loc[vote.index], vote),
                video_acc_meanproba=accuracy_score(y_vid.loc[mean_p.index], mean_p))


def subject_bootstrap_ci(df, pred, n_boot=1000, seed=0):
    """95% CI of window accuracy, resampling SUBJECTS (not windows) with replacement."""
    d = pd.DataFrame({"s": df["subject"].values, "c": (pred == df["y"].values).astype(float)})
    g = d.groupby("s")["c"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    accs = g["sum"].values[idx].sum(1) / g["count"].values[idx].sum(1)
    return np.percentile(accs, [2.5, 97.5])


def paired_subject_bootstrap(df, pred_a, pred_b, n_boot=2000, seed=0):
    """Accuracy(A) - Accuracy(B) with a 95% CI from resampling SUBJECTS. Same windows, so this is a paired comparison."""
    y = df["y"].values
    d = pd.DataFrame({"s": df["subject"].values, "a": (pred_a == y).astype(float), "b": (pred_b == y).astype(float)})
    g = d.groupby("s").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "count"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    diff = (g["a"].values[idx].sum(1) - g["b"].values[idx].sum(1)) / g["n"].values[idx].sum(1)
    obs = (g["a"].sum() - g["b"].sum()) / g["n"].sum()
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def summarize(results, df, protocol):
    """results: {model_name: out-of-fold proba}. Returns a tidy table."""
    rows = []
    for name, P in results.items():
        pred = P.argmax(1)
        r = window_metrics(df["y"].values, pred)
        r.update(video_level(df, P))
        lo, hi = subject_bootstrap_ci(df, pred)
        r.update(model=name, protocol=protocol, acc_ci95=f"[{lo:.3f}, {hi:.3f}]")
        rows.append(r)
    cols = ["protocol", "model", "accuracy", "acc_ci95", "macro_precision", "macro_recall",
            "macro_f1", "video_acc_vote", "video_acc_meanproba"]
    return pd.DataFrame(rows)[cols]


def plot_cm(y, pred, ax, title):
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3), LABEL_NAMES, rotation=20)
    ax.set_yticks(range(3), LABEL_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    return cm
