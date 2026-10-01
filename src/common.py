"""Shared configuration, data loading, model factories and metrics.

Used by part1_reproduce.py (reproduction of Bhagat et al., 2025) and
part2_proposed.py (the proposed leakage-free, diversity-pruned stacking).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

SEED = 42
TARGET = "target"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
# Feature typing used by the PROPOSED method (Part 2)
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ORDINAL = ["ca"]                      # number of vessels 0-3 (ordered count)
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
# Undocumented sentinel codes = original UCI missing values ('?')
SENTINELS = {"ca": 4, "thal": 0}

MODEL_ORDER = ["LR", "NB", "KNN", "DT", "RF", "XGB", "Stacking"]

# ---------------------------------------------------------------- XGBoost
try:
    import xgboost
    from xgboost import XGBClassifier
    XGB_BACKEND = f"xgboost {xgboost.__version__}"
    HAS_XGB = True
except ImportError:  # pragma: no cover - depends on environment
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_BACKEND = "sklearn HistGradientBoostingClassifier (xgboost not installed)"
    HAS_XGB = False


def make_xgb(seed: int = SEED, **kw):
    """Extreme Gradient Boosting with library defaults (paper gives none)."""
    if HAS_XGB:
        return XGBClassifier(eval_metric="logloss", random_state=seed,
                             n_jobs=1, verbosity=0, **kw)
    return HistGradientBoostingClassifier(random_state=seed, **kw)


# ---------------------------------------------------------------- data
def load_data(dedup: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def data_audit(df: pd.DataFrame) -> dict:
    feats = df[FEATURES]
    u = df.drop_duplicates()
    group_sizes = df.groupby(FEATURES + [TARGET]).size()
    conflicting = int((df.groupby(FEATURES)[TARGET].nunique() > 1).sum())
    return {
        "n_rows": int(len(df)), "n_features": len(FEATURES),
        "n_unique": int(len(u)), "n_duplicates": int(df.duplicated().sum()),
        "copies_per_patient": {int(k): int(v) for k, v in group_sizes.value_counts().sort_index().items()},
        "conflicting_label_groups": conflicting,
        "class_counts_full": {int(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "class_counts_unique": {int(k): int(v) for k, v in u[TARGET].value_counts().sort_index().items()},
        "explicit_missing": int(feats.isna().sum().sum()),
        "ca_eq_4_rows": int((df.ca == 4).sum()), "ca_eq_4_unique": int((u.ca == 4).sum()),
        "thal_eq_0_rows": int((df.thal == 0).sum()), "thal_eq_0_unique": int((u.thal == 0).sum()),
    }


# ---------------------------------------------------------------- Part 1: paper pipeline
def paper_preprocessor():
    """Paper Sec. 3.1: linear-regression imputation + normalisation.

    The CSV has no NaNs, so the imputer is a faithful no-op; features stay
    label-coded (13 columns), matching the paper's Fig. 2 feature set.
    """
    return Pipeline([
        ("impute", IterativeImputer(estimator=LinearRegression(), random_state=SEED)),
        ("scale", StandardScaler()),
    ])


def paper_base_learners(seed: int = SEED) -> dict:
    """Six base classifiers; paper reports no hyper-parameters -> library defaults."""
    return {
        "LR": LogisticRegression(max_iter=1000),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(random_state=seed, n_jobs=1),
        "XGB": make_xgb(seed),
    }


def paper_stacking(seed: int = SEED, meta=None, cv=5):
    """5-fold stacking of all six base learners (paper Table 8 / Sec. 5)."""
    meta = meta if meta is not None else LogisticRegression(max_iter=1000)
    return StackingClassifier(
        estimators=list(paper_base_learners(seed).items()),
        final_estimator=meta, cv=cv, n_jobs=1)


def paper_models(seed: int = SEED) -> dict:
    models = {k: Pipeline([("prep", paper_preprocessor()), ("clf", v)])
              for k, v in paper_base_learners(seed).items()}
    models["Stacking"] = Pipeline([("prep", paper_preprocessor()),
                                   ("clf", paper_stacking(seed))])
    return models


# ---------------------------------------------------------------- Part 2: type-aware preprocessing
def _sentinel_to_nan(X):
    X = X.copy()
    for col, code in SENTINELS.items():
        X[col] = X[col].where(X[col] != code, np.nan)
    return X


def typed_preprocessor():
    """Sentinel repair + feature-type-aware encoding, fitted inside each fold."""
    ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), CONTINUOUS + ORDINAL),
        ("bin", "passthrough", BINARY),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), NOMINAL),
    ])
    return Pipeline([
        ("sentinel", FunctionTransformer(_sentinel_to_nan, feature_names_out="one-to-one")),
        ("ct", ct),
    ])


# ---------------------------------------------------------------- metrics
def evaluate(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_score),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Specificity", "MCC"]

# Values reported by Bhagat et al. (2025), Table 11 (Recall/Precision/F1/AUC/Acc
# columns; the paper's Sensitivity/Specificity/MCC columns are copy errors).
PAPER = {
    "LR": {"Accuracy": .8439, "Precision": .8196, "Recall": .9090, "F1": .8620, "AUC": .9204},
    "NB": {"Accuracy": .8439, "Precision": .8250, "Recall": .9000, "F1": .8608, "AUC": .9185},
    "KNN": {"Accuracy": .8585, "Precision": .8648, "Recall": .8727, "F1": .8687, "AUC": .9304},
    "DT": {"Accuracy": .9268, "Precision": .9702, "Recall": .8909, "F1": .9289, "AUC": .9702},
    "RF": {"Accuracy": .9268, "Precision": .9130, "Recall": .9545, "F1": .9333, "AUC": .9734},
    "XGB": {"Accuracy": .9073, "Precision": .9174, "Recall": .9090, "F1": .9132, "AUC": .9830},
    "Stacking": {"Accuracy": .9853, "Precision": 1.0, "Recall": .9727, "F1": .9861, "AUC": .9880},
}


def save_json(obj, name):
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
