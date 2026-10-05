"""Classification: High vs Normal demand sessions."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C
from .features import make_preprocessor

SLOW = {"KNN", "SVM"}


def make_target(y_train: pd.Series, y_test: pd.Series):
    """High Demand (1) if Demand >= median of TRAIN set. Threshold never sees test data."""
    thr = float(y_train.quantile(C.HIGH_DEMAND_QUANTILE))
    return (y_train >= thr).astype(int), (y_test >= thr).astype(int), thr


def class_balance(y: pd.Series) -> dict:
    share = y.value_counts(normalize=True).sort_index()
    minority = float(share.min())
    return {"shares": share.to_dict(), "minority_share": minority,
            "is_imbalanced": minority < C.IMBALANCE_THRESHOLD}


def get_models(fast: bool = False) -> dict:
    n = 60 if fast else 200
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=n // 2, max_depth=14, n_jobs=-1, random_state=rs),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0),
        "XGBoost": XGBClassifier(n_estimators=n, learning_rate=0.08, max_depth=6, n_jobs=-1, random_state=rs),
        "LightGBM": LGBMClassifier(n_estimators=n, learning_rate=0.08, num_leaves=48, random_state=rs, verbose=-1),
    }


def evaluate(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    score = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_test)
    return {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0), "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, score)}


def _subsample(X, y, n):
    if len(X) <= n:
        return X, y
    idx = X.sample(n, random_state=C.RANDOM_STATE).index
    return X.loc[idx], y.loc[idx]


def run_classification(X_train, y_train, X_test, y_test, fast: bool = False):
    results, fitted = [], {}
    for name, model in get_models(fast).items():
        pipe = ImbPipeline([("pre", make_preprocessor()), ("model", model)])
        Xt, yt = _subsample(X_train, y_train, C.SLOW_MODEL_MAX_ROWS) if name in SLOW else (X_train, y_train)
        pipe.fit(Xt, yt)
        results.append({"Model": name, **evaluate(pipe, X_test, y_test)})
        fitted[name] = pipe
        print(f"  [classification] {name:<20} done")
    table = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return table, fitted


def imbalance_experiment(X_train, y_train, X_test, y_test, fast: bool = False) -> pd.DataFrame:
    """Compare no-resampling vs ROS / SMOTE / ADASYN using LightGBM.
    Resampling happens INSIDE the pipeline so it only touches training data."""
    samplers = {"None": None, "Random Oversampling": RandomOverSampler(random_state=C.RANDOM_STATE),
                "SMOTE": SMOTE(random_state=C.RANDOM_STATE), "ADASYN": ADASYN(random_state=C.RANDOM_STATE)}
    rows = []
    for name, sampler in samplers.items():
        steps = [("pre", make_preprocessor())]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("model", LGBMClassifier(n_estimators=60 if fast else 200, random_state=C.RANDOM_STATE, verbose=-1)))
        try:
            pipe = ImbPipeline(steps).fit(X_train, y_train)
            rows.append({"Strategy": name, **evaluate(pipe, X_test, y_test)})
        except Exception as exc:  # e.g. ADASYN cannot find neighbours on balanced data
            print(f"  [imbalance] {name} skipped: {exc}")
    return pd.DataFrame(rows)
