"""Regression: charging demand prediction."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import config as C
from .features import make_preprocessor

TREE_MODELS = {"Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"}


def get_models(fast: bool = False) -> dict:
    n = 60 if fast else 200
    rs = C.RANDOM_STATE
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=rs),
        "Random Forest": RandomForestRegressor(n_estimators=n // 2, max_depth=14, n_jobs=-1, random_state=rs),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=n // 2, random_state=rs),
        "XGBoost": XGBRegressor(n_estimators=n, learning_rate=0.08, max_depth=6, n_jobs=-1, random_state=rs),
        "LightGBM": LGBMRegressor(n_estimators=n, learning_rate=0.08, num_leaves=48, random_state=rs, verbose=-1),
    }


def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def run_regression(X_train, y_train, X_test, y_test, fast: bool = False):
    results, fitted = [], {}
    for name, model in get_models(fast).items():
        pipe = Pipeline([("pre", make_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        results.append({"Model": name, **evaluate(y_test, pipe.predict(X_test))})
        fitted[name] = pipe
        print(f"  [regression] {name:<18} done")
    table = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return table, fitted


def feature_importance(pipe: Pipeline, top: int = 20) -> pd.DataFrame:
    """Native importance (tree models) or |coef| (linear)."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    model = pipe.named_steps["model"]
    imp = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_)
    out = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    return out.head(top).reset_index(drop=True)
