"""Explainability: SHAP and LIME."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from . import config as C


def transform(pipe, X):
    return pipe.named_steps["pre"].transform(X)


def shap_analysis(pipe, X_sample: pd.DataFrame, out_prefix: str, max_rows: int = 1500):
    """Save SHAP bar + beeswarm plots; return mean |SHAP| table. Tree models only."""
    X_sample = X_sample.sample(min(max_rows, len(X_sample)), random_state=C.RANDOM_STATE)
    names = pipe.named_steps["pre"].get_feature_names_out()
    Xt = pd.DataFrame(transform(pipe, X_sample), columns=names)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older shap: list per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap: (n, features, classes)
        sv = sv[:, :, 1]
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(); shap.summary_plot(sv, Xt, plot_type="bar", show=False, max_display=15)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / f"{out_prefix}_shap_bar.png", dpi=130); plt.close()
    plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / f"{out_prefix}_shap_beeswarm.png", dpi=130); plt.close()
    table = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def make_lime_explainer(pipe, X_train: pd.DataFrame, mode: str, max_rows: int = 2000) -> LimeTabularExplainer:
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    data = transform(pipe, X_train.sample(min(max_rows, len(X_train)), random_state=C.RANDOM_STATE))
    return LimeTabularExplainer(data, feature_names=names, mode=mode,
                                class_names=["Normal", "High"] if mode == "classification" else None,
                                discretize_continuous=True, random_state=C.RANDOM_STATE)


def lime_explain(pipe, explainer: LimeTabularExplainer, X_row: pd.DataFrame, mode: str, num_features: int = 10):
    """Return LIME (feature rule, weight) pairs for one row."""
    x = transform(pipe, X_row)[0]
    model = pipe.named_steps["model"]
    fn = model.predict_proba if mode == "classification" else model.predict
    exp = explainer.explain_instance(x, fn, num_features=num_features, num_samples=1000)
    return pd.DataFrame(exp.as_list(), columns=["feature_rule", "weight"])
