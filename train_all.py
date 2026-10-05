"""End-to-end pipeline: clean -> features -> regression -> classification
-> time series -> explainability. Saves models/ and reports/ for the dashboard.

Usage:
    python train_all.py                # full run
    python train_all.py --fast         # smaller models for a quick test
"""
import argparse
import json

import joblib
import pandas as pd

from src import classification as clf
from src import config as C
from src import data_prep as dp
from src import explain as xai
from src import features as fe
from src import regression as reg
from src import timeseries as ts


def main(fast: bool = False) -> None:
    C.MODEL_DIR.mkdir(exist_ok=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data understanding & cleaning
    raw = dp.load_raw()
    print("Raw overview:", {k: v for k, v in dp.data_overview(raw).items() if k in ("rows", "duplicates")})
    df = dp.clean(raw)
    dp.save_processed(df)
    print(f"Clean rows: {len(df)}")

    # 2. Feature engineering (history features use only past sessions)
    feat = fe.build_features(df)
    train, test = fe.chronological_split(feat)
    X_tr, y_tr = fe.get_Xy(train)
    X_te, y_te = fe.get_Xy(test)
    cutoff = str(test["StartDatetime"].min())
    print(f"Train {len(train)} | Test {len(test)} | split at {cutoff}")

    # 3. Regression
    reg_table, reg_models = reg.run_regression(X_tr, y_tr, X_te, y_te, fast)
    reg_table.to_csv(C.REPORT_DIR / "regression_results.csv", index=False)
    best_reg = reg_table.iloc[0]["Model"]
    tree_reg = reg_table[reg_table["Model"].isin(reg.TREE_MODELS)].iloc[0]["Model"]
    reg.feature_importance(reg_models[tree_reg]).to_csv(C.REPORT_DIR / "regression_importance.csv", index=False)
    joblib.dump(reg_models[best_reg], C.MODEL_DIR / "regression_best.joblib")
    joblib.dump(reg_models[tree_reg], C.MODEL_DIR / "regression_tree.joblib")
    print(reg_table.round(4).to_string(index=False))

    # 4. Classification (threshold from TRAIN median only)
    yc_tr, yc_te, thr = clf.make_target(y_tr, y_te)
    balance = clf.class_balance(yc_tr)
    print("Class balance:", balance)
    imb_table = clf.imbalance_experiment(X_tr, yc_tr, X_te, yc_te, fast)
    imb_table.to_csv(C.REPORT_DIR / "imbalance_comparison.csv", index=False)
    clf_table, clf_models = clf.run_classification(X_tr, yc_tr, X_te, yc_te, fast)
    clf_table.to_csv(C.REPORT_DIR / "classification_results.csv", index=False)
    best_clf = clf_table.iloc[0]["Model"]
    tree_clf = clf_table[clf_table["Model"].isin(
        ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"])].iloc[0]["Model"]
    joblib.dump(clf_models[best_clf], C.MODEL_DIR / "classification_best.joblib")
    joblib.dump(clf_models[tree_clf], C.MODEL_DIR / "classification_tree.joblib")
    print(clf_table.round(4).to_string(index=False))
    print(imb_table.round(4).to_string(index=False))

    # 5. Time series
    series = ts.to_series(df)
    adf = ts.adf_test(series)
    ts_table, _, _ = ts.evaluate_forecasts(series)
    ts_table.to_csv(C.REPORT_DIR / "timeseries_results.csv", index=False)
    best_ts = ts_table.iloc[0]["Model"]
    ts.future_forecast(series, best_ts).to_csv(C.REPORT_DIR / "future_forecast.csv", index=False)
    ts.trend_periods(series).to_csv(C.REPORT_DIR / "trend_periods.csv", index=False)
    print(ts_table.round(3).to_string(index=False), f"\nADF p-value={adf['p-value']:.4f}")

    # 6. Explainability (SHAP on best tree models)
    xai.shap_analysis(reg_models[tree_reg], X_te, "regression").to_csv(C.REPORT_DIR / "regression_shap.csv", index=False)
    xai.shap_analysis(clf_models[tree_clf], X_te, "classification").to_csv(C.REPORT_DIR / "classification_shap.csv", index=False)
    sample = X_te.sample(min(300, len(X_te)), random_state=C.RANDOM_STATE)
    joblib.dump({"X_train_sample": X_tr.sample(min(2000, len(X_tr)), random_state=C.RANDOM_STATE),
                 "X_test_sample": sample,
                 "y_test_sample": y_te.loc[sample.index]}, C.MODEL_DIR / "explain_bundle.joblib")

    meta = {"split_cutoff": cutoff, "high_demand_threshold": thr, "class_balance": {str(k): v for k, v in balance["shares"].items()},
            "best_regression": best_reg, "tree_regression": tree_reg, "best_classifier": best_clf,
            "tree_classifier": tree_clf, "best_timeseries": best_ts, "adf_p_value": adf["p-value"],
            "stationary": adf["stationary"], "use_duration_feature": C.USE_DURATION_AS_FEATURE}
    (C.MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("\nAll artifacts saved. Run:  streamlit run app.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    main(p.parse_args().fast)
