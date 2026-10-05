"""EV Charging Demand Prediction & Usage Pattern Analysis - Streamlit dashboard.
Run:  streamlit run app.py   (after:  python train_all.py)
"""
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src import classification as clf
from src import config as C
from src import data_prep as dp
from src import explain as xai
from src import features as fe
from src import timeseries as ts

st.set_page_config(page_title="EV Charging Analytics", page_icon="⚡", layout="wide")


# ---------- cached loaders ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    return dp.load_processed()


@st.cache_data
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(C.REPORT_DIR / name)


@st.cache_resource
def load_model(name: str):
    return joblib.load(C.MODEL_DIR / name)


@st.cache_data
def load_meta() -> dict:
    return json.loads((C.MODEL_DIR / "metadata.json").read_text())


@st.cache_resource
def lime_explainer(model_file: str, mode: str):
    bundle = load_model("explain_bundle.joblib")
    return xai.make_lime_explainer(load_model(model_file), bundle["X_train_sample"], mode)


if not (C.PROCESSED_PATH.exists() and (C.MODEL_DIR / "metadata.json").exists()):
    st.error("Artifacts not found. Place the CSV in data/raw/charging_record.csv and run `python train_all.py` first.")
    st.stop()

df, meta = load_data(), load_meta()

# ---------- sidebar filters ----------
st.sidebar.title("⚡ Filters")
dmin, dmax = df["StartDatetime"].min().date(), df["StartDatetime"].max().date()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
locs = st.sidebar.multiselect("Location", sorted(df["Location"].unique()))
types = st.sidebar.multiselect("Charger type", sorted(df["ChargerType"].unique()))
comps = st.sidebar.multiselect("Charger company", sorted(df["ChargerCompany"].unique()))

f = df.copy()
if len(date_range) == 2:
    f = f[(f["StartDatetime"].dt.date >= date_range[0]) & (f["StartDatetime"].dt.date <= date_range[1])]
if locs:
    f = f[f["Location"].isin(locs)]
if types:
    f = f[f["ChargerType"].isin(types)]
if comps:
    f = f[f["ChargerCompany"].isin(comps)]
if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.title("⚡ EV Charging Demand Prediction & Usage Pattern Analysis")
tabs = st.tabs(["📊 Overview", "🔎 EDA", "📈 Regression", "🎯 Classification",
                "⏱️ Time Series", "🏆 Model Comparison", "🧠 Explainability"])

# ---------- Overview ----------
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sessions", f"{len(f):,}")
    c2.metric("Total demand", f"{f['Demand'].sum():,.0f}")
    c3.metric("Avg demand / session", f"{f['Demand'].mean():.2f}")
    c4.metric("Avg duration (h)", f"{f['Duration'].mean():.2f}")
    c5.metric("Active chargers", f"{f['ChargerID'].nunique():,}")
    a, b = st.columns(2)
    by_loc = f.groupby("Location")["Demand"].sum().sort_values().reset_index()
    a.plotly_chart(px.bar(by_loc, x="Demand", y="Location", orientation="h", title="Total demand by location"),
                   width="stretch")
    by_type = f.groupby("ChargerType")["Demand"].agg(["sum", "mean", "count"]).reset_index()
    b.plotly_chart(px.pie(by_type, names="ChargerType", values="sum", title="Demand share by charger type", hole=0.4),
                   width="stretch")
    by_comp = f.groupby("ChargerCompany")["Demand"].agg(["sum", "mean", "count"]).reset_index()
    st.plotly_chart(px.bar(by_comp, x="ChargerCompany", y="sum", color="mean", title="Demand by company (colour = avg per session)"),
                    width="stretch")

# ---------- EDA ----------
with tabs[1]:
    g = f.copy()
    g["hour"], g["dow"] = g["StartDatetime"].dt.hour, g["StartDatetime"].dt.day_name()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(g.groupby("hour")["Demand"].mean().reset_index(), x="hour", y="Demand",
                          title="Average demand by hour of day"), width="stretch")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = g.groupby("dow")["Demand"].mean().reindex(order).reset_index()
    b.plotly_chart(px.bar(dow, x="dow", y="Demand", title="Average demand by weekday"), width="stretch")
    heat = g.pivot_table(index="dow", columns="hour", values="Demand", aggfunc="mean").reindex(order)
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                              title="Demand heatmap: weekday × hour"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(g, x="Demand", nbins=60, title="Demand distribution"), width="stretch")
    b.plotly_chart(px.box(g, x="ChargerType", y="Demand", color="ChargerType", title="Demand by charger type (outliers)"),
                   width="stretch")
    util = g.groupby("ChargerID")["Demand"].agg(["count", "sum"]).sort_values("count").reset_index()
    st.subheader("Charger utilisation: least-used chargers")
    st.dataframe(util.head(10), width="stretch")

# ---------- Regression ----------
with tabs[2]:
    st.subheader("Predict charging demand for a session")
    best = load_model("regression_best.joblib")
    c1, c2, c3 = st.columns(3)
    charger = c1.selectbox("Charger ID", sorted(df["ChargerID"].unique()))
    last = df[df["ChargerID"] == charger].iloc[-1]
    company = c2.selectbox("Company", sorted(df["ChargerCompany"].unique()),
                           index=sorted(df["ChargerCompany"].unique()).index(last["ChargerCompany"]))
    location = c3.selectbox("Location ", sorted(df["Location"].unique()),
                            index=sorted(df["Location"].unique()).index(last["Location"]))
    c1, c2, c3 = st.columns(3)
    ctype = c1.selectbox("Charger type ", sorted(df["ChargerType"].unique()),
                         index=sorted(df["ChargerType"].unique()).index(last["ChargerType"]))
    d = c2.date_input("Charging date", df["StartDatetime"].max().date() + pd.Timedelta(days=1))
    t = c3.time_input("Start time", pd.Timestamp("18:00").time())
    if st.button("Predict demand", type="primary"):
        row = fe.build_inference_row(df, charger, pd.Timestamp.combine(d, t), company, location, ctype)
        X, _ = fe.get_Xy(row.assign(Demand=0))
        pred = float(best.predict(X)[0])
        st.success(f"Predicted demand: **{pred:.2f}** (average session: {df['Demand'].mean():.2f})")
    st.subheader("Regression metrics (chronological test set)")
    st.dataframe(load_report("regression_results.csv").round(4), width="stretch")

# ---------- Classification ----------
with tabs[3]:
    st.subheader("High-demand session prediction")
    thr = meta["high_demand_threshold"]
    st.info(f"High Demand = Demand ≥ **{thr:.2f}** (median of training data). "
            f"Class shares: { {k: round(v, 3) for k, v in meta['class_balance'].items()} }")
    best_c = load_model("classification_best.joblib")
    c1, c2, c3 = st.columns(3)
    charger_c = c1.selectbox("Charger ID", sorted(df["ChargerID"].unique()), key="cc")
    d2 = c2.date_input("Date", df["StartDatetime"].max().date() + pd.Timedelta(days=1), key="cd")
    t2 = c3.time_input("Start time", pd.Timestamp("18:00").time(), key="ct")
    if st.button("Classify session"):
        row = fe.build_inference_row(df, charger_c, pd.Timestamp.combine(d2, t2))
        X, _ = fe.get_Xy(row.assign(Demand=0))
        if hasattr(best_c, "predict_proba"):
            p = float(best_c.predict_proba(X)[0, 1])
            label = "High Demand" if p >= 0.5 else "Normal Demand"
            st.success(f"**{label}**, P(high) = {p:.1%}")
        else:
            st.success("High Demand" if best_c.predict(X)[0] else "Normal Demand")
    st.dataframe(load_report("classification_results.csv").round(4), width="stretch")
    st.subheader("Class-imbalance strategies")
    st.dataframe(load_report("imbalance_comparison.csv").round(4), width="stretch")

# ---------- Time Series ----------
with tabs[4]:
    freq = st.radio("Aggregation", ["D", "H"], format_func=lambda x: "Daily" if x == "D" else "Hourly", horizontal=True)
    series = ts.to_series(f, freq)
    window = st.slider("Moving-average window", 3, 30 if freq == "D" else 72, 7)
    ma = ts.rolling_moving_average(series, window)
    fig = px.line(series.reset_index(), x="StartDatetime", y="Demand", title="Demand over time")
    fig.add_scatter(x=ma.index, y=ma.values, name=f"MA({window})", line=dict(width=3))
    st.plotly_chart(fig, width="stretch")
    if len(series) > 60:
        dec = ts.decompose(series, freq)
        for name, comp in [("Trend", dec.trend), ("Seasonality", dec.seasonal), ("Residual (noise)", dec.resid)]:
            st.plotly_chart(px.line(comp.reset_index(), x="StartDatetime", y=comp.name, title=name, height=250),
                            width="stretch")
        adf = ts.adf_test(series)
        st.write(f"**ADF test** p-value = {adf['p-value']:.4f} → series is "
                 f"{'stationary' if adf['stationary'] else 'NON-stationary (trend/seasonality present)'}")
        hold = min(C.TS_HOLDOUT, len(series) // 4)
        table, test, preds = ts.evaluate_forecasts(series, hold, freq)
        st.dataframe(table.round(3), width="stretch")
        best_ts = table.iloc[0]["Model"]
        fut = ts.future_forecast(series, best_ts, C.TS_HORIZON, freq)
        fig = px.line(series.tail(120 if freq == "D" else 500).reset_index(), x="StartDatetime", y="Demand",
                      title=f"Forecast ({best_ts})")
        fig.add_scatter(x=fut["Datetime"], y=fut["Forecast"], name="Forecast", line=dict(dash="dash", width=3))
        st.plotly_chart(fig, width="stretch")
        if freq == "D":
            tp = ts.trend_periods(series)
            st.subheader("Increasing / decreasing demand periods")
            st.dataframe(tp.assign(**{c: tp[c].round(2) for c in ["Demand", "Change %"]}), width="stretch")
    else:
        st.info("Not enough data points for decomposition/forecasting with the current filters.")

# ---------- Model comparison ----------
with tabs[5]:
    r = load_report("regression_results.csv")
    c = load_report("classification_results.csv")
    t = load_report("timeseries_results.csv")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(r, x="Model", y="RMSE", title="Regression RMSE (lower is better)"), width="stretch")
    b.plotly_chart(px.bar(c, x="Model", y="ROC_AUC", title="Classification ROC-AUC"), width="stretch")
    st.plotly_chart(px.bar(t, x="Model", y=["MAE", "RMSE"], barmode="group", title="Forecasting error"),
                    width="stretch")
    st.write(f"🏅 Best regressor: **{meta['best_regression']}** | best classifier: **{meta['best_classifier']}** "
             f"| best forecaster: **{meta['best_timeseries']}**")

# ---------- Explainability ----------
with tabs[6]:
    st.subheader("Feature importance & SHAP")
    a, b = st.columns(2)
    imp = load_report("regression_importance.csv").head(15)
    a.plotly_chart(px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h",
                          title=f"Regression importance ({meta['tree_regression']})"), width="stretch")
    shap_c = load_report("classification_shap.csv").head(15)
    b.plotly_chart(px.bar(shap_c.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                          title=f"Classifier mean |SHAP| ({meta['tree_classifier']})"), width="stretch")
    for prefix in ("regression", "classification"):
        p = C.FIG_DIR / f"{prefix}_shap_beeswarm.png"
        if p.exists():
            st.image(str(p), caption=f"SHAP beeswarm – {prefix}")
    st.subheader("LIME: explain a single prediction")
    bundle = load_model("explain_bundle.joblib")
    mode_label = st.radio("Model", ["Regression", "Classification"], horizontal=True)
    idx = st.slider("Test-set sample index", 0, len(bundle["X_test_sample"]) - 1, 0)
    row = bundle["X_test_sample"].iloc[[idx]]
    mode, fname = ("regression", "regression_tree.joblib") if mode_label == "Regression" else ("classification", "classification_tree.joblib")
    if st.button("Explain with LIME"):
        with st.spinner("Computing LIME explanation..."):
            pipe = load_model(fname)
            out = xai.lime_explain(pipe, lime_explainer(fname, mode), row, mode)
        out["effect"] = out["weight"].apply(lambda w: "↑ increases" if w > 0 else "↓ decreases")
        st.plotly_chart(px.bar(out.iloc[::-1], x="weight", y="feature_rule", orientation="h", color="effect"),
                        width="stretch")
        st.caption(f"Actual demand for this session: {bundle['y_test_sample'].iloc[idx]:.2f}")
