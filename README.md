# ⚡ EV Charging Demand Prediction & Usage Pattern Analysis
HCL GUVI Capstone | Domain: Electric Vehicles & Sustainable Transportation

End-to-end ML system that analyses EV charging sessions and delivers:
1. **Regression** – predict the `Demand` of a charging session
2. **Classification** – High vs Normal demand (High = Demand ≥ median of training data)
3. **Time series** – trend / seasonality / stationarity, Moving Average & Exponential Smoothing forecasts
4. **Explainability** – feature importance, SHAP, LIME
5. **Streamlit dashboard** – KPIs, EDA, predictions, forecasts, model comparison, explanations

## Project structure
```
ev_charging_project/
├── data/raw/charging_record.csv      <- put the real dataset here
├── data/processed/                   <- cleaned data (auto-generated)
├── src/                              <- config, data_prep, features, regression,
│                                        classification, timeseries, explain
├── scripts/generate_sample_data.py   <- synthetic data (smoke test only)
├── train_all.py                      <- runs the whole pipeline
├── app.py                            <- Streamlit dashboard
├── models/  reports/                 <- saved models, metric tables, SHAP figures
└── requirements.txt
```

## Setup & run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. place the real CSV at data/raw/charging_record.csv
#    (no dataset yet? `python scripts/generate_sample_data.py` makes a synthetic one)
# 2. train everything (add --fast for a quick test)
python train_all.py
# 3. launch the dashboard
streamlit run app.py
```

## Methodology
| Stage | What is done |
|---|---|
| Cleaning | duplicates removed, datetimes parsed (falls back to Day+Time), invalid/negative demand dropped, categorical gaps filled, outliers flagged (IQR) and winsorised at 99.5th percentile |
| Features | hour, day, month, weekday, weekend flag, week of year, cyclical hour (sin/cos), time-of-day bucket; history features per charger/user (previous demand, rolling mean of last 5 sessions, session index, hours since last session) |
| Leakage control | **chronological** 80/20 split (train on past, test on future); history features use `shift(1)` so they only see earlier sessions; High-Demand threshold computed on **train only**; resampling happens inside the pipeline (train data only); `Duration` excluded by default because it is only known after the session (toggle `USE_DURATION_AS_FEATURE` in `src/config.py`) |
| Regression | Linear, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM → MAE, MSE, RMSE, R² |
| Classification | Logistic, Decision Tree, Random Forest, KNN, SVM, XGBoost, LightGBM → Accuracy, Precision, Recall, F1, ROC-AUC; class balance check; Random Oversampling / SMOTE / ADASYN comparison |
| Time series | daily (or hourly) aggregation, decomposition, ADF stationarity test, Moving Average, Simple Exp. Smoothing, Holt-Winters; MAE & RMSE on a 30-period hold-out; 30-period future forecast; increasing/decreasing monthly periods |
| Explainability | native importance, SHAP (bar + beeswarm) for the best tree models, interactive LIME in the dashboard |

## Notes
* Because the median is the threshold, classes are ≈50/50 on real data; the imbalance table documents that resampling is not required (and is applied automatically-by-comparison if you change `HIGH_DEMAND_QUANTILE`, e.g. 0.8).
* KNN and SVM train on a 20k-row subsample for speed (`SLOW_MODEL_MAX_ROWS`).
* If your `Demand` / column names differ, adjust `src/config.py` and `src/data_prep.py`.

## Deliverables checklist
- [x] Source code (modular, commented) – `src/`, `train_all.py`, `app.py`
- [x] Processed dataset – `data/processed/charging_clean.csv`
- [x] Model results – `reports/*.csv`, `reports/figures/`
- [x] Streamlit dashboard – `app.py`
- [ ] Presentation & documentation – write up from `reports/` after running on the real dataset
