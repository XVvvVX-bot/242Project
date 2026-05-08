# Retail Demand Forecasting with Temporal Deep Learning

This project trains our own demand forecasting models on the public M5 Walmart retail dataset. The goal is to forecast daily department-level demand for each store using historical sales and calendar signals.

## Data source

- Dataset: M5 Forecasting - Accuracy
- Public mirror used by this project: https://zenodo.org/records/12636070
- Original competition page: https://www.kaggle.com/c/m5-forecasting-accuracy

The downloaded files are stored under `data/m5/`. Raw data is intentionally not committed to git because the extracted M5 CSVs exceed normal GitHub size limits. To download the data after cloning the repo, run:

```powershell
.\scripts\download_m5_data.ps1
```

The pipeline aggregates item-level daily sales into 70 store-department time series: 10 stores x 7 departments.

## Learning problem

For each store-department series, predict next-day unit sales from:

- the previous 56 days of demand,
- calendar features such as day of week, month, event indicator, and SNAP indicator,
- a learned series identity embedding.

The holdout set is the final 56 days. This is a supervised time-series forecasting task.

## Models trained

- Seasonal naive baseline: predicts demand from the same weekday one week ago.
- Moving-average baseline: predicts the mean of the previous 28 days.
- Lag MLP: neural baseline trained from lagged demand and current calendar features.
- Temporal Transformer: Transformer encoder trained from scratch on lagged demand, calendar features, and series embeddings.

## Run

```powershell
python src\train_forecaster.py
```

The checked-in result used:

```powershell
python src\train_forecaster.py --epochs 8 --max-train-windows-per-series 900 --batch-size 512
```

Outputs are written to:

- `outputs/metrics.csv`
- `outputs/train_history.csv`
- `outputs/predictions.csv`
- `outputs/figures/`
- `outputs/models/`
- `outputs/dashboard.html`
- `outputs/interactive_dashboard.html`

`outputs/interactive_dashboard.html` is a self-contained bonus dashboard. Open it in a browser to inspect forecasts by store-department series, toggle models, and compare validation errors.

## Report angle

This is a practical demand-planning system: a retailer could use forecasts to plan replenishment and staffing. The methodology section can compare simple forecasting heuristics against neural models and discuss whether attention-based temporal modeling improves validation error.
