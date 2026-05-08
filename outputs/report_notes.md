# Report Notes

## Project description and motivation

Retailers need daily demand forecasts to decide how much inventory to replenish, where to allocate labor, and when to investigate likely stockout risk. We model demand for Walmart store-department combinations using the public M5 sales dataset.

## Data and learning problem

The original M5 data contains daily item-level unit sales across Walmart stores from 2011-01-29 through 2016-04-24. To keep training computationally feasible and operationally interpretable, this project aggregates item sales into 70 store-department series: 10 stores times 7 departments. The supervised learning task is:

Given the previous 56 days of sales for a store-department series, plus calendar variables, predict the next day's unit demand.

The final 56 days are held out for validation.

## Methodology

Baselines:

- Seasonal naive: next demand equals demand from the same weekday one week earlier.
- Moving average: next demand equals the previous 28-day average.
- Lag MLP: feed-forward neural network over lagged sales, current calendar features, and a learned series embedding.

Main model:

- Temporal Transformer encoder trained from scratch.
- Inputs are normalized lagged demand and calendar features.
- The model uses positional embeddings and a learned series embedding so one global network can share statistical strength across departments and stores.
- Objective: minimize mean absolute error on normalized demand using AdamW with early model selection by validation MAE.

## Current validation result

Best model by MAE: lag_mlp

| Model | MAE | RMSE | sMAPE (%) | WAPE (%) |
|---|---:|---:|---:|---:|
| lag_mlp | 54.800 | 95.480 | 12.92 | 9.10 |
| temporal_transformer | 59.780 | 108.408 | 13.43 | 9.93 |
| seasonal_naive_7 | 86.069 | 158.331 | 18.53 | 14.30 |
| moving_average_28 | 104.758 | 180.311 | 18.94 | 17.40 |

## Result discussion

Use `outputs/figures/training_curves.png` to discuss underfitting/overfitting. Use `outputs/figures/model_mae_comparison.png` to compare the neural models against simple operational baselines. Forecast plots for high-volume series are saved under `outputs/figures/forecast_*.png`.

## Safety, security, and ethics

The model should not directly automate replenishment without guardrails. Forecast errors can create stockouts for essential goods or overstock waste for perishable goods. Historical demand reflects past prices, promotions, local demographics, and possible stockout censoring; these biases can be amplified by a deployed system. A practical deployment should include human override, service-level constraints, uncertainty estimates, and monitoring for drift after holidays, economic shocks, or assortment changes.
