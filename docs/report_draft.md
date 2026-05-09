# Retail Demand Forecasting with Temporal Deep Learning on Walmart M5 Sales Data

## Abstract

This project studies daily retail demand forecasting for Walmart store-department combinations using the public M5 Forecasting dataset. We aggregate item-level sales into 70 operational time series, one for each store and department pair, and train models to predict next-day unit demand from rolling sales histories and calendar covariates. We compare two simple operational baselines, a lag-based neural network, a temporal Transformer trained from scratch, and reversible instance normalization (RevIN) variants of both neural models. After bounded hyperparameter tuning, the Lag MLP + RevIN model achieves the lowest mean absolute error on the final 56-day validation period. RevIN improves both neural architectures, suggesting that local window normalization is useful for nonstationary retail demand.

## 1. Motivation and Use Case

Retailers rely on demand forecasts to decide how much inventory to replenish, how to allocate warehouse capacity, and when to investigate likely stockout risk. Poor forecasts can cause stockouts, excess inventory, wasted labor, and unnecessary logistics costs. This project frames forecasting as a practical decision-support problem: given recent sales history for a store-department pair, estimate tomorrow's demand so an operations team can plan replenishment and staffing.

The project is intentionally scoped to a level that is useful for a retailer but feasible for a course project. Instead of forecasting every individual SKU, we aggregate demand by store and department. This reduces noise, allows the model to learn from multiple related series, and creates outputs that can support department-level planning.

## 2. Data and Learning Problem

The data source is the M5 Forecasting - Accuracy dataset, originally released through Kaggle and mirrored publicly on Zenodo. It contains daily unit sales for Walmart products across stores, along with calendar information such as weekday, month, events, and SNAP indicators. We use `sales_train_validation.csv` and `calendar.csv`.

The raw item-level dataset contains thousands of item-store time series. We aggregate item-level daily unit sales into 70 store-department series: 10 stores times 7 departments. For each series, the target variable is daily unit sales. The model observes a rolling historical window and predicts demand for the next day. Hyperparameter tuning considers 56-, 84-, and 112-day windows; the final MLP family uses 84 days, while the final Transformer family uses 56 days.

Formally, for series `s` and day `t`, the learning task is to estimate:

```text
y_{s,t} = f_theta(y_{s,t-L:t-1}, c_{s,t-L:t-1}, c_{s,t}, s)
```

where `y` is unit demand, `c` contains calendar features, `s` is the store-department identity, and `L` is the tuned lookback length. The final 56 days are held out for validation. The original neural models normalize each series using its training-period mean and standard deviation. The RevIN variants additionally normalize each rolling input window using only that window's mean and standard deviation.

## 3. Methodology

We train and evaluate six models.

The first baseline is seasonal naive: demand for a day is predicted as demand from the same weekday one week earlier. This is a strong retail baseline because weekly seasonality is common in store traffic and shopping behavior.

The second baseline is a 28-day moving average. This captures recent demand level while smoothing day-to-day noise.

The first neural model is a lag MLP. It receives normalized lagged demand values, current calendar features, and a learned series embedding, then predicts normalized next-day demand. It is trained with mean absolute error using AdamW.

The advanced methodology is a temporal Transformer encoder trained from scratch. The Transformer receives the sequence of lagged demand and historical calendar features. It uses positional embeddings to encode time order and a learned series embedding so the model can share information across stores and departments while preserving series-specific behavior. A final prediction head maps the last encoded time step to next-day demand. This architecture is more flexible than a feed-forward lag model because self-attention can learn nonlocal dependencies across the lookback window.

We add reversible instance normalization (RevIN) as an additional time-series normalization experiment. For each training example, RevIN computes the mean and standard deviation of the input window, normalizes the input window using those statistics, trains the model to predict the next day in this local normalized space, and then reverses the transformation:

```text
prediction = prediction_norm * window_std + window_mean
```

Only the input window statistics are used, so the target day is never leaked into the features. We also provide the local mean and standard deviation as auxiliary features so the models can still use information about the local demand level.

The optimization problem is:

```text
min_theta (1/N) sum_i |f_theta(X_i) - Y_i| + lambda ||theta||_2^2
```

where the loss is mean absolute error on normalized demand. AdamW supplies the weight decay regularization term, and dropout is used in the neural models. The best checkpoint is selected by validation MAE.

After adding RevIN, we tune both neural model families with a bounded sweep. The search varies lookback length, learning rate, model width/depth, dropout, and embedding size. The final MLP configuration uses an 84-day lookback, hidden layers 384 and 192, dropout 0.15, embedding size 24, and learning rate 0.0007. The final Transformer configuration uses a 56-day lookback, model dimension 96, 4 attention heads, 2 encoder layers, dropout 0.15, and learning rate 0.0007. Final training uses 10 epochs with best-checkpoint selection.

## 4. Results

The final validation metrics are:

| Model | MAE | RMSE | sMAPE (%) | WAPE (%) |
|---|---:|---:|---:|---:|
| Lag MLP + RevIN | 52.914 | 90.672 | 12.66 | 8.79 |
| Lag MLP | 53.982 | 93.920 | 12.77 | 8.97 |
| Temporal Transformer + RevIN | 56.616 | 99.007 | 12.87 | 9.41 |
| Temporal Transformer | 59.180 | 107.437 | 13.18 | 9.83 |
| Seasonal naive | 86.069 | 158.331 | 18.53 | 14.30 |
| Moving average 28 | 104.758 | 180.311 | 18.94 | 17.40 |

All neural models outperform the operational baselines by a wide margin. The best model is Lag MLP + RevIN, with MAE 52.91 and WAPE 8.79%. This improves over the tuned Lag MLP and reduces MAE by about 39 percent relative to seasonal naive. RevIN also improves the Transformer, reducing MAE from 59.18 to 56.62.

The training curves show that the lag MLP converges quickly and that RevIN provides a consistent validation improvement. The Transformer continues improving across epochs, and the RevIN Transformer closes part of the gap to the MLP. In this aggregated department-level setting, short-term lag structure remains very strong, but local window normalization helps the models handle shifts in recent demand level.

The forecast plots show that the neural models capture the broad demand level for high-volume food departments, though sharp spikes remain difficult. These spikes may correspond to promotions, stockout recovery, local events, or other demand shocks not fully represented by the calendar variables.

## 5. Safety, Security, and Ethics

A forecasting model should not directly automate replenishment without guardrails. Forecast errors can create stockouts for essential goods or excess inventory for perishable goods. Both outcomes have social and operational costs: stockouts can disproportionately affect customers with fewer shopping alternatives, while overstock creates waste.

The historical data may also encode censored demand. If a product was out of stock, observed sales understate true demand, so the model may learn to forecast artificially low demand for constrained products. Calendar and regional demand patterns can also reflect local socioeconomic differences. A deployed model should therefore include human override, service-level constraints, uncertainty estimates, and post-deployment drift monitoring.

Security concerns are also relevant. If forecasts are exposed through a planning dashboard, competitors or malicious actors could infer sales patterns, promotion timing, or inventory constraints. Access control and aggregation thresholds should be used before deployment.

## 6. Conclusion

This project demonstrates a complete data-to-model pipeline for practical retail demand forecasting. The best current model is Lag MLP + RevIN, which beats seasonal naive, moving-average, standard MLP, and standard Transformer models on the final 56-day holdout. The temporal Transformer is a meaningful advanced method and improves substantially over non-neural baselines; adding RevIN further improves it. A strong next step would be to include price features, train at item-store granularity, and evaluate whether probabilistic forecasts improve inventory decisions under stockout and holding-cost tradeoffs.

## Appendix: Reproducibility

Run:

```powershell
python src\train_forecaster.py --epochs 10 --max-train-windows-per-series 900 --batch-size 512
```

Key outputs:

- `outputs/metrics.csv`
- `outputs/train_history.csv`
- `outputs/predictions.csv`
- `outputs/figures/model_mae_comparison.png`
- `outputs/figures/training_curves.png`
- `outputs/figures/forecast_CA_1_FOODS_3.png`
