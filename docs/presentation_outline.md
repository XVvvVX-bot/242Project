# 10-Minute Presentation Outline

## Slide 1 - Title

Retail Demand Forecasting with Temporal Deep Learning on Walmart M5 Sales Data

## Slide 2 - Motivation

- Retailers need reliable short-term forecasts for replenishment and staffing.
- Bad forecasts cause stockouts, waste, and poor service levels.
- Goal: forecast next-day demand for each store-department pair.

## Slide 3 - Data

- M5 Walmart sales dataset.
- Aggregated item-level sales into 70 store-department series.
- Inputs: previous 56 days of sales, calendar features, SNAP indicator, series identity.
- Holdout: final 56 days.

## Slide 4 - Learning Formulation

- Supervised time-series forecasting.
- Map historical sales window and calendar variables to next-day demand.
- Normalize each series using training-period statistics.

## Slide 5 - Models

- Seasonal naive baseline.
- 28-day moving average baseline.
- Lag MLP with learned series embedding.
- Temporal Transformer encoder trained from scratch.

## Slide 6 - Training Setup

- Loss: mean absolute error on normalized demand.
- Optimizer: AdamW with weight decay.
- Regularization: dropout and validation checkpoint selection.
- Metrics: MAE, RMSE, sMAPE, WAPE.

## Slide 7 - Results

Use `outputs/figures/model_mae_comparison.png`.

Main result:

- Lag MLP: MAE 54.8.
- Temporal Transformer: MAE 59.8.
- Seasonal naive: MAE 86.1.
- Moving average: MAE 104.8.

## Slide 8 - Diagnostics

Use `outputs/figures/training_curves.png` and one forecast plot.

- Neural models beat simple baselines.
- Transformer improves with training but does not beat MLP.
- Spikes remain difficult.

## Slide 9 - Safety and Ethics

- Forecast errors can create stockouts or waste.
- Historical sales may understate true demand when stockouts occurred.
- Deployment should include guardrails, monitoring, human override, and uncertainty estimates.

## Slide 10 - Conclusion

- Built a full data-to-model retail forecasting pipeline.
- Best model is a lag MLP; Transformer is a useful advanced comparison.
- Next steps: add price features, item-level forecasting, probabilistic forecasts, inventory-cost evaluation.
