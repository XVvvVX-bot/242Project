# Project Outline

## Working title

Retail Demand Forecasting with Temporal Deep Learning on Walmart M5 Sales Data

## Core question

Can a neural temporal model trained on historical sales and calendar features improve next-day department demand forecasts compared with simple retail forecasting rules?

## Proposed report structure

1. Abstract
2. Motivation and use case
3. Dataset and supervised learning formulation
4. Methodology
5. Experiments and results
6. Safety, security, and ethics
7. Conclusion

## Methods to discuss

- Seasonal naive and moving-average baselines
- Lag MLP neural baseline
- Temporal Transformer trained from scratch
- MAE loss, AdamW optimizer, dropout, validation model selection

## Required figures

- Training and validation loss curves
- Model comparison bar chart
- Example actual-vs-forecast time-series plots
- Error breakdown by department

## Potential deployment demo

A lightweight app can load `outputs/predictions.csv` and let a user select a store-department pair to compare the actual and predicted validation demand. A second version could forecast the next day from a user-provided recent demand history.
