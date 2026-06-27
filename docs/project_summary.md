# Project Summary — Wine Quality Supervised ML

This project applies supervised Machine Learning to the UCI Wine Quality dataset.

## Objectives

1. Predict whether a wine is red or white from physicochemical variables.
2. Estimate wine quality using regression models.
3. Compare baseline models against optimized ensemble models.

## Main Results

### Classification

The best classifier was an optimized XGBoost model with:

- Accuracy: **0.9975**
- Weighted F1-score: **0.9975**

### Regression

The best regressor was an optimized XGBoost model with:

- MAE: **0.4515**
- R²: **0.4865**

## Interpretation

Wine type classification is highly reliable because red and white wines have clearly different physicochemical profiles.

Wine quality prediction is harder because quality is subjective, ordinal and influenced by variables that are not fully represented in the dataset.
