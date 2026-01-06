# Logistic Regression – Breast Cancer Recurrence

## Dataset
Breast Cancer dataset from the UCI Machine Learning Repository.
The goal is to predict whether breast cancer will recur
(`recurrence-events`) or not (`no-recurrence-events`).

## Problem Type
Binary classification (supervised learning).

## Model
- Logistic Regression
- One-Hot Encoding for categorical features
- Class balancing to handle class imbalance

## Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-score
- ROC Curve and ROC-AUC
- 5-Fold Cross-Validation ROC-AUC

## Results
- ROC-AUC ≈ 0.64–0.65
- Cross-validation confirms consistent performance
- Custom threshold analysis shows precision–recall trade-off

## Notes
This dataset is small (286 samples), so cross-validation
is used to obtain a more reliable performance estimate.
