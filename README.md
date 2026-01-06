Logistic Regression – Case Studies

This repository contains end-to-end logistic regression case studies built on real-world datasets.
Each case study follows a complete machine-learning workflow, from data loading and preprocessing to model evaluation and interpretation.

The goal of this repository is to demonstrate practical understanding of logistic regression, not just theory.

Case Study 1: Breast Cancer Recurrence

Problem
Predict whether breast cancer will recur based on clinical features.

Key Steps

Data loading & inspection

Categorical feature encoding (One-Hot Encoding)

Logistic Regression with class balancing

Model evaluation using:

Confusion Matrix

Classification Report

ROC Curve & AUC

Outcome

Demonstrates handling of categorical medical data

Focus on recall vs precision trade-offs in healthcare


Case Study 2: Customer Churn Prediction

Dataset
Telco Customer Churn dataset (Kaggle)

Problem
Predict whether a customer is likely to churn (leave the service).

Key Steps

Dataset download using kagglehub

Cleaning numeric columns (TotalCharges)

Handling missing values

Feature preprocessing using ColumnTransformer

Logistic Regression with regularization

Probability-based predictions

Threshold-based decision control

Model evaluation using:

Confusion Matrix

Precision / Recall / F1

ROC Curve & AUC

Extra

Churn probabilities are attached back to the test dataframe

Threshold can be adjusted (e.g. 0.7) based on business needs

