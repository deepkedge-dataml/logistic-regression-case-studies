# Customer Churn Prediction using Logistic Regression

This project implements an end-to-end **binary classification pipeline** to predict customer churn using **Logistic Regression** on the IBM Telco Customer Churn dataset.

The goal is to identify customers who are likely to churn so that preventive actions can be taken.

---

## Dataset
- **Source:** IBM Telco Customer Churn (via Kaggle)
- **Rows:** 7,043
- **Target variable:** `Churn` (Yes / No)

Key challenge handled:
- `TotalCharges` stored as strings with blank values

---

## Data Preprocessing
- Removed `customerID` (non-informative)
- Converted `TotalCharges` to numeric
- Handled missing values using **median imputation**
- Encoded target:
  - `No → 0`
  - `Yes → 1`
- Used **ColumnTransformer**:
  - Categorical features → OneHotEncoder
  - Numerical features → StandardScaler

---

##  Model
- **Algorithm:** Logistic Regression
- **Class imbalance handled using:** `class_weight="balanced"`
- **Pipeline used:** preprocessing + model
- **Train/Test split:** 80 / 20 (stratified)

---

## Evaluation Metrics
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC Curve & ROC-AUC score
- Predicted churn probabilities added to test dataframe

---

## Thresholding
- Default threshold: `0.5`
- Custom thresholds can be applied using predicted probabilities
- Enables business-driven decision making

---

## Project Structure

customer_churn/
│
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── results/
│ ├── confusion_matrix.png
│ └── roc_curve.png
│
├── churn_logistic_regression.py
├── README.md
└── .gitignore