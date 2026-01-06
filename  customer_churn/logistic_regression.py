import kagglehub
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt



# Download dataset
DATA_DIR = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
print("Dataset path:", DATA_DIR)

# Load CSV (file is already inside DATA_DIR)
csv_path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(csv_path)

print(df.shape)
print(df["Churn"].value_counts())

print(df.head())

df.drop('customerID', axis=1, inplace=True)

print(df.columns)
print(df.isnull().sum())
print(df.isna().sum())
print(f'{(df["TotalCharges"].astype(str).str.strip() == "").sum()}')


# Fix TotalCharges (stored as string with blanks)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())



# Encode target
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)



model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0
    ))
])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

probs = model.predict_proba(X_test)[:, 1]
df_test = X_test.copy()
df_test["churn_probability"] = probs
df_test["churn_prediction"] = (probs >= 0.5).astype(int)




# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix (Default threshold = 0.5)")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve – Customer Churn")
plt.show()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# after CM plot
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")

# after ROC plot
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=200, bbox_inches="tight")
