import pandas as pd
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt

# -----------------------------
# paths
# -----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

zip_path = DATA_DIR / "breast_cancer.zip"

# -----------------------------
# extract the zip only if the .data file is not already there
# -----------------------------
data_file = DATA_DIR / "breast-cancer.data"
if not data_file.exists():
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

# -----------------------------
# read dataset
# -----------------------------
df = pd.read_csv(data_file, header=None)

# column names based on the .names file
df.columns = [
    "Class",
    "age",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "breast",
    "breast_quad",
    "irradiat"
]

# -----------------------------
# split features and target
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# encode the target (so we can do thresholding easily)
le = LabelEncoder()
y_bin = le.fit_transform(y)

# make sure we take probability of recurrence-events as "positive"
positive_class = "recurrence-events"
pos_label = le.transform([positive_class])[0]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# -----------------------------
# preprocessing: one hot encode categorical columns
# -----------------------------
cat_cols = X.select_dtypes(include="object").columns

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# logistic regression model
model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1))
])

# train
model.fit(X_train, y_train)

# -----------------------------
# predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, pos_label]

print("Report (default threshold = 0.5)")
print(classification_report(y_test, y_pred))

# confusion matrix for default threshold
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix (threshold=0.5)")
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------
# custom threshold (example)
# -----------------------------
threshold = 0.7
y_pred_custom = (y_prob >= threshold).astype(int)

print(f"\nReport (custom threshold = {threshold})")
print(classification_report(y_test, y_pred_custom))

cm2 = confusion_matrix(y_test, y_pred_custom)
ConfusionMatrixDisplay(cm2).plot()
plt.title(f"Confusion Matrix (threshold={threshold})")
plt.savefig(RESULTS_DIR / f"confusion_matrix_threshold_{threshold}.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------
# ROC curve + AUC
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------
# cross validation (more stable since dataset is small)
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X, y_bin, cv=cv, scoring="roc_auc")

print(f"\n5-fold CV ROC-AUC: mean={cv_auc.mean():.3f}, std={cv_auc.std():.3f}")
