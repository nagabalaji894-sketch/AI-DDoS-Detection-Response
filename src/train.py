
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv(csv_files[0], low_memory=False)
df.columns = df.columns.astype(str).str.strip()

# 2. Normalize the target labels
label_col = next(
    (c for c in df.columns if c.lower() == "label"),
    None,
)
if label_col is None:
    raise ValueError("The dataset does not contain a Label column.")

df[label_col] = df[label_col].astype("string").str.strip()
df = df.dropna(subset=[label_col])
df = df[df[label_col] != ""]

print(f"Original usable rows: {len(df):,}")

# 3. Remove the verified duplicate feature column
duplicate_col = "Fwd Header Length.1"
if duplicate_col in df.columns:
    df = df.drop(columns=[duplicate_col])
    print(f"Removed duplicate feature: {duplicate_col}")

# 4. Remove exact duplicate records
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Exact duplicate records removed: {before - len(df):,}")

# 5. Separate features and target
X = df.drop(columns=[label_col]).copy()
y = df[label_col].astype(str)

# Convert features to numeric and replace infinite values.
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

# Remove features that contain no usable numeric values.
empty_cols = X.columns[X.isna().all()].tolist()
if empty_cols:
    print(f"Removing entirely missing features: {empty_cols}")
    X = X.drop(columns=empty_cols)

if X.shape[1] == 0:
    raise ValueError("No usable numeric features remain.")

print(f"Rows for modeling: {len(X):,}")
print(f"Numeric features: {X.shape[1]}")
print("\nClass distribution:")
print(y.value_counts())

# 6. Split before fitting preprocessing or the model.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")

# 7. Build the first baseline model.
# Imputation is fitted only on the training data.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# 8. Evaluate on the held-out test set.
predictions = model.predict(X_test)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["BENIGN", "DDoS"],
))
print("Class order: BENIGN, DDoS")

# 9. Save the fitted pipeline and expected feature order.
model_path = MODEL_DIR / "random_forest_ddos.joblib"
joblib.dump({
    "pipeline": model,
    "feature_columns": X.columns.tolist(),
    "labels": ["BENIGN", "DDoS"],
}, model_path)

print(f"\nSaved model to: {model_path}")
print("Training and evaluation complete.")
