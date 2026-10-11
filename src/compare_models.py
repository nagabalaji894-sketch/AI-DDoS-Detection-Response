
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
files = sorted(DATA_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV found in {DATA_DIR}")

print("Loading dataset...")
df = pd.read_csv(files[0], low_memory=False)
df.columns = df.columns.astype(str).str.strip()

label_col = next(
    (c for c in df.columns if c.lower() == "label"), None
)
if label_col is None:
    raise ValueError("Label column not found.")

df[label_col] = df[label_col].astype("string").str.strip()
df = df.dropna(subset=[label_col])
df = df[df[label_col] != ""]

# Remove the duplicate column already verified as identical.
if "Fwd Header Length.1" in df.columns:
    df = df.drop(columns=["Fwd Header Length.1"])

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Exact duplicate rows removed: {before - len(df):,}")

X = df.drop(columns=[label_col]).copy()
y = df[label_col].astype(str)

# Binary target: BENIGN = 0, DDoS = 1
unknown = set(y.unique()) - {"BENIGN", "DDoS"}
if unknown:
    raise ValueError(f"Unexpected labels: {unknown}")

y = y.map({"BENIGN": 0, "DDoS": 1})

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Imputation is fitted on training data only.
# SVM receives standardized features; tree models do not need scaling.
models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        )),
    ]),
    "Linear SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearSVC(
            random_state=42,
            class_weight="balanced",
            max_iter=5000,
        )),
    ]),
    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

results = []

for name, pipeline in models.items():
    print(f"\nTraining {name}...")
    start = time.perf_counter()

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        elapsed = time.perf_counter() - start

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "DDoS Precision": precision,
            "DDoS Recall": recall,
            "DDoS F1": f1,
            "Training + evaluation seconds": elapsed,
        })

        print(f"Accuracy:      {accuracy:.5f}")
        print(f"DDoS precision:{precision:.5f}")
        print(f"DDoS recall:   {recall:.5f}")
        print(f"DDoS F1:       {f1:.5f}")
        print("Confusion matrix (actual rows, predicted columns):")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))
        print("Class order: BENIGN (0), DDoS (1)")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "DDoS"],
            digits=5,
            zero_division=0,
        ))

        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_columns": X.columns.tolist(),
                "labels": ["BENIGN", "DDoS"],
            },
            MODEL_DIR / (
                name.lower().replace(" ", "_") + "_ddos.joblib"
            ),
        )

    except Exception as exc:
        print(f"{name} failed: {type(exc).__name__}: {exc}")

if not results:
    raise RuntimeError("All model training attempts failed.")

results_df = pd.DataFrame(results).sort_values(
    "DDoS F1", ascending=False
)

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

results_df.to_csv(ROOT / "model_comparison.csv", index=False)
print(f"\nResults saved to: {ROOT / 'model_comparison.csv'}")
print(f"Models saved in: {MODEL_DIR}")
