
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
TRAIN_DIR = ROOT / "data"
EXTERNAL_PATH = (
    ROOT / "data" / "external" / "cic_ddos2019_test.csv"
)

# Features shared by both datasets.
FEATURES = [
    "Flow Duration",
    "Fwd Packet Length Std",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd Header Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "ACK Flag Count",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
]

def normalize_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    return df

def find_label_column(df):
    for col in df.columns:
        if col.lower() == "label":
            return col
    raise ValueError("No Label column found.")

def binary_labels(series):
    labels = series.astype("string").str.strip().str.upper()
    labels = labels.replace({"": pd.NA, "NAN": pd.NA})
    valid = labels.notna()
    labels = labels[valid]

    # BENIGN = 0; every recognized non-benign class = 1.
    y = (labels != "BENIGN").astype(int)
    return valid, y

# 1. Locate the original CIC-IDS2017 training file.
source_files = [
    p for p in TRAIN_DIR.glob("*.csv")
    if p.name != "model_comparison.csv"
]

if not source_files:
    raise FileNotFoundError(
        "No source CSV found in data/. Keep the original "
        "CIC-IDS2017 training file in that folder."
    )

source_path = source_files[0]

if not EXTERNAL_PATH.exists():
    raise FileNotFoundError(
        f"Download the external test file to: {EXTERNAL_PATH}"
    )

print("Source training file:", source_path.name)
print("External test file:", EXTERNAL_PATH.name)

# 2. Load and clean the source training dataset.
source = normalize_columns(
    pd.read_csv(source_path, low_memory=False)
)

source_label_col = find_label_column(source)
source[source_label_col] = (
    source[source_label_col].astype("string").str.strip()
)

source = source.dropna(subset=[source_label_col])
source = source[
    source[source_label_col].isin(["BENIGN", "DDoS"])
].copy()

# Remove the redundant duplicate column from the original CSV.
source = source.drop(
    columns=["Fwd Header Length.1"],
    errors="ignore",
)

source = source.drop_duplicates().reset_index(drop=True)

missing_source = [
    c for c in FEATURES if c not in source.columns
]
if missing_source:
    raise ValueError(
        f"Source dataset missing features: {missing_source}"
    )

X_train = source[FEATURES].copy()
y_train = (source[source_label_col] == "DDoS").astype(int)

for col in FEATURES:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)

# 3. Load the external test dataset.
external = normalize_columns(
    pd.read_csv(EXTERNAL_PATH, low_memory=False)
)

external_label_col = find_label_column(external)
missing_external = [
    c for c in FEATURES if c not in external.columns
]
if missing_external:
    raise ValueError(
        f"External dataset missing features: {missing_external}"
    )

valid, y_test = binary_labels(external[external_label_col])
external = external.loc[valid].reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_test = external[FEATURES].copy()

for col in FEATURES:
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X_test = X_test.replace([np.inf, -np.inf], np.nan)

print("\nSource training rows:", len(X_train))
print("External test rows:", len(X_test))
print("\nSource class counts:")
print(y_train.value_counts().rename(index={0: "BENIGN", 1: "DDoS"}))
print("\nExternal class counts:")
print(y_test.value_counts().rename(index={0: "BENIGN", 1: "DDoS"}))

if y_test.nunique() != 2:
    raise ValueError(
        "External test file must contain both BENIGN and attack records."
    )

# 4. Fit preprocessing and XGBoost on source data only.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )),
])

print("\nTraining on CIC-IDS2017 only...")
model.fit(X_train, y_train)

# 5. Evaluate once on the separate CIC-DDoS2019 dataset.
print("\nEvaluating on CIC-DDoS2019...")
pred = model.predict(X_test)

print(f"\nExternal accuracy:  {accuracy_score(y_test, pred):.5f}")
print(
    f"External precision: {precision_score(y_test, pred, zero_division=0):.5f}"
)
print(
    f"External recall:    {recall_score(y_test, pred, zero_division=0):.5f}"
)
print(
    f"External F1-score:  {f1_score(y_test, pred, zero_division=0):.5f}"
)

print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print("Class order: BENIGN (0), attack (1)")

print("\nClassification report:")
print(classification_report(
    y_test,
    pred,
    labels=[0, 1],
    target_names=["BENIGN", "DDoS/attack"],
    digits=5,
    zero_division=0,
))

print("\nExternal evaluation complete.")
print("The external test records were not used to fit this model.")
