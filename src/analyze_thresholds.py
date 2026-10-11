
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data" / "external" / "cic_ddos2019_test.csv"
MODEL_PATH = ROOT / "models" / "xgboost_cicddos2019.joblib"
OUTPUT_PATH = ROOT / "cicddos2019_threshold_analysis.csv"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def main():
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    saved = joblib.load(MODEL_PATH)
    model = saved["pipeline"]
    features = saved["feature_columns"]

    df = pd.read_csv(TEST_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    label_column = next(
        (col for col in df.columns if col.lower() == "label"),
        None,
    )
    if label_column is None:
        raise ValueError("No Label column found in test CSV.")

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    labels = df[label_column].astype("string").str.strip().str.upper()
    valid = labels.notna()

    df = df.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)

    y_true = (labels != "BENIGN").astype(int).to_numpy()

    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Reuse the trained model; do not retrain it.
    probabilities = model.predict_proba(X)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, predictions),
            "Attack_Precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "Attack_Recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "Attack_F1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\nTHRESHOLD COMPARISON")
    print(result_df.round(5).to_string(index=False))
    print(f"\nSaved results to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
