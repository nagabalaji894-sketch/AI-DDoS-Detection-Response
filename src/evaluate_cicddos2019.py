
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external"
MODELS_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "cic_ddos2019_train.csv"
TEST_PATH = DATA_DIR / "cic_ddos2019_test.csv"

MODEL_PATH = MODELS_DIR / "xgboost_cicddos2019.joblib"
ERROR_PATH = ROOT / "cicddos2019_prediction_errors.csv"
RESULTS_PATH = ROOT / "cicddos2019_evaluation_results.csv"

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

LABEL_MAP = {
    0: "BENIGN",
    1: "ATTACK",
}


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_dataset(file_path):
    """Load a CSV, validate its columns, and prepare binary labels."""

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    # Find the label column regardless of capitalization.
    label_column = next(
        (column for column in df.columns
         if column.lower() == "label"),
        None,
    )

    if label_column is None:
        raise ValueError(
            f"No Label column found in {file_path.name}."
        )

    # Validate all required features.
    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{file_path.name} is missing required features: "
            f"{missing_features}"
        )

    # Normalize labels and remove missing labels.
    df[label_column] = (
        df[label_column]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df = df.dropna(subset=[label_column]).copy()

    # Keep original attack names for error analysis.
    original_labels = df[label_column].reset_index(drop=True)

    # BENIGN = 0; all non-BENIGN classes = ATTACK (1).
    y = (
        original_labels != "BENIGN"
    ).astype("int8")

    # Extract the same 14 features in a fixed order.
    X = df[FEATURES].copy()

    # Convert invalid strings to NaN.
    for feature in FEATURES:
        X[feature] = pd.to_numeric(
            X[feature], errors="coerce"
        )

    # Replace infinite values; the pipeline imputes missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Reset indices to keep X, y, and labels aligned.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, original_labels


# ============================================================
# 3. MAIN TRAINING AND EVALUATION
# ============================================================

def main():

    # Check that both files exist before starting.
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_PATH}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets independently.
    X_train, y_train, train_labels = load_dataset(TRAIN_PATH)
    X_test, y_test, test_labels = load_dataset(TEST_PATH)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training or test dataset is empty.")

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain benign and attack records."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "Test data must contain benign and attack records."
        )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CIC-DDOS2019 BINARY ATTACK DETECTION")
    print("=" * 65)

    print(f"Training file: {TRAIN_PATH.name}")
    print(f"Test file:     {TEST_PATH.name}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print(f"Features:      {len(FEATURES)}")

    print("\nTraining class counts:")
    print(
        y_train.map(LABEL_MAP)
        .value_counts()
        .to_string()
    )

    print("\nTest class counts:")
    print(
        y_test.map(LABEL_MAP)
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Build model pipeline
    # --------------------------------------------------------

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    # --------------------------------------------------------
    # Train using training data only
    # --------------------------------------------------------

    print("\nTraining XGBoost on training split...")
    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Evaluate on held-out test data
    # --------------------------------------------------------

    print("Predicting on held-out test split...")

    predictions = model.predict(X_test).astype(int)

    probabilities = model.predict_proba(X_test)[:, 1]

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

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    # --------------------------------------------------------
    # Print evaluation metrics
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    print(f"Accuracy:             {accuracy:.5f}")
    print(f"Attack precision:     {precision:.5f}")
    print(f"Attack recall:        {recall:.5f}")
    print(f"Attack F1-score:      {f1:.5f}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nConfusion matrix interpretation:")
    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives:  {tp:,}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            digits=5,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save misclassified records
    # --------------------------------------------------------

    error_report = X_test.copy()

    error_report["Actual_Original_Label"] = test_labels
    error_report["Actual"] = y_test.map(LABEL_MAP)
    error_report["Predicted"] = pd.Series(
        predictions
    ).map(LABEL_MAP)

    error_report["Attack_Probability"] = probabilities

    error_report["Error_Type"] = np.select(
        [
            (y_test.to_numpy() == 1)
            & (predictions == 0),

            (y_test.to_numpy() == 0)
            & (predictions == 1),
        ],
        [
            "False Negative",
            "False Positive",
        ],
        default="Correct",
    )

    errors = error_report[
        error_report["Error_Type"] != "Correct"
    ].copy()

    errors.to_csv(ERROR_PATH, index=False)

    print("\n" + "=" * 65)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 65)

    print(f"Total incorrect predictions: {len(errors):,}")

    if errors.empty:
        print("No incorrect predictions on this test split.")
    else:
        print("\nErrors by type:")
        print(errors["Error_Type"].value_counts().to_string())

        print("\nFirst 10 incorrect predictions:")
        print(
            errors[
                [
                    "Actual_Original_Label",
                    "Actual",
                    "Predicted",
                    "Attack_Probability",
                    "Error_Type",
                ]
            ].head(10).to_string(index=False)
        )

    print(f"\nError report saved to:\n{ERROR_PATH}")

    # --------------------------------------------------------
    # Save a compact metrics report
    # --------------------------------------------------------

    metrics = pd.DataFrame([{
        "Dataset": "CIC-DDoS2019",
        "Training_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Accuracy": accuracy,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": f1,
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
    }])

    metrics.to_csv(RESULTS_PATH, index=False)

    print(f"Metrics report saved to:\n{RESULTS_PATH}")

    # --------------------------------------------------------
    # Save separate model without replacing dashboard model
    # --------------------------------------------------------

    joblib.dump(
        {
            "pipeline": model,
            "feature_columns": FEATURES,
            "labels": LABEL_MAP,
        },
        MODEL_PATH,
    )

    print(f"\nSeparate model saved to:\n{MODEL_PATH}")
    print("Existing dashboard model was not changed.")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR: Evaluation could not complete.")
        print(f"{type(error).__name__}: {error}")
        raise
