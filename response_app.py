
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgboost_cicddos2019.joblib"

st.set_page_config(
    page_title="AI DDoS Response Simulator",
    page_icon="🛡️",
    layout="wide",
)

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

# Additional fields are included in reports if present.
CONTEXT_COLUMNS = [
    "Timestamp",
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Address",
    "Destination IP",
    "Dst IP",
    "Destination Port",
    "Protocol",
]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    saved = joblib.load(MODEL_PATH)

    if "pipeline" not in saved or "feature_columns" not in saved:
        raise ValueError(
            "The model file does not contain the expected pipeline "
            "and feature_columns."
        )

    return saved["pipeline"], saved["feature_columns"]


# ============================================================
# RESPONSE POLICY — SIMULATION ONLY
# ============================================================

def response_policy(probability, threshold):
    """
    Priority is based on model confidence, not confirmed attack impact.
    No actual network action is performed.
    """

    if probability < threshold:
        return (
            "BENIGN",
            "No attack alert",
            "No response proposed",
            "Not applicable",
        )

    if probability >= 0.95:
        return (
            "ATTACK",
            "CRITICAL",
            "Simulate temporary block",
            "SIMULATED - NOT BLOCKED",
        )

    if probability >= 0.80:
        return (
            "ATTACK",
            "HIGH",
            "Simulate rate-limit",
            "SIMULATED - NOT RATE-LIMITED",
        )

    if probability >= 0.65:
        return (
            "ATTACK",
            "MEDIUM",
            "Raise security alert",
            "ALERT RECORDED ONLY",
        )

    return (
        "ATTACK",
        "LOW",
        "Monitor suspicious flow",
        "MONITORING SIMULATED",
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze_data(uploaded_df, model, features, threshold):
    df = uploaded_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Fail early if required features are missing.
    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(
            "Uploaded CSV is missing required features:\n"
            + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("The uploaded CSV contains no records.")

    # Prepare the model inputs in the correct order.
    X = df[features].copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Pipeline handles missing values with its trained imputer.
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    run_time = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    # Preserve optional context fields for investigation.
    context = [
        col for col in CONTEXT_COLUMNS if col in df.columns
    ]

    results = df[context].copy().reset_index(drop=True)

    results.insert(0, "Record_ID", np.arange(1, len(df) + 1))
    results["Attack_Probability"] = probabilities
    results["Predicted_Label"] = np.where(
        predictions == 1, "ATTACK", "BENIGN"
    )
    results["Detection_Time"] = run_time

    policies = [
        response_policy(float(probability), threshold)
        for probability in probabilities
    ]

    results["Alert_Priority"] = [p[1] for p in policies]
    results["Proposed_Action"] = [p[2] for p in policies]
    results["Response_Status"] = [p[3] for p in policies]

    # Attack alerts only.
    alerts = results[
        results["Predicted_Label"] == "ATTACK"
    ].copy()

    # Optional metrics if the uploaded data has ground-truth labels.
    metrics = None

    label_column = next(
        (col for col in df.columns if col.lower() == "label"),
        None,
    )

    if label_column is not None:
        actual_labels = (
            df[label_column].astype("string").str.strip().str.upper()
        )
        valid = actual_labels.notna().to_numpy()

        if valid.any():
            y_true = (
                actual_labels[valid] != "BENIGN"
            ).astype(int).to_numpy()

            y_pred = predictions[valid]
            matrix = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            )

            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Attack precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "Attack recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "Attack F1": f1_score(
                    y_true, y_pred, zero_division=0
                ),
                "Confusion matrix": matrix,
                "Evaluated rows": int(valid.sum()),
            }

    return results, alerts, metrics


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🛡️ AI-Based DDoS Detection & Response Simulator")

st.write(
    "Upload a network-flow CSV to classify traffic, assign alert "
    "priorities, and simulate incident-response actions."
)

st.warning(
    "SIMULATION MODE ONLY: This application does not block traffic, "
    "modify firewall rules, or apply real rate limits. Alert priority "
    "is based on model probability, not verified attack severity."
)

try:
    model, model_features = load_model()
    st.success("CIC-DDoS2019 XGBoost model loaded successfully.")
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.caption(f"Model file: {MODEL_PATH.name}")
st.caption(f"Required input features: {len(model_features)}")

with st.form("analysis_form"):
    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        help="The CSV must contain all 14 model features.",
    )

    threshold = st.slider(
        "Attack detection threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Records with attack probability at or above this "
            "threshold are flagged as attacks."
        ),
    )

    submitted = st.form_submit_button(
        "Analyze traffic and simulate response",
        type="primary",
    )

if submitted:
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    else:
        try:
            with st.spinner(
                "Analyzing traffic and generating simulated alerts..."
            ):
                input_df = pd.read_csv(
                    uploaded_file,
                    low_memory=False,
                )

                results, alerts, metrics = analyze_data(
                    input_df,
                    model,
                    model_features,
                    threshold,
                )

                st.session_state["response_results"] = results
                st.session_state["response_alerts"] = alerts
                st.session_state["response_metrics"] = metrics
                st.session_state["response_threshold"] = threshold
                st.session_state["response_filename"] = uploaded_file.name

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

# ============================================================
# DISPLAY RESULTS
# ============================================================

if "response_results" in st.session_state:
    results = st.session_state["response_results"]
    alerts = st.session_state["response_alerts"]
    metrics = st.session_state["response_metrics"]

    st.divider()
    st.subheader("Traffic analysis results")

    benign_count = int(
        (results["Predicted_Label"] == "BENIGN").sum()
    )
    attack_count = int(
        (results["Predicted_Label"] == "ATTACK").sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total records", f"{len(results):,}")
    col2.metric("Predicted benign", f"{benign_count:,}")
    col3.metric("Attack alerts", f"{attack_count:,}")
    col4.metric(
        "Alert rate",
        f"{attack_count / len(results) * 100:.2f}%"
        if len(results) else "0.00%",
    )

    st.caption(
        f"Detection threshold used: "
        f"{st.session_state['response_threshold']:.2f}"
    )

    if metrics is not None:
        st.subheader("Evaluation metrics")
        st.caption(
            "These metrics are meaningful only when the uploaded "
            "labels are reliable ground truth. Testing on a dataset "
            "used during model development is not independent validation."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.4%}")
        c2.metric(
            "Attack precision",
            f"{metrics['Attack precision']:.4%}",
        )
        c3.metric(
            "Attack recall",
            f"{metrics['Attack recall']:.4%}",
        )
        c4.metric("Attack F1", f"{metrics['Attack F1']:.4%}")

        st.write("Confusion matrix (actual rows, predicted columns)")
        st.dataframe(
            pd.DataFrame(
                metrics["Confusion matrix"],
                index=["Actual benign", "Actual attack"],
                columns=["Predicted benign", "Predicted attack"],
            ),
            use_container_width=True,
        )

    st.subheader("Simulated response summary")

    if alerts.empty:
        st.info("No records crossed the selected attack threshold.")
    else:
        priority_counts = (
            alerts["Alert_Priority"]
            .value_counts()
            .reindex(
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                fill_value=0,
            )
        )

        st.dataframe(
            priority_counts.rename("Alert count"),
            use_container_width=True,
        )

        st.subheader("Attack alert log")

        selected_priorities = st.multiselect(
            "Filter by alert priority",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )

        filtered_alerts = alerts[
            alerts["Alert_Priority"].isin(selected_priorities)
        ]

        st.dataframe(
            filtered_alerts,
            use_container_width=True,
            height=420,
        )

    st.subheader("All predictions")
    st.dataframe(
        results.head(1000),
        use_container_width=True,
        height=300,
    )

    st.caption(
        "The table displays at most the first 1,000 prediction rows. "
        "Use the download buttons to export all rows."
    )

    st.download_button(
        "Download all prediction results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="ddos_prediction_results.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download simulated incident log",
        data=alerts.to_csv(index=False).encode("utf-8"),
        file_name="ddos_simulated_incident_log.csv",
        mime="text/csv",
    )

    st.success(
        "Analysis complete. All response actions shown are simulated; "
        "no real network changes were made."
    )
