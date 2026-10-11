
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgboost_ddos.joblib"

st.set_page_config(
    page_title="AI DDoS Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI-Based DDoS Detection")
st.caption("Machine-learning network traffic analysis dashboard")

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["pipeline"], saved["feature_columns"]

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

try:
    pipeline, feature_columns = load_model()
    st.success("XGBoost model loaded successfully.")
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

st.info(
    "Upload a network-flow CSV to classify records. "
    "This application does not monitor live traffic or block connections."
)

uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        df.columns = df.columns.astype(str).str.strip()

        st.subheader("Dataset preview")
        st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        st.dataframe(df.head(10), use_container_width=True)

        label_cols = [c for c in df.columns if c.lower() == "label"]
        actual_labels = (
            df[label_cols[0]].astype("string").str.strip()
            if label_cols else None
        )

        X = df.drop(columns=label_cols, errors="ignore").copy()
        X = X.drop(columns=["Fwd Header Length.1"], errors="ignore")

        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            st.error(f"Missing required model features: {missing[:15]}")
            st.stop()

        X = X[feature_columns].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        if st.button("Analyze network traffic", type="primary"):
            with st.spinner("Analyzing traffic..."):
                predictions = pipeline.predict(X)

            results = df.copy()
            results["Predicted Traffic"] = np.where(
                predictions == 1, "DDoS", "BENIGN"
            )

            ddos_count = int(np.sum(predictions == 1))
            benign_count = len(predictions) - ddos_count

            st.subheader("Detection summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total records", f"{len(predictions):,}")
            c2.metric("Predicted DDoS", f"{ddos_count:,}")
            c3.metric("Predicted benign", f"{benign_count:,}")

            st.subheader("Predicted traffic distribution")
            fig, ax = plt.subplots()
            ax.bar(
                ["BENIGN", "DDoS"],
                [benign_count, ddos_count],
            )
            ax.set_ylabel("Number of records")
            ax.set_title("Model predictions")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("DDoS alert log")
            alerts = results[
                results["Predicted Traffic"] == "DDoS"
            ].copy()

            if alerts.empty:
                st.success("No records predicted as DDoS.")
            else:
                st.warning(
                    f"{len(alerts):,} record(s) classified as DDoS. "
                    "Review them before taking action."
                )

                limit = st.selectbox(
                    "Number of alert rows to display",
                    [25, 100, 500, 1000],
                    index=1,
                )
                st.dataframe(
                    alerts.head(limit),
                    use_container_width=True,
                )

                st.download_button(
                    "Download DDoS alert log",
                    data=alerts.to_csv(index=False).encode("utf-8"),
                    file_name="ddos_alert_log.csv",
                    mime="text/csv",
                )

            st.subheader("Simulated defensive response")
            st.write(
                f"Would generate {ddos_count:,} alert(s). "
                "No network connections are blocked."
            )

            st.download_button(
                "Download all prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="ddos_detection_results.csv",
                mime="text/csv",
            )

            # Evaluation is shown only when supplied labels are recognized.
            if actual_labels is not None:
                valid = actual_labels.isin(["BENIGN", "DDoS"])
                if valid.any():
                    st.subheader("Evaluation against supplied labels")
                    st.caption(
                        "These metrics are only independent evidence "
                        "when the uploaded records were not used for "
                        "training or model selection."
                    )

                    from sklearn.metrics import (
                        accuracy_score,
                        precision_score,
                        recall_score,
                        f1_score,
                        confusion_matrix,
                    )

                    y_true = actual_labels[valid].to_numpy()
                    y_pred = results.loc[
                        valid, "Predicted Traffic"
                    ].to_numpy()

                    a, b, c, d = st.columns(4)
                    a.metric(
                        "Accuracy",
                        f"{accuracy_score(y_true, y_pred):.4%}",
                    )
                    b.metric(
                        "DDoS precision",
                        f"{precision_score(y_true, y_pred, pos_label='DDoS', zero_division=0):.4%}",
                    )
                    c.metric(
                        "DDoS recall",
                        f"{recall_score(y_true, y_pred, pos_label='DDoS', zero_division=0):.4%}",
                    )
                    d.metric(
                        "DDoS F1",
                        f"{f1_score(y_true, y_pred, pos_label='DDoS', zero_division=0):.4%}",
                    )

                    cm = confusion_matrix(
                        y_true,
                        y_pred,
                        labels=["BENIGN", "DDoS"],
                    )
                    st.write("Confusion matrix: actual rows, predicted columns")
                    st.dataframe(
                        pd.DataFrame(
                            cm,
                            index=["Actual BENIGN", "Actual DDoS"],
                            columns=["Predicted BENIGN", "Predicted DDoS"],
                        ),
                        use_container_width=True,
                    )

    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")
