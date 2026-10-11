from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgboost_cicddos2019.joblib"
DATA_PATH = ROOT / "data" / "external" / "cic_ddos2019_test.csv"

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

st.set_page_config(
    page_title="Live-Style DDoS Monitor",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Live-Style DDoS Traffic Monitoring")
st.caption(
    "Replay-based demonstration using saved CIC-DDoS2019 records and your trained model. "
    "It does not capture live packets or change network/firewall settings."
)

st.warning(
    "**SIMULATION / DATASET REPLAY ONLY.** Traffic records are sampled from the local "
    "CIC-DDoS2019 test CSV and presented as a live-style feed. No real network traffic is "
    "captured, blocked, rate-limited, or modified."
)


@st.cache_resource
def load_model(path: str) -> Any:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            "The model file must be a dictionary containing a 'pipeline' entry."
        )
    if "feature_columns" in bundle:
        model_features = list(bundle["feature_columns"])
    else:
        model_features = FEATURES
    return bundle["pipeline"], model_features


@st.cache_data
def load_replay_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time  # Used to invalidate cache when the file changes.
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.astype(str).str.strip()
    missing = [col for col in FEATURES if col not in data.columns]
    if missing:
        raise ValueError(
            "Replay CSV is missing required model features: " + ", ".join(missing)
        )
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    return data


def score_batch(
    batch: pd.DataFrame,
    pipeline: Any,
    model_features: list[str],
    threshold: float,
    first_id: int,
) -> pd.DataFrame:
    X = batch[model_features].copy()
    probabilities = pipeline.predict_proba(X)[:, 1]
    output = pd.DataFrame(
        {
            "Event ID": np.arange(first_id, first_id + len(batch)),
            "Event time": datetime.now().strftime("%H:%M:%S"),
            "Attack probability": probabilities,
            "Prediction": np.where(probabilities >= threshold, "ATTACK", "BENIGN"),
        }
    )

    # Add contextual columns only when available in the replay dataset.
    for col in ["Source IP", "Src IP", "Destination IP", "Dst IP", "Destination Port", "Protocol", "Flow ID"]:
        if col in batch.columns and col not in output.columns:
            output[col] = batch[col].astype(str).to_numpy()

    output["Priority"] = np.select(
        [
            (probabilities >= threshold) & (probabilities >= 0.95),
            (probabilities >= threshold) & (probabilities >= 0.80),
            (probabilities >= threshold) & (probabilities >= 0.65),
            probabilities >= threshold,
        ],
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default="—",
    )
    output["Proposed response (simulated)"] = np.select(
        [
            output["Priority"].eq("CRITICAL"),
            output["Priority"].eq("HIGH"),
            output["Priority"].eq("MEDIUM"),
            output["Priority"].eq("LOW"),
        ],
        [
            "Temporary block (simulation only)",
            "Rate-limit (simulation only)",
            "Raise alert (no network action)",
            "Monitor suspicious flow",
        ],
        default="No response proposed",
    )

    if "Label" in batch.columns:
        output["Dataset label (reference only)"] = batch["Label"].astype(str).to_numpy()
    return output


def priority_counts(data: pd.DataFrame) -> pd.DataFrame:
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    counts = data["Priority"].value_counts().reindex(order, fill_value=0)
    return counts.rename_axis("Priority").reset_index(name="Alerts")


if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.info("Run this app from the project root and check that the CIC-DDoS2019 model exists in models/.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Replay dataset not found: `{DATA_PATH}`")
    st.info("Place `cic_ddos2019_test.csv` in `data/external/` or update DATA_PATH in this file.")
    st.stop()

try:
    pipeline, model_features = load_model(str(MODEL_PATH))
    replay_data = load_replay_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not initialize the dashboard: {exc}")
    st.stop()

missing_model_features = [col for col in model_features if col not in replay_data.columns]
if missing_model_features:
    st.error("The replay data does not contain all model features: " + ", ".join(missing_model_features))
    st.stop()

with st.sidebar:
    st.header("Simulation controls")
    threshold = st.slider("Attack detection threshold", 0.10, 0.90, 0.50, 0.05)
    batch_size = st.select_slider("Records per batch", options=[25, 50, 100, 250, 500], value=100)
    batches_to_run = st.slider("Batches per run", 1, 20, 8)
    interval = st.slider("Delay between batches (seconds)", 0.0, 2.0, 0.5, 0.1)
    st.divider()
    st.caption(f"Model: `{MODEL_PATH.name}`")
    st.caption(f"Replay records available: {len(replay_data):,}")
    st.caption(f"Model features: {len(model_features)}")
    st.caption("Source: local CIC-DDoS2019 test CSV")

if "event_log" not in st.session_state:
    st.session_state.event_log = pd.DataFrame()
if "next_event_id" not in st.session_state:
    st.session_state.next_event_id = 1
if "replay_cursor" not in st.session_state:
    st.session_state.replay_cursor = 0
if "last_run_at" not in st.session_state:
    st.session_state.last_run_at = "Not started"

control_col1, control_col2, control_col3 = st.columns([1.2, 1, 2])
with control_col1:
    run_clicked = st.button("▶ Run simulation", type="primary", use_container_width=True)
with control_col2:
    clear_clicked = st.button("Clear session", use_container_width=True)
with control_col3:
    st.caption("Each run replays saved records in small batches to imitate a streaming feed.")

if clear_clicked:
    st.session_state.event_log = pd.DataFrame()
    st.session_state.next_event_id = 1
    st.session_state.replay_cursor = 0
    st.session_state.last_run_at = "Cleared"
    st.rerun()

status_slot = st.empty()
progress_slot = st.empty()
kpi_slot = st.container()
chart_slot = st.container()
feed_slot = st.container()

if run_clicked:
    progress = progress_slot.progress(0, text="Preparing replay batches…")
    for batch_index in range(batches_to_run):
        start = st.session_state.replay_cursor
        stop = start + batch_size

        if stop <= len(replay_data):
            batch = replay_data.iloc[start:stop]
        else:
            first_part = replay_data.iloc[start:]
            remaining = stop - len(replay_data)
            second_part = replay_data.iloc[:remaining]
            batch = pd.concat([first_part, second_part], ignore_index=True)

        new_events = score_batch(
            batch=batch,
            pipeline=pipeline,
            model_features=model_features,
            threshold=threshold,
            first_id=st.session_state.next_event_id,
        )
        st.session_state.next_event_id += len(new_events)
        st.session_state.replay_cursor = (start + batch_size) % len(replay_data)

        current_log = st.session_state.event_log
        st.session_state.event_log = (
            pd.concat([current_log, new_events], ignore_index=True)
            .tail(5000)
            .reset_index(drop=True)
        )
        st.session_state.last_run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        progress.progress(
            (batch_index + 1) / batches_to_run,
            text=f"Replaying batch {batch_index + 1} of {batches_to_run}…",
        )
        status_slot.info(
            f"Processed {len(st.session_state.event_log):,} events in this session. "
            f"Latest batch: {len(new_events):,} records."
        )
        time.sleep(interval)

    progress_slot.empty()
    status_slot.success("Simulation run complete. All response actions remain simulated.")

events = st.session_state.event_log.copy()

if events.empty:
    st.info("Select the simulation settings in the sidebar and click **Run simulation** to generate the traffic feed.")
else:
    total = len(events)
    attacks = int(events["Prediction"].eq("ATTACK").sum())
    benign = total - attacks
    alert_rate = attacks / total * 100 if total else 0
    critical = int(events["Priority"].eq("CRITICAL").sum())
    high = int(events["Priority"].eq("HIGH").sum())

    with kpi_slot:
        st.subheader("Monitoring overview")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Events processed", f"{total:,}")
        k2.metric("Attack alerts", f"{attacks:,}")
        k3.metric("Predicted benign", f"{benign:,}")
        k4.metric("Alert rate", f"{alert_rate:.2f}%")
        k5.metric("Critical / high", f"{critical + high:,}")

    with chart_slot:
        left, right = st.columns(2)
        with left:
            st.subheader("Prediction distribution")
            distribution = events["Prediction"].value_counts().reindex(["ATTACK", "BENIGN"], fill_value=0)
            st.bar_chart(distribution)
        with right:
            st.subheader("Alert priority distribution")
            st.bar_chart(priority_counts(events).set_index("Priority"))

        st.subheader("Attack probability over recent events")
        recent = events.tail(250).set_index("Event ID")[["Attack probability"]]
        st.line_chart(recent, y="Attack probability")

    with feed_slot:
        st.subheader("Recent alert feed")
        alerts = events.loc[events["Prediction"].eq("ATTACK")].copy()
        if alerts.empty:
            st.write("No attack alerts in the current session.")
        else:
            filter_priority = st.multiselect(
                "Filter alert priorities",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            )
            visible_alerts = alerts.loc[alerts["Priority"].isin(filter_priority)]
            st.dataframe(visible_alerts.tail(100).iloc[::-1], use_container_width=True, hide_index=True)

        st.subheader("Full session event log")
        st.caption("Only the most recent 5,000 events are retained in this browser session.")
        st.dataframe(events.tail(200).iloc[::-1], use_container_width=True, hide_index=True)

        csv_bytes = events.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download session event log (CSV)",
            data=csv_bytes,
            file_name="ddos_simulated_live_event_log.csv",
            mime="text/csv",
        )

st.caption(f"Last simulation run: {st.session_state.last_run_at}")
st.caption(
    "Interpretation note: probability-based alert priority is a demonstration policy, not a "
    "validated measure of attack impact. For a defensible evaluation, use a separate dataset "
    "that was not used for training or threshold tuning."
)
