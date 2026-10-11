
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

files = {
    "CIC-IDS2017": ROOT / "data" / (
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ),
    "CIC-DDoS2019": ROOT / "data" / "external" / (
        "cic_ddos2019_test.csv"
    ),
}

features = [
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

for name, path in files.items():
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    label_col = next(
        (c for c in df.columns if c.lower() == "label"),
        None,
    )

    if label_col is None:
        print("ERROR: Label column not found.")
        continue

    df[label_col] = df[label_col].astype(str).str.strip()

    print("\nLabels:")
    print(df[label_col].value_counts().to_string())

    print("\nFirst 5 rows of selected features:")
    available = [c for c in features if c in df.columns]
    print(df[available + [label_col]].head().to_string(index=False))

    print("\nFeature statistics by label:")
    for label in df[label_col].value_counts().index:
        subset = df.loc[df[label_col] == label, available]
        print(f"\nLabel: {label} | rows: {len(subset)}")

        for feature in available:
            values = pd.to_numeric(
                subset[feature], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if len(values) == 0:
                print(f"  {feature}: no valid values")
                continue

            print(
                f"  {feature}: "
                f"median={values.median():.4g}, "
                f"p95={values.quantile(0.95):.4g}, "
                f"min={values.min():.4g}, "
                f"max={values.max():.4g}"
            )
