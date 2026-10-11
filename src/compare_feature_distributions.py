
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
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

source_path = ROOT / "data" / (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
external_path = ROOT / "data" / "external" / (
    "cic_ddos2019_test.csv"
)

source = pd.read_csv(source_path, low_memory=False)
external = pd.read_csv(external_path, low_memory=False)

source.columns = source.columns.astype(str).str.strip()
external.columns = external.columns.astype(str).str.strip()

missing_source = [f for f in FEATURES if f not in source.columns]
missing_external = [f for f in FEATURES if f not in external.columns]

if missing_source or missing_external:
    print("Missing in source:", missing_source)
    print("Missing in external:", missing_external)
    raise SystemExit("Feature schema mismatch.")

rows = []

for feature in FEATURES:
    a = pd.to_numeric(source[feature], errors="coerce")
    b = pd.to_numeric(external[feature], errors="coerce")

    a = a.replace([np.inf, -np.inf], np.nan).dropna()
    b = b.replace([np.inf, -np.inf], np.nan).dropna()

    rows.append({
        "Feature": feature,
        "Source median": a.median(),
        "External median": b.median(),
        "Source p95": a.quantile(0.95),
        "External p95": b.quantile(0.95),
        "Source zero %": (a == 0).mean() * 100,
        "External zero %": (b == 0).mean() * 100,
        "Source valid rows": len(a),
        "External valid rows": len(b),
    })

result = pd.DataFrame(rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("\nFEATURE DISTRIBUTION COMPARISON")
print(result.round(4).to_string(index=False))

output = ROOT / "external_feature_comparison.csv"
result.to_csv(output, index=False)
print(f"\nSaved report: {output}")
