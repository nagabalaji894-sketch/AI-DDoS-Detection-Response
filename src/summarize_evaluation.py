
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

results = pd.DataFrame([
    {
        "Experiment": "CIC-IDS2017 -> CIC-DDoS2019",
        "Accuracy": 0.08264,
        "Attack Precision": 0.00000,
        "Attack Recall": 0.00000,
        "Attack F1": 0.00000,
        "False Positives": 35,
        "False Negatives": 110048,
    },
    {
        "Experiment": "CIC-DDoS2019 internal test split",
        "Accuracy": 0.99989,
        "Attack Precision": 0.99995,
        "Attack Recall": 0.99994,
        "Attack F1": 0.99994,
        "False Positives": 6,
        "False Negatives": 7,
    },
])

output = ROOT / "external_evaluation_summary.csv"
results.to_csv(output, index=False)

print(results.to_string(index=False))
print(f"\nSaved report: {output}")
