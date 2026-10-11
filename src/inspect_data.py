
from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

path = csv_files[0]
print(f"Loading: {path.name}")

df = pd.read_csv(path, low_memory=False)
df.columns = df.columns.astype(str).str.strip()

label_cols = [c for c in df.columns if c.lower() == "label"]
if len(label_cols) != 1:
    raise ValueError(f"Expected one label column; found {label_cols}")

label_col = label_cols[0]
print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nClass distribution:")
print(df[label_col].value_counts(dropna=False))

# pandas automatically adds suffixes such as .1 to duplicate headers.
def canonical_name(name):
    return re.sub(r"\.\d+$", "", name).strip().lower()

header_groups = {}
for col in df.columns:
    header_groups.setdefault(canonical_name(col), []).append(col)

duplicates = {
    key: cols for key, cols in header_groups.items()
    if len(cols) > 1
}

print("\nPotential duplicate header names:")
if duplicates:
    for key, cols in duplicates.items():
        print(f"{key}: {cols}")
else:
    print("None detected.")

# Inspect numeric features without changing the original dataframe.
X = df.drop(columns=[label_col]).copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

print("\nMissing/invalid values in features:", int(X.isna().sum().sum()))
print("Infinite values in features:",
      int(np.isinf(X.to_numpy(dtype=float)).sum()))

print("\nExact duplicate rows:", int(df.duplicated().sum()))
print("Missing labels:", int(df[label_col].isna().sum()))
print("Distinct labels:", df[label_col].nunique(dropna=False))

print("\nInspection complete. Original CSV unchanged.")
