
from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

dataset_path = csv_files[0]
print(f"Dataset: {dataset_path.name}")

# Read the original header without modifying the CSV.
header = pd.read_csv(dataset_path, nrows=0, low_memory=False)
original_columns = list(header.columns)

# Find the original column name, including any leading spaces.
label_column = next(
    (col for col in original_columns
     if str(col).strip().lower() == "label"),
    None
)

if label_column is None:
    print("Available columns:")
    print([repr(col) for col in original_columns])
    raise ValueError("Could not find the Label column.")

print(f"Label column found: {label_column!r}")
print("Counting labels across the complete dataset...")

label_counts = Counter()
total_rows = 0

for chunk in pd.read_csv(
    dataset_path,
    usecols=[label_column],
    chunksize=100_000,
    low_memory=False
):
    # Find the selected column as pandas returned it.
    actual_column = chunk.columns[0]
    labels = chunk[actual_column].astype("string").str.strip()

    label_counts.update(labels.value_counts(dropna=False).to_dict())
    total_rows += len(chunk)

print(f"\nTotal rows: {total_rows:,}")
print("\nLabel distribution:")

for label, count in label_counts.most_common():
    percentage = count / total_rows * 100 if total_rows else 0
    print(f"{label}: {count:,} ({percentage:.2f}%)")

print("\nInspection complete.")
