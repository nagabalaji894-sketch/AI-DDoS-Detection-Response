
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

df = pd.read_csv(csv_files[0], low_memory=False)
df.columns = df.columns.astype(str).str.strip()

col_a = "Fwd Header Length"
col_b = "Fwd Header Length.1"

print("Dataset rows:", len(df))

if col_a in df.columns and col_b in df.columns:
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    comparable = a.notna() & b.notna()
    equal = a[comparable].eq(b[comparable])

    print("\nDuplicate column comparison")
    print("Comparable rows:", int(comparable.sum()))
    print("Equal values:", int(equal.sum()))
    print("Different values:", int((~equal).sum()))
    print("Missing values in first column:", int(a.isna().sum()))
    print("Missing values in second column:", int(b.isna().sum()))

    if (~equal).any():
        print("\nFirst differing examples:")
        print(df.loc[comparable & a.ne(b), [col_a, col_b]].head(10))
else:
    print("Expected duplicate columns were not found.")
    print("Available columns:", df.columns.tolist())

print("\nCheck complete. Original CSV unchanged.")
