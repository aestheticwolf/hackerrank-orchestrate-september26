import pandas as pd
from pathlib import Path

DATASET = Path("dataset")

for file in DATASET.glob("*.csv"):
    print("\n" + "=" * 80)
    print(file.name)
    print("=" * 80)

    df = pd.read_csv(file)

    print("Rows:", len(df))
    print("Columns:")
    print(df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))