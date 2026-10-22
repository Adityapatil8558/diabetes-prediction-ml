
from pathlib import Path

import pandas as pd

# Load the dataset from the project folder
DATA_PATH = Path(__file__).parent / "diabetes_dataset.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place diabetes_dataset.csv beside this Python file."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DIABETES PREDICTION - DATA EXPLORATION")
print("=" * 60)

# 1. Dataset size
print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 2. Column names and data types
print("\n--- COLUMN DATA TYPES ---")
print(df.dtypes.to_string())

# 3. Missing values
print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values")

# 4. Duplicate rows
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# 5. Target distribution
target = "diagnosed_diabetes"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' is missing.")

print("\n--- TARGET DISTRIBUTION ---")
print(df[target].value_counts(dropna=False).sort_index())

print("\n--- TARGET PERCENTAGES ---")
print(
    (df[target].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
    .sort_index()
)

# 6. Numeric summary
print("\n--- NUMERIC SUMMARY ---")
print(df.describe().round(2).to_string())

# 7. Investigate potential leakage
print("\n--- DIABETES STAGE VS TARGET ---")
if "diabetes_stage" in df.columns:
    print(pd.crosstab(
        df["diabetes_stage"],
        df[target],
        margins=True
    ))

print("\n--- RISK SCORE BY TARGET ---")
if "diabetes_risk_score" in df.columns:
    print(df.groupby(target)["diabetes_risk_score"].agg(
        ["count", "mean", "median", "min", "max"]
    ).round(2))

print("\n--- GLUCOSE AND HbA1c BY TARGET ---")
clinical_columns = [
    c for c in [
        "glucose_fasting",
        "glucose_postprandial",
        "insulin_level",
        "hba1c",
        "bmi",
    ]
    if c in df.columns
]

print(df.groupby(target)[clinical_columns].mean().round(2))

print("\nExploration complete.")
