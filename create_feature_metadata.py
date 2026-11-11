
import json
from pathlib import Path
import pandas as pd

# File paths
DATA_PATH = Path(__file__).parent / "diabetes_dataset.csv"
OUTPUT_PATH = Path(__file__).parent / "feature_metadata.json"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove target and leakage-related columns
excluded_columns = [
    "diagnosed_diabetes",
    "diabetes_stage",
    "diabetes_risk_score"
]

features = df.drop(columns=excluded_columns)

# Create feature metadata
metadata = {}

for column in features.columns:

    if pd.api.types.is_numeric_dtype(features[column]):
        values = features[column].dropna()

        metadata[column] = {
            "type": "numeric",
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "integer": pd.api.types.is_integer_dtype(features[column])
        }

    else:
        metadata[column] = {
            "type": "categorical",
            "options": sorted(
                features[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        }

# Save metadata as JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Feature metadata created successfully!")
print("Total features:", len(metadata))
print("Output file:", OUTPUT_PATH)
