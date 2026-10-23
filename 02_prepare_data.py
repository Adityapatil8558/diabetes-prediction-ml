
from pathlib import Path
import pandas as pd

# Load dataset
DATA_PATH = Path(__file__).parent / "diabetes_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Target we want to predict
TARGET = "diagnosed_diabetes"

# Columns that must not be used as model inputs
DROP_COLUMNS = [
    "diabetes_stage",
    "diabetes_risk_score",
]

# Separate input features (X) and target (y)
X = df.drop(columns=[TARGET] + DROP_COLUMNS)
y = df[TARGET]

print("=" * 50)
print("PART 2: DATA PREPARATION")
print("=" * 50)

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures used by the model:")
print(X.columns.tolist())

print("\nTarget values:")
print(y.value_counts().sort_index())

print("\nTarget missing values:", y.isnull().sum())
print("Feature missing values:", X.isnull().sum().sum())

print("\nCategorical columns:")
print(X.select_dtypes(include=["object", "str"]).columns.tolist())

print("\nNumeric columns:")
print(X.select_dtypes(include="number").columns.tolist())

print("\nPart 2 completed.")
