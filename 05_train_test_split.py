
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

# Load dataset
DATA_PATH = Path(__file__).parent / "diabetes_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Define target and remove leakage-prone columns
TARGET = "diagnosed_diabetes"

DROP_COLUMNS = [
    "diabetes_stage",
    "diabetes_risk_score"
]

# Separate features and target
X = df.drop(columns=[TARGET] + DROP_COLUMNS)
y = df[TARGET]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("=" * 50)
print("DAY 5: TRAIN-TEST SPLIT")
print("=" * 50)

print("\nOriginal dataset:", X.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTrain-test split completed.")
