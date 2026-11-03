
from pathlib import Path
import pandas as pd

# Load dataset
DATA_PATH = Path(__file__).parent / "diabetes_dataset.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("DAY 4: DATA QUALITY ANALYSIS")
print("=" * 50)

# Select numerical columns
numeric_df = df.select_dtypes(include="number")

# Display minimum and maximum values
print("\n--- NUMERICAL VALUE RANGES ---")

for column in numeric_df.columns:
    print(
        f"{column}: "
        f"Minimum = {numeric_df[column].min()}, "
        f"Maximum = {numeric_df[column].max()}"
    )

# Check invalid target values
print("\n--- TARGET VALIDATION ---")

print("Unique target values:")
print(df["diagnosed_diabetes"].unique())

print("\nTarget value counts:")
print(df["diagnosed_diabetes"].value_counts())

print("\nData quality check completed.")

# Check potential outliers using the IQR method

print("\n--- POTENTIAL OUTLIER ANALYSIS ---")

columns_to_check = [
    "age",
    "bmi",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
]

for column in columns_to_check:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\nColumn: {column}")
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Potential outliers:", len(outliers))

print("\nOutlier analysis completed.")

# Check categorical feature values

print("\n--- CATEGORICAL VALUE ANALYSIS ---")

categorical_columns = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
]

for column in categorical_columns:
    print(f"\nColumn: {column}")
    print(df[column].value_counts(dropna=False))
    
print("\nCategorical analysis completed.")
