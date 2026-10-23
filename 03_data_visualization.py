
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
DATA_PATH = Path(__file__).parent / "diabetes_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Count the target classes
target_counts = df["diagnosed_diabetes"].value_counts().sort_index()

# Create the bar chart
target_counts.plot(
    kind="bar",
    figsize=(7, 5),
    xlabel="Diagnosed diabetes (0 = No, 1 = Yes)",
    ylabel="Number of records",
    title="Distribution of Diabetes Diagnosis"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# Compare clinical measurements by diagnosis
clinical_columns = [
    "glucose_fasting",
    "hba1c",
]

group_means = df.groupby("diagnosed_diabetes")[
    clinical_columns
].mean()

group_means.index = ["No Diabetes (0)", "Diabetes (1)"]

group_means.plot(
    kind="bar",
    figsize=(8, 5),
    xlabel="Diagnosis label",
    ylabel="Mean value",
    title="Average Glucose and HbA1c by Diagnosis"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
