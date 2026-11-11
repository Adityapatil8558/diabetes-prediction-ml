
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
# Correlation analysis of numerical features

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(14, 10))

plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()
# HbA1c distribution by diabetes diagnosis

hba1c_no_diabetes = df[df["diagnosed_diabetes"] == 0]["hba1c"]
hba1c_diabetes = df[df["diagnosed_diabetes"] == 1]["hba1c"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [hba1c_no_diabetes, hba1c_diabetes],
    tick_labels=["No Diabetes", "Diabetes"]
)

plt.xlabel("Diagnosis")
plt.ylabel("HbA1c")
plt.title("HbA1c Distribution by Diabetes Diagnosis")

plt.tight_layout()
plt.show()
