import pandas as pd
from scipy.stats import mannwhitneyu
# Load the dataset
df = pd.read_csv("diabetes_dataset.csv")
# Display basic information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nDiagnosis group counts:")
print(df["diagnosed_diabetes"].value_counts())
# Separate HbA1c values by diagnosis group
group_0 = df[df["diagnosed_diabetes"] == 0]["hba1c"]
group_1 = df[df["diagnosed_diabetes"] == 1]["hba1c"]

# Calculate descriptive statistics
print("\nHbA1c Statistics by Diagnosis Group:")

print("\nGroup 0:")
print(group_0.describe())

print("\nGroup 1:")
print(group_1.describe())
# Perform Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(
    group_0,
    group_1,
    alternative="two-sided"
)

print("\nMann-Whitney U Test Results:")
print("U Statistic:", u_statistic)
print("P-value: < 0.001" if p_value < 0.001 else f"P-value: {p_value:.6f}")

# Interpret the result
if p_value < 0.05:
    print("Result: There is a statistically significant difference in HbA1c values between the two groups.")
else:
    print("Result: There is no statistically significant difference in HbA1c values between the two groups.")
    # Calculate rank-biserial correlation
n0 = len(group_0)
n1 = len(group_1)

effect_size = 1 - (2 * u_statistic) / (n0 * n1)

print("Rank-biserial correlation:", round(effect_size, 4))