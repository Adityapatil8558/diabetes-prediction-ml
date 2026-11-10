import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

# Separate target and input features
target = "diagnosed_diabetes"

X = df.drop(columns=[
    target,
    "diabetes_stage",
    "diabetes_risk_score"
])

y = df[target]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns

categorical_features = X.select_dtypes(include=["str"]).columns

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
# Numerical preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# Categorical preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
# Create Random Forest classifier
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
# Create the complete machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", random_forest_model)
])
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Train the Random Forest model
model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")
# Make predictions on test data
y_pred = model.predict(X_test)

print("\nPredictions completed successfully!")
print("Total predictions:", len(y_pred))
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Random Forest Evaluation ---")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
# Display final model summary
print("\n========== FINAL MODEL SUMMARY ==========")
print("Model: Random Forest")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")
print("Total Test Samples:", len(y_test))
print("Incorrect Predictions:", (y_test != y_pred).sum())