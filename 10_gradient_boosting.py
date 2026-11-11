# Import required libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# Load dataset

df = pd.read_csv("diabetes_dataset.csv")

# Separate features and target
X = df.drop(
    columns=["diagnosed_diabetes", "diabetes_stage", "diabetes_risk_score"]
)
y = df["diagnosed_diabetes"]

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(include=["object", "str"]).columns

print("Dataset loaded successfully!")
print("Features:", X.shape[1])
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))
# Preprocessing for numerical features

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for categorical features

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine preprocessing steps

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])
# Create Gradient Boosting classifier

gradient_boosting_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Build complete machine learning pipeline

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", gradient_boosting_model)
])
# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
# Train the Gradient Boosting model

model_pipeline.fit(X_train, y_train)

# Make predictions

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

print("Gradient Boosting model trained successfully!")
# Evaluate model performance

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nGradient Boosting Model Performance")
print("-----------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Confusion matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
