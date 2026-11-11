# Import required libraries
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
# Load dataset

df = pd.read_csv("diabetes_dataset.csv")

# Separate features and target

X = df.drop(
    columns=["diagnosed_diabetes", "diabetes_stage", "diabetes_risk_score"]
)
y = df["diagnosed_diabetes"]

# Identify numerical and categorical features

numerical_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(include=["object", "str"]).columns

print("Dataset loaded successfully!")
print("Total features:", X.shape[1])
# Preprocessing numerical features

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing categorical features

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
# Define machine learning models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

print("Four models are ready for comparison.")
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
# Store model evaluation results

results = []

# Train and evaluate each model

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    print(f"{name} evaluation completed.")
    # Create comparison DataFrame

results_df = pd.DataFrame(results)

# Display results sorted by F1 Score

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nMODEL COMPARISON RESULTS")
print("========================")
print(results_df.round(4).to_string(index=False))
# Visualize model performance

metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Performance Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("\nComparison chart saved as model_comparison.png")
# Perform 3-fold cross-validation

print("\nCROSS-VALIDATION RESULTS")
print("========================")

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=3,
        scoring="f1"
    )

    print(f"\n{name}")
    print("F1 scores:", scores.round(4))
    print(f"Mean F1-score: {scores.mean():.4f}")
    # Hyperparameter tuning for Random Forest

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_grid,
    n_iter=6,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nStarting Random Forest hyperparameter tuning...")

random_search.fit(X_train, y_train)

print("\nBest parameters:")
print(random_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(random_search.best_score_, 4))
# Evaluate the tuned Random Forest on test data

best_model = random_search.best_estimator_

tuned_predictions = best_model.predict(X_test)
tuned_probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nTUNED RANDOM FOREST TEST RESULTS")
print("================================")
print(f"Accuracy: {accuracy_score(y_test, tuned_predictions):.4f}")
print(f"Precision: {precision_score(y_test, tuned_predictions):.4f}")
print(f"Recall: {recall_score(y_test, tuned_predictions):.4f}")
print(f"F1 Score: {f1_score(y_test, tuned_predictions):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, tuned_probabilities):.4f}")
# Analyze feature importance

feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = best_model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print("=========================")
print(importance_df.head(10).to_string(index=False))

# Plot feature importance

importance_df.head(10).plot(
    x="Feature",
    y="Importance",
    kind="barh",
    figsize=(10, 6),
    legend=False
)

plt.title("Top 10 Important Features - Tuned Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)
plt.show()# Analyze feature importance

feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = best_model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print("=========================")
print(importance_df.head(10).to_string(index=False))

# Plot feature importance

importance_df.head(10).plot(
    x="Feature",
    y="Importance",
    kind="barh",
    figsize=(10, 6),
    legend=False
)

plt.title("Top 10 Important Features - Tuned Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)
plt.show()