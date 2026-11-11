# Diabetes Prediction Using Machine Learning
## Live Demo

**Try the application here:**  
https://ml-diabetes-project.streamlit.app/

## 1. Project Overview

This project uses machine learning classification algorithms to analyze clinical and lifestyle features and predict diabetes diagnosis.

The main goal is to build and compare different machine learning models and evaluate their performance using standard classification metrics.

## 2. Project Objectives

- Explore and understand the diabetes dataset.
- Perform data cleaning and quality checks.
- Analyze numerical and categorical features.
- Prepare data for machine learning.
- Train and evaluate multiple classification models.
- Compare model performance using evaluation metrics.
- Identify important features influencing model predictions.

## 3. Dataset Description

The project uses `diabetes_dataset.csv`.

- Total records: 100,000
- Original columns: 31
- Input features used: 28
- Target column: `diagnosed_diabetes`
- Target type: Binary classification (0 and 1)

The columns `diabetes_stage` and `diabetes_risk_score` were excluded from model inputs to avoid potential target leakage.

## 4. Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Git and GitHub
- Visual Studio Code

## 5. Project Workflow

1. Data exploration
2. Data preparation
3. Data visualization
4. Data quality checks
5. Train-test splitting
6. Numerical and categorical preprocessing
7. Logistic Regression
8. Decision Tree
9. Random Forest
10. Gradient Boosting
11. Model comparison and cross-validation
12. Hyperparameter tuning
13. Feature importance analysis

## 6. Data Preprocessing

The dataset was divided into 80% training data and 20% testing data using stratified splitting.

The preprocessing steps included:

- Median imputation for numerical features.
- Standard scaling for numerical features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical features.

A Scikit-learn Pipeline was used to combine preprocessing and model training.

## 7. Machine Learning Models

Four classification algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## 8. Model Performance

The following results were obtained on the test dataset:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 86.09% | 87.57% | 89.52% | 88.54% | 93.37% |
| Decision Tree | 85.80% | 87.36% | 89.25% | 88.30% | 84.94% |
| Random Forest | 91.93% | 99.88% | 86.65% | 92.80% | 94.06% |
| Gradient Boosting | 91.98% | 99.99% | 86.65% | 92.84% | 94.45% |

Gradient Boosting achieved the highest test accuracy and F1-score among the four original models.

## 9. Cross-Validation and Hyperparameter Tuning

Three-fold cross-validation was performed to evaluate model performance across different data splits.

Mean F1-scores:

- Logistic Regression: 0.8836
- Decision Tree: 0.8861
- Random Forest: 0.9298
- Gradient Boosting: 0.9299

RandomizedSearchCV was used to tune Random Forest hyperparameters.

Best parameters:

- Number of estimators: 150
- Maximum depth: 20
- Minimum samples split: 2
- Minimum samples leaf: 1

The tuned Random Forest achieved an F1-score of 0.9284 on the test set.

## 10. Feature Importance

Feature importance analysis was performed using the tuned Random Forest model.

The three most important features were:

1. HbA1c
2. Postprandial glucose
3. Fasting glucose

The project includes visualizations of model performance and feature importance.

## 11. Project Visualizations

The following visualizations were generated:

- `model_comparison.png` – comparison of the four machine learning models.
- `feature_importance.png` – top 10 important features identified by Random Forest.

## 12. How to Run the Project

Clone the repository:

```bash
git clone https://github.com/Adityapatil8558/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib
```

Run an individual Python script:

```bash
python 01_data_exploration.py
```

Make sure `diabetes_dataset.csv` is available in the project root before running the scripts.

## 13. Statistical Analysis

A Mann–Whitney U test was performed to compare HbA1c values between the two diagnosis groups.

**Results:**
- Group 0 mean HbA1c: 5.84
- Group 1 mean HbA1c: 6.97
- U Statistic: 162003922.5
- P-value: < 0.001
- Rank-biserial correlation: 0.865

**Interpretation:**

The test showed a statistically significant difference in HbA1c distributions between the two diagnosis groups. The rank-biserial correlation indicates a strong difference between the groups.

This analysis identifies an association in the dataset and does not establish causation.

## 14. Conclusion

Four machine learning classification models were developed and evaluated. Gradient Boosting achieved the strongest overall test-set results among the original models, while Random Forest produced very similar performance.

This project demonstrates data exploration, preprocessing, model development, evaluation, cross-validation, hyperparameter tuning and feature importance analysis.
