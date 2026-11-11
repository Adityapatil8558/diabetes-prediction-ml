
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "random_forest_pipeline_compressed.pkl"
METADATA_PATH = BASE_DIR / "feature_metadata.json"

# --------------------------------------------------
# LOAD MODEL AND METADATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


with open(METADATA_PATH, "r", encoding="utf-8") as file:
    feature_metadata = json.load(file)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Diabetes ML Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Diabetes Prediction",
        "Data Visualization",
        "Model Performance"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("Diabetes Prediction System")

    st.markdown(
        "An interactive Machine Learning application "
        "for diabetes-related data analysis and prediction."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", "100,000")

    with col2:
        st.metric("Input Features", "28")

    with col3:
        st.metric("ML Models", "4")

    st.divider()

    st.subheader("About This Project")

    st.write(
        "This project uses Machine Learning algorithms "
        "to analyze diabetes-related data and predict "
        "the existing diagnosis label."
    )

    st.subheader("Available Features")

    st.write("- Interactive diabetes prediction")
    st.write("- Exploratory data visualizations")
    st.write("- Machine Learning model comparison")
    st.write("- Feature importance analysis")

    st.info(
        "This application is an educational ML demonstration, "
        "not a medical diagnostic tool."
    )

# --------------------------------------------------
# DIABETES PREDICTION PAGE
# --------------------------------------------------

elif page == "Diabetes Prediction":

    st.title("Diabetes Prediction")

    st.write(
        "Enter the details below to generate a model prediction."
    )

    feature_columns = list(feature_metadata.keys())

    numeric_features = [
        col for col, info in feature_metadata.items()
        if info["type"] == "numeric"
    ]

    categorical_features = [
        col for col, info in feature_metadata.items()
        if info["type"] == "categorical"
    ]

    input_data = {}

    with st.expander(
        "Personal and Demographic Details",
        expanded=True
    ):

        cols = st.columns(3)

        for i, col in enumerate(categorical_features):

            options = feature_metadata[col]["options"]

            with cols[i % 3]:

                input_data[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options,
                    key=f"input_{col}"
                )

    with st.expander(
        "Health and Lifestyle Measurements",
        expanded=True
    ):

        cols = st.columns(3)

        for i, col in enumerate(numeric_features):

            info = feature_metadata[col]

            with cols[i % 3]:

                if info["integer"]:

                    input_data[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=int(info["min"]),
                        max_value=int(info["max"]),
                        value=int(round(info["median"])),
                        step=1,
                        key=f"input_{col}"
                    )

                else:

                    input_data[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=float(info["min"]),
                        max_value=float(info["max"]),
                        value=float(info["median"]),
                        step=0.1,
                        format="%.2f",
                        key=f"input_{col}"
                    )

    if st.button(
        "Predict Diabetes Status",
        type="primary"
    ):

        try:

            model = load_model()

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            prediction = model.predict(input_df)[0]

            probabilities = model.predict_proba(input_df)[0]

            classes = list(model.classes_)

            no_diabetes_probability = (
                probabilities[classes.index(0)] * 100
            )

            diabetes_probability = (
                probabilities[classes.index(1)] * 100
            )

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "Model prediction: Diabetes (Class 1)"
                )

            else:

                st.success(
                    "Model prediction: No Diabetes (Class 0)"
                )

            st.subheader("Model Probability")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "No Diabetes",
                    f"{no_diabetes_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Diabetes",
                    f"{diabetes_probability:.2f}%"
                )

            st.caption(
                "Probabilities are model estimates, not calibrated "
                "clinical risk estimates or medical diagnoses."
            )

            st.warning(
                "This application is for educational purposes only. "
                "It is not a substitute for professional medical advice."
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# DATA VISUALIZATION PAGE
# --------------------------------------------------

elif page == "Data Visualization":

    st.title("Data Visualization")

    st.subheader("Diabetes Diagnosis Distribution")

    diagnosis_counts = pd.Series(
        {
            "No Diabetes": 40002,
            "Diabetes": 59998
        }
    )

    st.bar_chart(diagnosis_counts)

    st.divider()

    st.subheader("Average Clinical Measurements")

    group_means = pd.DataFrame(
        {
            "Fasting Glucose": [110, 126],
            "HbA1c": [5.84, 6.97]
        },
        index=["No Diabetes", "Diabetes"]
    )

    st.bar_chart(group_means)

    st.divider()

    st.subheader("Average HbA1c by Diagnosis")

    hba1c_means = pd.Series(
        {
            "No Diabetes": 5.84,
            "Diabetes": 6.97
        }
    )

    st.bar_chart(hba1c_means)

    st.caption(
        "Charts display summary statistics from the project dataset. "
        "Values are rounded where applicable."
    )

# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance")

    comparison_path = BASE_DIR / "model_comparison.png"

    importance_path = BASE_DIR / "feature_importance.png"

    st.subheader("Model Comparison")

    if comparison_path.exists():

        st.image(
            str(comparison_path),
            caption="Comparison of Machine Learning Models",
            use_container_width=True
        )

    else:

        st.warning("Model comparison image not found.")

    st.divider()

    st.subheader("Feature Importance")

    if importance_path.exists():

        st.image(
            str(importance_path),
            caption="Random Forest Feature Importance",
            use_container_width=True
        )

    else:

        st.warning("Feature importance image not found.")
