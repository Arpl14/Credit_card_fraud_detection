import streamlit as st
import pandas as pd
import random
import joblib
from sklearn.metrics import classification_report
import numpy as np  
import shap
import matplotlib.pyplot as plt

# ----------------- Load Data ------------------
@st.cache_data
def load_data():
    return pd.read_csv("sample_test.csv")  

df = load_data()

# ----------------- Load Models ------------------
@st.cache_resource
def load_models():
    iso_model = joblib.load("iso_forest.pkl")
    xgb_model = joblib.load("xgb_model_clean.pkl")
    return iso_model, xgb_model

iso_model, xgb_model = load_models()

# ----------------- Title & Description ------------------
st.title("Credit Card Fraud Detection")
st.markdown("<h6 style='text-align: right;'>Created by Arpita Lonakadi</h6>", unsafe_allow_html=True)
st.markdown("""
### Project Overview
This app predicts the likelihood of fraudulent transactions using two models:
- **Isolation Forest** (unsupervised anomaly detection)
- **XGBoost** (supervised classification)

### Dataset Information
The dataset contains anonymized credit card transaction features (`V1` to `V28`, `Time`, `Amount`), and a binary label `Class`:
- `Class = 0`: Non-Fraud
- `Class = 1`: Fraud

We use a sample of **992 records**, evenly balanced with **50% fraudulent** and **50% non-fraudulent** transactions to provide balanced prediction evaluation.
""")

# ----------------- Random Sample & Prediction ------------------
if df.empty:
    st.error("Dataset is empty or not loaded.")
else:
    st.write(f"Dataset has **{len(df)}** samples.")

    if "sample" not in st.session_state:
        st.session_state.sample = None
        st.session_state.true_class = None

    if st.button("Show Random Sample"):
        idx = random.randint(0, len(df) - 1)
        st.session_state.sample = df.drop(columns=["Class"]).iloc[idx]
        st.session_state.true_class = df.iloc[idx]["Class"]

    if st.session_state.sample is not None:
        sample = st.session_state.sample
        true_class = st.session_state.true_class

        st.subheader(f"Sample - Actual Class: {'Fraud' if true_class == 1 else 'Non-Fraud'}")
        st.write(sample)

        # Model Predictions
        st.markdown("#### Model Predictions")
        iso_pred = iso_model.predict([sample])[0]
        xgb_pred = xgb_model.predict([sample])[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='color:orange;'>Isolation Forest</h5>", unsafe_allow_html=True)
            st.metric("Prediction", "Fraud" if iso_pred == -1 else "Non-Fraud")

        with col2:
            st.markdown("<h5 style='color:green;'>XGBoost</h5>", unsafe_allow_html=True)
            st.metric("Prediction", "Fraud" if xgb_pred == 1 else "Non-Fraud")

# ----------------- Model Metrics Comparison (Hardcoded) ------------------
st.subheader("Model Performance Comparison")

st.markdown("""
We compared the performance of **XGBoost (AutoML)** with **Isolation Forest** on the same balanced dataset.
The following table highlights key evaluation metrics focused on fraud detection capability.
""")


comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "MCC", "Kappa", "ROC AUC"],
    "Isolation Forest": [0.9978, 0.5147, 0.4367, 0.4310, np.nan],
    "XGBoost (AutoML)": [0.9999, 0.9411, 0.9553, 0.9552, 0.99]
})

comparison_df.set_index("Metric", inplace=True)

# Highlight max only for numeric rows
numeric_metrics_df = comparison_df.drop(index="ROC AUC", errors="ignore")

styled_df = numeric_metrics_df.style \
    .format(precision=4) \
    .highlight_max(axis=1, color="#d4edda")

# Append ROC AUC row manually
styled_df = styled_df.concat(
    comparison_df.loc[["ROC AUC"]].style.format(na_rep="N/A")
)

st.dataframe(styled_df)


#--------------SHAP---------------------



import shap
import matplotlib.pyplot as plt

st.subheader("Feature Contribution: SHAP Explanations")
st.subheader("SHAP Explanation Overview")

st.markdown("""
**SHAP (SHapley Additive exPlanations)** is a method to explain individual predictions by attributing feature importance.

#### 🔍 What the SHAP Waterfall Graphs Show:
- Each graph shows how the model made its prediction for the selected transaction.
- The **left side** of the plot is the **expected value** (baseline prediction across the dataset).
- The **right side** is the **final model prediction** for the sample.
- Features are shown from top to bottom in **order of impact** on the prediction.
- Each bar represents the **contribution of a feature**:
  - **Positive contribution** moves prediction towards *fraud*.
  - **Negative contribution** moves prediction towards *non-fraud*.
- The value next to each feature (e.g., `+1.5`, `-0.4`) is the **SHAP value** — how much that feature influenced the result.
- Red and blue bars represent the **direction and magnitude** of feature impact for each model.

####  Why SHAP is Important:
- Helps you **interpret the decision logic** behind each prediction.
- Adds **transparency** to complex ML models (like XGBoost or Isolation Forest).
- Useful for **audits, debugging, and building user trust** in ML predictions.
""")

if "sample" not in st.session_state:
    st.info("Click 'Show Random Sample' first to activate SHAP explanations.")
else:
    if st.checkbox("Show SHapley Additive exPlanations (SHAP) Explanation for Current Sample"):
        shap_sample = st.session_state.sample.to_frame().T

        # --- SHAP for XGBoost ---
        xgb_explainer = shap.TreeExplainer(xgb_model)
        xgb_shap_values = xgb_explainer.shap_values(shap_sample)

        # --- SHAP for Isolation Forest using KernelExplainer ---
        background = df.drop(columns=["Class"]).sample(100, random_state=42)
        iso_explainer = shap.KernelExplainer(iso_model.predict, background)
        iso_shap_values = iso_explainer.shap_values(shap_sample)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h5 style='color:green;'>XGBoost</h5>", unsafe_allow_html=True)
            fig_xgb, ax = plt.subplots()
            shap.plots._waterfall.waterfall_legacy(
                xgb_explainer.expected_value,
                xgb_shap_values[0],
                features=shap_sample.iloc[0],
                feature_names=shap_sample.columns.tolist(),
                show=False
            )
            st.pyplot(fig_xgb)

        with col2:
            st.markdown("<h5 style='color:orange;'>Isolation Forest</h5>", unsafe_allow_html=True)
            fig_iso, ax = plt.subplots()
            shap.plots._waterfall.waterfall_legacy(
                iso_explainer.expected_value,
                iso_shap_values[0],
                features=shap_sample.iloc[0],
                feature_names=shap_sample.columns.tolist(),
                show=False
            )
            st.pyplot(fig_iso)

