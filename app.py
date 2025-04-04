import streamlit as st
import pandas as pd
import random
import joblib
from sklearn.metrics import classification_report

# ----------------- Load Data ------------------
@st.cache_data
def load_data():
    return pd.read_csv("sample_test.csv")  # Make sure this exists in your repo

df = load_data()

# ----------------- Load Models ------------------
@st.cache_resource
def load_models():
    iso_model = joblib.load("iso_forest.pkl")
    xgb_model = joblib.load("xgb_model_clean.pkl")
    return iso_model, xgb_model

iso_model, xgb_model = load_models()

# ----------------- App Title & Descriptions ------------------
st.title("💳 Credit Card Fraud Detection")

st.markdown("""
### 📝 Project Overview
This app demonstrates a credit card fraud detection system using two machine learning models:
- **Isolation Forest** for unsupervised anomaly detection
- **XGBoost** for supervised classification

### 📂 Dataset
The dataset includes anonymized transaction features and a target variable `Class`, where:
- `1` = Fraudulent
- `0` = Non-Fraudulent
""")

# ----------------- Random Sample ------------------
if df.empty:
    st.error("Dataset is empty or not loaded.")
else:
    st.write(f"✅ Dataset has **{len(df)}** samples.")

    if st.button("🎲 Show Random Sample"):
        idx = random.randint(0, len(df) - 1)
        sample = df.drop(columns=["Class"]).iloc[idx]
        true_class = df.iloc[idx]["Class"]

        st.subheader(f"🔍 Sample #{idx}")
        st.write(sample)

        # ----------------- Model Predictions ------------------
        st.subheader("🔮 Model Predictions")
        iso_pred = iso_model.predict([sample])[0]
        xgb_pred = xgb_model.predict([sample])[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Isolation Forest", "Fraud" if iso_pred == 1 else "Non-Fraud")
        with col2:
            st.metric("XGBoost", "Fraud" if xgb_pred == 1 else "Non-Fraud")

# ----------------- Model Comparison Table ------------------
st.subheader("📊 Model Performance Comparison")

@st.cache_data
def compute_metrics():
    y_true = df["Class"]
    X = df.drop(columns=["Class"])

    iso_preds = iso_model.predict(X)
    xgb_preds = xgb_model.predict(X)

    # For IsolationForest: anomaly = -1 → fraud (1), normal = 1 → non-fraud (0)
    iso_preds = [1 if p == -1 else 0 for p in iso_preds]

    iso_report = classification_report(y_true, iso_preds, output_dict=True, zero_division=0)
    xgb_report = classification_report(y_true, xgb_preds, output_dict=True, zero_division=0)

    metrics = pd.DataFrame({
        "Model": ["Isolation Forest", "XGBoost"],
        "Accuracy": [iso_report["accuracy"], xgb_report["accuracy"]],
        "Precision (Fraud)": [iso_report["1"]["precision"], xgb_report["1"]["precision"]],
        "Recall (Fraud)": [iso_report["1"]["recall"], xgb_report["1"]["recall"]],
        "F1 Score (Fraud)": [iso_report["1"]["f1-score"], xgb_report["1"]["f1-score"]],
    })

    return metrics.round(3)

metrics_df = compute_metrics()
st.dataframe(metrics_df)









# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from pycaret.classification import predict_model
# import shap
# import matplotlib.pyplot as plt

# # Load models
# xgb_model = joblib.load("models/xgb_model.pkl")
# iso_forest = joblib.load("models/iso_forest.pkl")

# # Load sample test data (for random fill)
# test_data = pd.read_csv("data/sample_test_data.csv")

# # Features to be used for input
# selected_features = ['V17', 'V14', 'V10', 'V12', 'V7', 'V4', 'V11', 'V3', 'V16', 'Amount', 'Time']

# # --- Sidebar Input Function ---
# def get_random_sample():
#     return test_data[selected_features].sample(1).iloc[0].to_dict()

# def user_input_sidebar():
#     st.sidebar.header("🔧 Simulate Transaction")
#     use_random = st.sidebar.checkbox("🎲 Use Random Sample")

#     if use_random:
#         input_data = get_random_sample()
#     else:
#         input_data = {feat: st.sidebar.number_input(f"{feat}", value=float(test_data[feat].median())) for feat in selected_features}

#     return pd.DataFrame([input_data])

# # --- Streamlit UI ---
# st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
# st.title("💳 Credit Card Fraud Detection App")
# st.markdown("Predict fraudulent transactions using both XGBoost (Supervised) and Isolation Forest (Unsupervised).")

# # Sidebar input
# input_df = user_input_sidebar()

# st.subheader("🔍 Input Transaction")
# st.write(input_df)

# # --- Model Predictions ---

# xgb_pred = predict_model(xgb_model, data=input_df)
# iso_raw = iso_forest.predict(input_df)
# iso_pred = np.where(iso_raw == 1, 0, 1)

# st.subheader("🧠 Predictions")
# col1, col2 = st.columns(2)

# with col1:
#     st.metric("XGBoost Prediction", "Fraud" if int(xgb_pred['prediction_label'][0]) == 1 else "Not Fraud")
#     st.write("Confidence Score:", round(xgb_pred['prediction_score'][0], 4))

# with col2:
#     st.metric("Isolation Forest Prediction", "Fraud" if iso_pred[0] == 1 else "Not Fraud")
#     st.write("(Unsupervised, no probability score)")





# # SHAP explainability (after predictions)
# st.subheader("🔎 SHAP Explanation (XGBoost)")

# # Toggle to show explanation
# if st.checkbox("Show SHAP explanation for XGBoost prediction"):
#     # Extract underlying model (PyCaret uses a pipeline)
#     xgb_native_model = xgb_model.steps[-1][1]

#     # Initialize explainer
#     explainer = shap.TreeExplainer(xgb_native_model)

#     # Get SHAP values for the single input
#     shap_values = explainer.shap_values(input_df)

#     # Plot SHAP values
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     st.pyplot(shap.plots._waterfall.waterfall_legacy(
#         explainer.expected_value,
#         shap_values[0],
#         features=input_df.iloc[0],
#         feature_names=input_df.columns.tolist(),
#         max_display=10
#     ))
