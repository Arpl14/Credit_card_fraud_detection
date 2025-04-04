import streamlit as st
import pandas as pd
import random

# Load your test dataset
@st.cache_data
def load_data():
    # Replace with your actual test set file path
    return pd.read_csv("sample_test.csv")

df = load_data()

st.title("Credit Card Fraud Detection - Sample Viewer")

if df.empty:
    st.error("Dataset is empty or not loaded.")
else:
    # Button to pick a random sample
    if st.button("Show Random Sample"):
        idx = random.randint(0, len(df) - 1)
        st.subheader(f"Sample #{idx}")
        st.write(df.iloc[idx])





























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
