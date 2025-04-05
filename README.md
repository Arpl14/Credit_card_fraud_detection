# 💳 Credit Card Fraud Detection App

A web app to explore and understand **fraudulent credit card transactions** using both **unsupervised anomaly detection** and **supervised AutoML classification** — poIred by interpretable ML techniques like **SHAP**.
(https://creditcardfrauddetection-pbmxavdkqko5shkrq42tie.streamlit.app/)

---

## Project Overview

Credit card fraud is a rising concern in financial systems, often involving small undetected amounts that accumulate into major losses. The goal of this project is to build and compare ML models to effectively detect such frauds.

I explored **state-of-the-art machine learning techniques** for fraud detection — starting from **unsupervised methods** like **Isolation Forest** and **One-Class SVM**, to **supervised AutoML pipelines** using **PyCaret**. Ultimately, the most performant models were integrated into a **Streamlit dashboard** for interactive prediction, model explainability (SHAP), and evaluation.

---

##  Key Features & Issues Tackled

- Tackled **extreme class imbalance** (0.172% fraud cases) using appropriate metrics.
- Compared **unsupervised vs. supervised learning** for fraud detection.
- Used **AutoML (PyCaret)** to streamline model selection and tuning.
- Incorporated **explainability with SHAP** to break down prediction logic.
- Created a fully **interactive frontend** for experimentation.
- Used metrics beyond confusion matrix: **Recall**, **MCC**, **Kappa**, and **AUC**.

---

## ⚙️ Technical Implementation

> **Tech Stack & Tools:**

- 🤖 **AutoML -PyCaret** **scikit-learn**, **XGBoost**, **OneClassSVM** , **Isolation Forest**
- 🧮 **Python** · **Pandas** · **NumPy** · **Matplotlib** · **Seaborn**
- 📊 **SHAP (SHapley Additive Explanations)** for model interpretability
- 🌐 **Streamlit** for interactive UI
- 🧪 **Classification Metrics**: Recall, F1-Score, MCC, Kappa, AUC, Precision,

> **Skills & Concepts:**

- Anomaly detection, supervised learning, AutoML, model tuning
- Explainable AI (XAI), feature attribution
- Data preprocessing, EDA, outlier analysis
- Imbalanced classification, evaluation under skeId class distributions

---

## 🚀 Project Workflow

### 1. 📊 Dataset

- Sourced from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 total transactions
- Only 492 (0.172%) are fraudulent
- Features:
  - `V1` to `V28`: PCA-transformed inputs
  - `Time`: Elapsed time
  - `Amount`: Transaction amount
  - `Class`: Target (0 = non-fraud, 1 = fraud)

---

### 2. 🔍 EDA + Preprocessing

- Visualized class imbalance and distributions
- Compared fraud vs. non-fraud `Amount` distributions
- Verified low multicollinearity using correlation heatmaps
- Performed outlier analysis before model development

---

### 3. 🤖 Model Development

#### 🔹 Unsupervised Models

- **Isolation Forest**:
  - Trained on `x_train`, predicts anomalies without labels
  - Used as baseline model

- **One-Class SVM**:
  - More sensitive to kernel/nu/gamma tuning
  - Not much better than Isolation Forest and very high processing time thus not included in UI

#### 🔸 Supervised Models with AutoML (PyCaret)

- Used `compare_models()` to shortlist best performers by **Recall**
- Tuned `xgboost` using `tune_model()`, but base model had better balance
- Final model: **XGBoost with clean export**

---

### 4. 🖥️  Ib App

Interactive app features:

- Random test sample selection
- Model predictions side-by-side
- True label shown alongside predictions
- **Metric comparison** table for both models
- **SHAP explanations** for both:
  - `TreeExplainer` for XGBoost
  - `KernelExplainer` for Isolation Forest

---

## 📊 Evaluation Metrics

I emphasized metrics critical for fraud detection:

| Metric      | Isolation Forest | XGBoost (AutoML) |
|-------------|------------------|------------------|
| Accuracy    | 0.9978           | 0.9999           |
| Precision   | 0.3723           | 0.9697           |
| Recall      | 0.5147           | 0.9411           |
| F1 Score    | 0.4321           | 0.9552           |
| MCC         | 0.4367           | 0.9553           |
| Kappa       | 0.4310           | 0.9552           |

---
## 🎯 Why These Metrics?

In fraud detection, the dataset is **highly imbalanced** — with fraudulent cases representing less than 0.2% of the total. In such scenarios, traditional metrics like **accuracy** can be misleading. For instance, predicting every transaction as "non-fraud" would yield >99% accuracy but fail to detect any actual frauds.

Instead, I emphasize metrics that better reflect model performance on minority classes:

- **Recall**  
  Measures the ability to correctly identify actual frauds (`True Positives / (True Positives + False Negatives)`).  
  🔸 High recall ensures feIr fraud cases are missed.

- **Precision**  
  Indicates how many predicted frauds Ire actually frauds (`True Positives / (True Positives + False Positives)`).  
  🔸 High precision reduces false alarms, which is crucial in real-world applications.

- **F1 Score**  
  The harmonic mean of precision and recall.  
  🔸 Balances both under- and over-prediction of fraud.

- **MCC (Matthews Correlation Coefficient)**  
  A balanced metric that considers all four outcomes of a confusion matrix.  
  🔸 Particularly robust for imbalanced data, unlike accuracy or precision alone.

- **Cohen's Kappa**  
  Measures agreement betIen predicted and actual classes, adjusted for chance.  
  🔸 Useful in evaluating how much better the model is compared to random guessing.

These metrics together provide a **holistic view** of the model's performance in the context of rare event detection.

---

## 🤖 Why I Chose These Models

I explored both **unsupervised** and **supervised** approaches:

- **Isolation Forest**:  
  Ideal for situations where **labeled data is unavailable**. It isolates anomalies without prior knowledge of fraud patterns.  
  🔹 Useful as a baseline and in real-time systems with delayed labeling.

- **One-Class SVM** *(tested offline)*:  
  Another anomaly detector that tries to capture the boundary of normal behavior.  
  🔸 HoIver, it was sensitive to hyperparameters and underperformed compared to other models.

- **XGBoost (via AutoML - PyCaret)**:  
  A poIrful, supervised ensemble model. After comparing multiple classifiers (RF, ET, LDA), XGBoost provided the **best balance** across precision, recall, and MCC.  
  🔹 It consistently achieved high performance in detecting rare frauds and was interpretable using SHAP.

I finalized **XGBoost** for its superior fraud capture rate and **Isolation Forest** for comparative benchmarking and unsupervised use cases.

## 📈 SHAP Explanations

> **SHAP (SHapley Additive Explanations)** is used to break down predictions into feature contributions.

- Each prediction shows a **waterfall graph** where:
  - Red bars push prediction toward **fraud**
  - Blue bars push it toward **non-fraud**
  - Width of bars = strength of impact
- Provides interpretability into model logic
- Used:
  - `TreeExplainer` for XGBoost
  - `KernelExplainer` for Isolation Forest

---

## 🏆 Achievements

- Trained and compared **unsupervised and supervised models**
- Achieved **94% recall and 95% Kappa** with XGBoost
- Demonstrated **interpretable ML** via SHAP in a user-friendly dashboard
- Built a **scalable, deployable Streamlit app**
- Integrated multiple evaluation metrics beyond accuracy

---

## 🛠️ How to Use This Project
Use this link to see the project live and understand the process for fraud detevtion via various ML models. (https://creditcardfrauddetection-pbmxavdkqko5shkrq42tie.streamlit.app/)

You can go through the ipynb file (Credit_fraud_methodology.ipynb) to run all the steps using this dataset from Kaggle (https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023 )

To clone : 
```bash
git clone https://github.com/Arpl14/credit_card_fraud_detection.git
cd credit_card_fraud_detection
pip install -r requirements.txt
