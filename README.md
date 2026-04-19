# 🩺 Diabetes Risk Predictor

An end-to-end Machine Learning web app that predicts diabetes risk based on patient health parameters.

## Demo🔗 **[Live App](https://abdurhaq-diabetes-risk-predictor-app-8rk23g.streamlit.app/)**

## 🔍 Overview

- Trained and compared 4 ML models: Logistic Regression, SVM, Random Forest, XGBoost
- Best model: **Random Forest** — Accuracy: 77%, ROC-AUC: 0.83
- Explainable AI via **SHAP waterfall plots** for every prediction
- Interactive web interface built with **Streamlit**

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- SHAP, Matplotlib, Seaborn
- Streamlit

## 📊 ML Pipeline

1. Data cleaning — replaced biologically impossible zero values with median
2. Feature engineering — created BMI_Age, Glucose_Insulin_Ratio, Risk_Score
3. Train-test split with stratification
4. StandardScaler normalization
5. Model comparison by Accuracy, F1, ROC-AUC
6. SHAP explainability for model transparency

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Dataset

[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 patients, 8 features.

## 📸 Screenshot

(https://github.com/abdurhaq/diabetes-risk-predictor/blob/main/Screenshot%202026-04-18%20131818.png)
