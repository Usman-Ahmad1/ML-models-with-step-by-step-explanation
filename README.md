# 🏥 Healthcare Risk Predictor

**An explainable machine learning system for predicting 10-year cardiovascular disease (CVD) risk** using the Framingham Heart Study dataset. Built with XGBoost, SHAP, and Streamlit for production-ready clinical decision support.

**Live Demo**: [Streamlit Cloud](https://ml-models-with-step-by-step-explanation-qr4erv2fiv8mjww4rhj3cd.streamlit.app/)

## 🎯 Problem Statement

Cardiovascular disease is the leading cause of death globally. Early identification of high-risk individuals enables preventive interventions and improved patient outcomes.

**Challenges**:
- Class imbalance (only ~15% develop CVD)
- Complex non-linear risk factors
- Need for interpretability in clinical settings
- Missing data in real-world records

## Screenshots

### Overview Page
<img width="680" height="317" alt="Screenshot 2026-07-21 153021" src="https://github.com/user-attachments/assets/b52973cc-c18e-42c6-be73-6833ac9381ca" />


### Risk Predictor
<img width="577" height="255" alt="Screenshot 2026-07-21 153157" src="https://github.com/user-attachments/assets/0ffbf1a1-8b96-4e08-acab-71358ca779b7" />


### What-If Analysis
<img width="589" height="284" alt="Screenshot 2026-07-21 153606" src="https://github.com/user-attachments/assets/536eaf5b-57c6-4701-b5d9-972f72be5d01" />


## 🛠️ Solution

### Architecture
- **4 Models**: Logistic Regression (best), Random Forest, XGBoost, LightGBM
- **Explainability**: SHAP for global/local interpretations
- **Dashboard**: Streamlit with Plotly visualizations
- **Optimization**: Optuna hyperparameter tuning
- **Imbalance Handling**: SMOTE + `scale_pos_weight`

### Data Pipeline
1. Load Framingham dataset (4,240 samples, 15 features)
2. Median imputation for missing values
3. Robust scaling for outliers
4. Stratified 80/20 train-test split

## 📊 Results

| Model                | CV AUC     | Test AUC | Test AUPRC | F1 Score | Precision | Recall |
|----------------------|------------|----------|------------|----------|-----------|--------|
| Logistic Regression  | 0.737 ± 0.015 | 0.698 | 0.295 | 0.361 | 0.255 | 0.620 |
| Random Forest        | 0.939 ± 0.008 | 0.663 | 0.251 | 0.293 | 0.266 | 0.326 |
| XGBoost              | 0.955 ± 0.007 | 0.623 | 0.217 | 0.180 | 0.254 | 0.140 |
| LightGBM             | 0.950 ± 0.008 | 0.623 | 0.221 | 0.237 | 0.289 | 0.202 |

**Best Model**: Logistic Regression (Test AUC: 0.698)

**Top Risk Factors** (from SHAP analysis):
1. Age
2. Systolic BP
3. Smoking status
4. Total Cholesterol
5. BMI / Glucose

## 🚀 Features

### Dashboard Pages
- **📋 Overview**: Dataset stats, model performance, top features
- **🔮 Predictor**: Individual risk prediction with SHAP explanation
- **📊 Explanations**: Global SHAP plots & clinical rules
- **🧪 What-If**: Interactive risk factor exploration
- **📈 Performance**: ROC curves & confusion matrices

### Clinical Decision Support
- **Risk Score**: Probability (0-100%)
- **Risk Category**: Very Low / Low / Moderate / High
- **Top Contributors**: Identifies key risk factors with SHAP
- **Recommendations**: Actionable clinical guidance (e.g., "Smoking cessation strongly recommended")

## 🛠️ Tech Stack

| Category       | Tools                              |
|----------------|------------------------------------|
| ML             | XGBoost, LightGBM, Scikit-learn    |
| Explainability | SHAP                               |
| Dashboard      | Streamlit, Plotly                  |
| Optimization   | Optuna                             |
| Data           | Pandas, NumPy                      |
| Imbalance      | SMOTE (imbalanced-learn)           |

## 📁 Project Structure

```bash
X-GBoost-portfolio/
├── app/                    # Streamlit dashboard
│   ├── main.py
│   └── pages/
├── src/
│   ├── data/
│   ├── models/
│   ├── explainability/
│   └── utils/
├── data/raw/               # Framingham dataset
├── models/                 # Saved .pkl models
├── assets/                 # Screenshots
├── notebooks/              # EDA & experiments
├── requirements.txt
└── README.md
