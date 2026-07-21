# 🏥 Healthcare Risk Predictor

**An explainable machine learning system for predicting 10-year cardiovascular disease (CVD) risk** using the Framingham Heart Study dataset. Built with XGBoost, SHAP, and Streamlit for production-ready clinical decision support.

**Live Demo**: [Streamlit Cloud]((https://ml-models-with-step-by-step-explanation-qr4erv2fiv8mjww4rhj3cd.streamlit.app/)p) *(replace with actual link)*

## 🎯 Problem Statement

Cardiovascular disease is the leading cause of death globally. Early identification of high-risk individuals enables preventive interventions and improved patient outcomes.

**Challenges**:
- Class imbalance (only ~15% develop CVD)
- Complex non-linear risk factors
- Need for interpretability in clinical settings
- Missing data in real-world records

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

| Model                | CV AUC | Test AUC | Status    |
|----------------------|--------|----------|-----------|
| Logistic Regression  | 0.731  | 0.701    | ✅ Best   |
| Random Forest        | 0.711  | 0.666    | ✅        |
| XGBoost              | 0.695  | 0.633    | ✅        |
| LightGBM             | 0.689  | 0.636    | ✅        |

**Top 5 Risk Factors (SHAP)**

1. Age (0.402)
2. Current Smoker (0.211)
3. Age × Smoking (0.192)
4. Cigarettes/Day (0.138)
5. Systolic BP (0.120)

## 🚀 Features

### Dashboard Pages

| Page          | Function                                      |
|---------------|-----------------------------------------------|
| 📋 Overview   | Dataset stats, model performance, top features |
| 🔮 Predictor  | Individual risk prediction with SHAP explanation |
| 📊 Explanations | Global SHAP plots & clinical rules           |
| 🧪 What-If    | Interactive risk factor exploration           |
| 📈 Performance| ROC curves & confusion matrices               |

### Clinical Decision Support
- **Risk Score**: Probability (0-100%)
- **Risk Category**: Very Low / Low / Moderate / High
- **Top Contributors**: Identifies key risk factors
- **Recommendations**: Actionable clinical guidance

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
│   ├── main.py            # Main app with navigation
│   └── pages/             # Individual pages
├── src/
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Model definitions
│   ├── explainability/    # SHAP analysis & rules
│   └── utils/             # Metrics & visualization
├── data/raw/              # Dataset
├── models/                # Saved models (.pkl)
├── notebooks/             # EDA and modeling notebooks
├── requirements.txt       # Dependencies
├── README.md
└── .gitignore
