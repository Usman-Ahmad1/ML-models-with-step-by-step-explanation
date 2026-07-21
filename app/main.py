# Main Streamlit application for Healthcare Risk Predictor.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add the project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.logger import project_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.models.tuner import HyperparameterTuner
from src.explainability.shap_analyzer import SHAPAnalyzer
from src.explainability.rules_generator import RulesGenerator

# Import page functions
from app.pages.overview import render_overview
from app.pages.predictor import render_predictor
from app.pages.explanations import render_explanations
from app.pages.what_if import render_what_if

# Page configuration
st.set_page_config(
    page_title="Healthcare Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'shap_analyzer' not in st.session_state:
    st.session_state.shap_analyzer = None
if 'rules_generator' not in st.session_state:
    st.session_state.rules_generator = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None

# Custom CSS - Dark Mode Compatible
st.markdown("""
<style>
    /* Force crisp font rendering */
    * {
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
    }
    
    /* Main heading - dark mode compatible */
    .main-header {
        font-size: 2.5rem;
        color: #1a2634 !important;
        margin-bottom: 1rem;
        font-weight: 700 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        letter-spacing: -0.5px !important;
        line-height: 1.2 !important;
        text-shadow: none !important;
        opacity: 1 !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    /* Fix for all headings - dark mode compatible */
    h1, h2, h3, h4, h5, h6 {
        color: #1a2634 !important;
        font-weight: 600 !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
        letter-spacing: -0.3px !important;
    }
    
    /* Fix for Streamlit's default heading styles */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a2634 !important;
        font-weight: 600 !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    /* Dark mode compatibility for all text */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #1a2634 !important;
    }
    
    /* Risk colors - visible in both modes */
    .risk-high {
        color: #dc3545 !important;
        font-weight: 700 !important;
        -webkit-font-smoothing: antialiased !important;
    }
    .risk-moderate {
        color: #fd7e14 !important;
        font-weight: 700 !important;
        -webkit-font-smoothing: antialiased !important;
    }
    .risk-low {
        color: #28a745 !important;
        font-weight: 700 !important;
        -webkit-font-smoothing: antialiased !important;
    }
    
    /* Metric cards - dark mode compatible */
    .metric-card {
        background-color: #f8f9fa !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1a2634 !important;
    }
    .metric-card h1, .metric-card h2, .metric-card h3 {
        color: #1a2634 !important;
        font-weight: 600 !important;
        -webkit-font-smoothing: antialiased !important;
    }
    .metric-card p {
        color: #6c757d !important;
    }
    
    /* Recommendation cards - dark mode compatible */
    .recommendation-card {
        background-color: #e3f0fd !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        color: #1a2634 !important;
    }
    .recommendation-card p, .recommendation-card div {
        color: #1a2634 !important;
    }
    
    /* Force dark text on all content */
    .element-container, .stColumn, .stMarkdown {
        color: #1a2634 !important;
    }
    
    /* Ensure Plotly charts are visible */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    /* Fix for Streamlit's default text color */
    .st-emotion-cache-1r6slb0, .st-emotion-cache-1v3fvcr {
        color: #1a2634 !important;
    }
    
    /* Force all text to be dark */
    .stApp, .stApp p, .stApp div, .stApp span {
        color: #1a2634 !important;
    }
</style>
""", unsafe_allow_html=True)

def load_full_data():
    """Load FULL dataset (4240 samples) and train model."""
    with st.spinner("Loading full dataset and training model..."):
        try:
            # Check if data file exists
            data_path = Path("data/raw/framingham.csv")
            if not data_path.exists():
                st.error(f"Data file not found at {data_path}")
                return
            
            # Load REAL data from CSV
            loader = DataLoader(str(data_path))
            df = loader.load_and_validate()
            
            st.info(f"📊 Loaded {len(df)} samples from framingham.csv")
            
            # Preprocess
            preprocessor = DataPreprocessor()
            X_processed, y = preprocessor.preprocess(df)
            
            # Convert to DataFrame with proper column names
            if hasattr(preprocessor, 'feature_names') and preprocessor.feature_names is not None:
                feature_names = preprocessor.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
            
            # Ensure X_processed is 2D
            if len(X_processed.shape) == 1:
                X_processed = X_processed.reshape(1, -1)
            
            X_df = pd.DataFrame(X_processed, columns=feature_names)
            
            # Split data: 80% train, 20% test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_df.values, y, test_size=0.2, random_state=42, stratify=y
            )
            
            st.info(f"📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            # Train model
            trainer = ModelTrainer(use_smote=True)
            models = trainer.create_models()
            results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
            
            # Get best model
            best_model = trainer.best_model
            
            # Create SHAP analyzer
            background_sample = X_train[:100]
            shap_analyzer = SHAPAnalyzer(
                best_model.model,
                preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else feature_names,
                background_sample
            )
            shap_analyzer.create_explainer()
            
            # Generate rules
            rules_generator = RulesGenerator(
                preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else feature_names,
                shap_analyzer
            )
            rules_generator.generate_rules(X_train)
            
            # Save to session state
            st.session_state.model = best_model
            st.session_state.preprocessor = preprocessor
            st.session_state.shap_analyzer = shap_analyzer
            st.session_state.rules_generator = rules_generator
            st.session_state.feature_names = preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else feature_names
            st.session_state.data_loaded = True
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.trainer = trainer
            
            st.success(f"✅ Data loaded and model trained successfully! Total: {len(df)} samples")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

def render_performance():
    """Model performance page."""
    st.markdown('<h1 class="main-header">📈 Model Performance</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("👈 Please load the data from the sidebar first.")
        return
    
    if not hasattr(st.session_state, 'trainer') or st.session_state.trainer is None:
        st.warning("No model performance data available.")
        return
    
    results = st.session_state.trainer.results
    
    tab1, tab2, tab3 = st.tabs(["Metrics Comparison", "ROC Curves", "Confusion Matrices"])
    
    with tab1:
        st.markdown("### 📊 Model Comparison")
        
        comparison_data = []
        for name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'CV AUC': f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}",
                'Test AUC': f"{metrics['test_auc']:.3f}",
                'Test AUPRC': f"{metrics['test_auprc']:.3f}",
                'F1': f"{metrics['test_f1']:.3f}",
                'Precision': f"{metrics['test_precision']:.3f}",
                'Recall': f"{metrics['test_recall']:.3f}"
            })
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        # Bar chart
        fig = go.Figure()
        metrics_to_plot = ['Test AUC', 'Test AUPRC', 'F1']
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=df_compare['Model'],
                y=df_compare[metric],
                text=df_compare[metric],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 ROC Curves")
        
        fig = go.Figure()
        
        for name, result in results.items():
            y_test = st.session_state.y_test
            y_pred_proba = result['probabilities']
            
            from sklearn.metrics import roc_curve, roc_auc_score
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})",
                mode='lines'
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔢 Confusion Matrices")
        
        cols = st.columns(2)
        
        for idx, (name, result) in enumerate(results.items()):
            with cols[idx % 2]:
                cm = np.array(result['metrics']['confusion_matrix'])
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Low Risk', 'High Risk'],
                    y=['Low Risk', 'High Risk'],
                    title=f"{name.replace('_', ' ').title()} Confusion Matrix",
                    color_continuous_scale='Blues'
                )
                
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

# Main navigation
def main():
    """Main application function."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
        st.title("Navigation")
        
        # Page selection
        pages = {
            "📋 Overview": "overview",
            "🔮 Predictor": "predictor",
            "📊 Explanations": "explanations",
            "🧪 What-If Analysis": "what_if",
            "📈 Model Performance": "performance"
        }
        
        selection = st.radio("Go to", list(pages.keys()))
        page_key = pages[selection]
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses machine learning to predict 
        10-year cardiovascular disease risk based on the 
        Framingham Heart Study dataset.
        """)
        
        st.markdown("### 📊 Data")
        if st.button("📥 Load Full Dataset"):
            load_full_data()
    
    # Render selected page
    if page_key == "overview":
        render_overview()
    elif page_key == "predictor":
        render_predictor()
    elif page_key == "explanations":
        render_explanations()
    elif page_key == "what_if":
        render_what_if()
    elif page_key == "performance":
        render_performance()

if __name__ == "__main__":
    main()
