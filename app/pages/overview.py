"""
Overview page for the Streamlit app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_overview():
    """Render the overview page with dataset statistics and model performance."""
    st.markdown('<h1 class="main-header">📋 Overview</h1>', unsafe_allow_html=True)
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.info("👈 Please click 'Load Full Dataset' in the sidebar to load data.")
        return
    
    # Dataset Overview Section
    st.markdown("### 📊 Dataset Overview")
    
    # Get data from session state
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    df = st.session_state.get('df', None)
    
    # Calculate metrics
    total_samples = len(X_train) + len(X_test) if X_train is not None else 0
    train_samples = len(X_train) if X_train is not None else 0
    test_samples = len(X_test) if X_test is not None else 0
    
    # Fix: Check if feature_names exists and is not None
    if st.session_state.feature_names is not None:
        n_features = len(st.session_state.feature_names)
    else:
        n_features = 0
    
    # Calculate CHD rate
    if y_train is not None:
        chd_rate = (y_train.sum() / len(y_train)) * 100
    else:
        chd_rate = 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Samples",
            f"{total_samples:,}",
            "full dataset"
        )
    
    with col2:
        st.metric(
            "📈 Train/Test Split",
            f"{train_samples:,} / {test_samples:,}",
            f"{(train_samples/total_samples*100):.0f}% / {(test_samples/total_samples*100):.0f}%"
        )
    
    with col3:
        st.metric(
            "⚠️ CHD Rate",
            f"{chd_rate:.1f}%",
            "in training set"
        )
    
    with col4:
        st.metric(
            "🔄 Features",
            n_features,
            "processed"
        )
    
    # Show dataset preview
    if df is not None:
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show data statistics
        st.markdown("### 📊 Data Statistics")
        
        # Show column info
        col1, col2 = st.columns(2)
        
        with col1:
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("**Numeric Columns Summary:**")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        with col2:
            # Categorical columns summary
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                st.markdown("**Categorical Columns Summary:**")
                for col in cat_cols:
                    st.write(f"- **{col}**: {df[col].nunique()} unique values")
                    st.write(f"  Top: {df[col].value_counts().head(3).to_dict()}")
    
    # Model Performance Summary
    st.markdown("### 📈 Model Performance Summary")
    
    if hasattr(st.session_state, 'trainer') and st.session_state.trainer is not None:
        results = st.session_state.trainer.results
        
        if results:
            # Create performance dataframe
            performance_data = []
            for name, result in results.items():
                metrics = result['metrics']
                performance_data.append({
                    'Model': name.replace('_', ' ').title(),
                    'CV AUC': f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}",
                    'Test AUC': f"{metrics['test_auc']:.3f}",
                    'Test AUPRC': f"{metrics['test_auprc']:.3f}",
                    'F1 Score': f"{metrics['test_f1']:.3f}",
                    'Precision': f"{metrics['test_precision']:.3f}",
                    'Recall': f"{metrics['test_recall']:.3f}"
                })
            
            df_perf = pd.DataFrame(performance_data)
            st.dataframe(df_perf, use_container_width=True)
            
            # Best model highlight
            best_model_name = st.session_state.trainer.best_model_name
            if best_model_name in st.session_state.trainer.results:
                best_model_auc = st.session_state.trainer.results[best_model_name]['metrics']['test_auc']
                st.success(f"🏆 **Best Model: {best_model_name.replace('_', ' ').title()}** with Test AUC: {best_model_auc:.3f}")
            else:
                st.success(f"🏆 **Best Model: {best_model_name.replace('_', ' ').title()}**")
            
            # Performance chart
            st.markdown("#### Performance Comparison")
            
            fig = go.Figure()
            
            # Get metrics for plotting
            plot_data = []
            for name, result in results.items():
                plot_data.append({
                    'Model': name.replace('_', ' ').title(),
                    'Test AUC': result['metrics']['test_auc'],
                    'Test AUPRC': result['metrics']['test_auprc'],
                    'F1 Score': result['metrics']['test_f1']
                })
            
            df_plot = pd.DataFrame(plot_data)
            
            # Melt for bar chart
            df_melted = df_plot.melt(id_vars=['Model'], 
                                     value_vars=['Test AUC', 'Test AUPRC', 'F1 Score'],
                                     var_name='Metric', value_name='Score')
            
            fig = px.bar(df_melted, 
                        x='Model', 
                        y='Score', 
                        color='Metric',
                        barmode='group',
                        title='Model Performance Comparison',
                        text='Score',
                        height=400)
            
            fig.update_traces(texttemplate='%{text:.3f}', textposition='auto')
            fig.update_layout(yaxis_range=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("No model performance data available. Please train a model first.")
    
    # Feature Importance
    st.markdown("### 🔑 Top Features")
    
    if st.session_state.shap_analyzer is not None:
        try:
            importance_df = st.session_state.shap_analyzer.global_importance()
            
            if importance_df is not None and not importance_df.empty:
                # Top 10 features
                fig = px.bar(
                    importance_df.head(10),
                    x='mean_shap',
                    y='feature',
                    orientation='h',
                    title='Top 10 Features by SHAP Importance',
                    labels={'mean_shap': 'Mean SHAP Value', 'feature': 'Feature'},
                    color='mean_shap',
                    color_continuous_scale='Viridis',
                    height=400
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Show feature values
                st.markdown("**Feature Importance Values:**")
                st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("No SHAP importance data available")
        except Exception as e:
            st.warning(f"Could not generate SHAP importance: {str(e)}")
    else:
        st.info("SHAP analyzer not available. Please train a model first.")
    
    # Dataset Distribution
    if y_train is not None:
        st.markdown("### 📊 Target Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            train_counts = pd.Series(y_train).value_counts()
            fig = px.pie(
                values=train_counts.values,
                names=['No CHD', 'CHD'],
                title='Training Set Distribution',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Test set distribution
            if y_test is not None:
                test_counts = pd.Series(y_test).value_counts()
                fig = px.pie(
                    values=test_counts.values,
                    names=['No CHD', 'CHD'],
                    title='Test Set Distribution',
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
