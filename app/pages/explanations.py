"""
Explanations page for the Streamlit app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

def render_explanations():
    """Render the explanations page with SHAP visualizations."""
    st.markdown('<h1 class="main-header">📊 Global Explanations</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("👈 Please load the data from the sidebar first.")
        return
    
    if st.session_state.shap_analyzer is None:
        st.warning("SHAP analyzer not available. Please train a model first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Summary", "Clinical Rules"])
    
    with tab1:
        st.markdown("### 🔑 Global Feature Importance")
        st.markdown("These are the most important features for predicting cardiovascular risk:")
        
        try:
            importance_df = st.session_state.shap_analyzer.global_importance()
            
            if importance_df is not None and not importance_df.empty:
                fig = px.bar(
                    importance_df,
                    x='mean_shap',
                    y='feature',
                    orientation='h',
                    title='Feature Importance (SHAP)',
                    labels={'mean_shap': 'Mean SHAP Value', 'feature': 'Feature'},
                    color='mean_shap',
                    color_continuous_scale='Viridis',
                    height=500
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Show full table
                st.markdown("#### Full Feature Importance Table")
                st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("No importance data available")
        except Exception as e:
            st.warning(f"Could not generate feature importance: {str(e)}")
    
    with tab2:
        st.markdown("### 📈 SHAP Summary Plot")
        st.markdown("The beeswarm plot shows feature impact across all patients:")
        
        try:
            X_sample = st.session_state.X_train[:100] if st.session_state.X_train is not None else None
            if X_sample is not None:
                fig = st.session_state.shap_analyzer.plot_summary(X_sample)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("SHAP summary plot not available for this model type")
            else:
                st.info("No training data available for SHAP summary")
        except Exception as e:
            st.warning(f"Could not generate SHAP summary: {str(e)}")
        
        st.markdown("#### SHAP Bar Plot")
        try:
            X_sample = st.session_state.X_train[:100] if st.session_state.X_train is not None else None
            if X_sample is not None:
                fig = st.session_state.shap_analyzer.plot_bar(X_sample)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("SHAP bar plot not available for this model type")
        except Exception as e:
            st.warning(f"Could not generate SHAP bar plot: {str(e)}")
    
    with tab3:
        st.markdown("### 📋 Clinical Decision Rules")
        st.markdown("Evidence-based rules derived from SHAP analysis:")
        
        if st.session_state.rules_generator:
            rules = st.session_state.rules_generator.rules
            
            if rules:
                for rule in rules[:15]:
                    with st.expander(f"📌 {rule.get('recommendation', 'Rule')}"):
                        st.markdown(f"**Condition:** {rule.get('condition', 'N/A')}")
                        st.markdown(f"**Clinical Evidence:** {rule.get('clinical_evidence', 'N/A')}")
                        
                        # Show feature if available
                        if 'feature' in rule:
                            st.markdown(f"**Feature:** {rule['feature']}")
            else:
                st.info("No rules generated. Please train a model first.")
        else:
            st.info("Rules generator not available.")
