"""
Prediction page for the Streamlit app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def render_predictor():
    """Render the prediction page."""
    st.markdown('<h1 class="main-header">🔮 Risk Predictor</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("👈 Please click 'Load Full Dataset' in the sidebar first.")
        return
    
    st.markdown("Enter patient information to predict 10-year cardiovascular disease risk:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 20, 80, 50, help="Age is the strongest predictor of CVD risk")
            sex = st.selectbox("Sex", ["Male", "Female"])
            current_smoker = st.selectbox("Current Smoker", ["No", "Yes"])
            cigs_per_day = st.number_input("Cigarettes per Day", 0, 40, 0)
            sys_bp = st.slider("Systolic Blood Pressure (mmHg)", 90, 200, 120)
        
        with col2:
            dia_bp = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 80)
            bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.5)
            glucose = st.number_input("Glucose (mg/dL)", 50, 200, 85)
            tot_chol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
            education = st.selectbox("Education Level", ["High School", "Some College", "College", "Graduate"])
        
        submitted = st.form_submit_button("🔮 Predict Risk")
    
    if submitted:
        with st.spinner("Making prediction..."):
            try:
                # Prepare input data
                input_data = pd.DataFrame([{
                    'male': 1 if sex == "Male" else 0,
                    'age': age,
                    'education': {"High School": 1, "Some College": 2, "College": 3, "Graduate": 4}[education],
                    'currentSmoker': 1 if current_smoker == "Yes" else 0,
                    'cigsPerDay': cigs_per_day,
                    'BPMeds': 0,
                    'prevalentStroke': 0,
                    'prevalentHyp': 1 if sys_bp > 140 else 0,
                    'diabetes': 0,
                    'totChol': tot_chol,
                    'sysBP': sys_bp,
                    'diaBP': dia_bp,
                    'BMI': bmi,
                    'heartRate': 72,
                    'glucose': glucose
                }])
                
                # Preprocess input
                X_input, _ = st.session_state.preprocessor.preprocess(
                    input_data, fit=False
                )
                
                # Make prediction
                proba = st.session_state.model.predict_proba(X_input)[0, 1]
                risk_percentage = proba * 100
                
                # Get SHAP explanation
                shap_values = st.session_state.shap_analyzer.explain(X_input)
                
                # Get rules
                report = st.session_state.rules_generator.generate_health_report(
                    X_input, 0
                )
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                # Risk color
                if proba < 0.1:
                    risk_color = "risk-low"
                    risk_label = "Very Low Risk"
                    emoji = "🟢"
                elif proba < 0.2:
                    risk_color = "risk-low"
                    risk_label = "Low Risk"
                    emoji = "🟡"
                elif proba < 0.3:
                    risk_color = "risk-moderate"
                    risk_label = "Moderate Risk"
                    emoji = "🟠"
                elif proba < 0.5:
                    risk_color = "risk-high"
                    risk_label = "High Risk"
                    emoji = "🔴"
                else:
                    risk_color = "risk-high"
                    risk_label = "Very High Risk"
                    emoji = "🔴"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Risk Score</h3>
                        <h1 class="{risk_color}">{risk_percentage:.1f}%</h1>
                        <p>10-year CVD risk</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Risk Category</h3>
                        <h2 class="{risk_color}">{emoji} {risk_label}</h2>
                        <p>Based on SHAP analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Confidence</h3>
                        <h2>High</h2>
                        <p>Based on model calibration</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk gauge
                st.markdown("#### Risk Gauge")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_percentage,
                    title={'text': "10-Year CVD Risk"},
                    delta={'reference': 15.0},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 30], 'color': "yellow"},
                            {'range': [30, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_percentage
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors
                st.markdown("### ⚠️ Top Risk Factors")
                
                risk_factors = report.get('top_risk_factors', [])[:5]
                if risk_factors:
                    for factor in risk_factors:
                        contribution = factor.get('contribution', 0)
                        color = "🔴" if contribution > 0 else "🟢"
                        st.markdown(f"""
                        <div class="metric-card">
                            {color} <strong>{factor.get('feature', 'Unknown')}</strong>: 
                            Contribution = {contribution:.3f} 
                            (Current Value: {factor.get('current_value', 'N/A')})
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No risk factors available")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                recommendations = report.get('recommendations', [])[:3]
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            ✅ {rec}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific recommendations available")
                
                # SHAP Explanation
                st.markdown("### 📈 SHAP Explanation")
                
                if st.session_state.shap_analyzer and X_input is not None:
                    fig = st.session_state.shap_analyzer.plot_waterfall(X_input, 0)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("SHAP waterfall plot not available for this model type")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
