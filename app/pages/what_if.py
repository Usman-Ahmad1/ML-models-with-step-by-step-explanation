"""
What-if analysis page for the Streamlit app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def render_what_if():
    """Render the what-if analysis page."""
    st.markdown('<h1 class="main-header">🧪 What-If Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("👈 Please load the data from the sidebar first.")
        return
    
    st.markdown("""
    ### 🔄 Explore How Changes in Risk Factors Affect Your Risk Score
    
    Adjust the sliders below to see how modifying risk factors impacts your predicted cardiovascular risk.
    The SHAP analysis will show which factors have the biggest impact.
    """)
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Risk Factors")
        
        # Get default values from training data if available
        if st.session_state.X_train is not None and len(st.session_state.X_train) > 0:
            mean_values = np.mean(st.session_state.X_train, axis=0)
            feature_names = st.session_state.feature_names
            defaults = {}
            for i, name in enumerate(feature_names):
                clean_name = name.replace('num__', '').replace('cat__', '')
                defaults[clean_name] = float(mean_values[i]) if i < len(mean_values) else 0
        else:
            defaults = {
                'age': 50, 'male': 0, 'education': 2, 'currentSmoker': 0,
                'cigsPerDay': 0, 'BPMeds': 0, 'prevalentStroke': 0,
                'prevalentHyp': 0, 'diabetes': 0, 'totChol': 200,
                'sysBP': 120, 'diaBP': 80, 'BMI': 25, 'heartRate': 72,
                'glucose': 85
            }
        
        # Create sliders for each risk factor with proper ranges
        cols = st.columns(2)
        
        with cols[0]:
            age = st.slider(
                "Age",
                min_value=20, max_value=80, value=int(defaults.get('age', 50)),
                key="whatif_age",
                help="Age is the strongest predictor of CVD risk"
            )
            
            sex = st.selectbox(
                "Sex",
                options=["Female", "Male"],
                index=1 if defaults.get('male', 0) == 1 else 0,
                key="whatif_sex"
            )
            sex_value = 1 if sex == "Male" else 0
            
            current_smoker = st.selectbox(
                "Current Smoker",
                options=["No", "Yes"],
                index=1 if defaults.get('currentSmoker', 0) == 1 else 0,
                key="whatif_smoker"
            )
            smoker_value = 1 if current_smoker == "Yes" else 0
            
            cigs_per_day = st.slider(
                "Cigarettes per Day",
                min_value=0, max_value=40, value=int(defaults.get('cigsPerDay', 0)),
                key="whatif_cigs"
            )
            
            sys_bp = st.slider(
                "Systolic BP (mmHg)",
                min_value=90, max_value=200, value=int(defaults.get('sysBP', 120)),
                key="whatif_sysbp",
                help="High systolic BP is a major CVD risk factor"
            )
            
            dia_bp = st.slider(
                "Diastolic BP (mmHg)",
                min_value=60, max_value=120, value=int(defaults.get('diaBP', 80)),
                key="whatif_diabp"
            )
        
        with cols[1]:
            bmi = st.slider(
                "BMI",
                min_value=15, max_value=45, value=int(defaults.get('BMI', 25)),
                step=1,
                key="whatif_bmi",
                help="BMI > 30 indicates obesity, increasing CVD risk"
            )
            
            glucose = st.slider(
                "Glucose (mg/dL)",
                min_value=50, max_value=200, value=int(defaults.get('glucose', 85)),
                step=5,
                key="whatif_glucose"
            )
            
            tot_chol = st.slider(
                "Total Cholesterol (mg/dL)",
                min_value=100, max_value=400, value=int(defaults.get('totChol', 200)),
                step=10,
                key="whatif_chol",
                help="High cholesterol > 240 increases CVD risk"
            )
            
            # Education - Fix the index calculation
            education_options = ["High School", "Some College", "College", "Graduate"]
            education_default = defaults.get('education', 2)
            
            # Ensure index is within valid range (0 to len(options)-1)
            education_index = min(max(int(education_default) - 1, 0), len(education_options) - 1)
            
            education = st.selectbox(
                "Education Level",
                options=education_options,
                index=education_index,
                key="whatif_education"
            )
            education_value = {"High School": 1, "Some College": 2, "College": 3, "Graduate": 4}[education]
            
            # Prevalent Hypertension
            prevalent_hyp = st.selectbox(
                "Prevalent Hypertension",
                options=["No", "Yes"],
                index=1 if defaults.get('prevalentHyp', 0) == 1 else 0,
                key="whatif_hyp"
            )
            hyp_value = 1 if prevalent_hyp == "Yes" else 0
        
        # Add a reset button
        if st.button("🔄 Reset to Average Values"):
            st.rerun()
    
    # Prepare input data for prediction
    input_data = pd.DataFrame([{
        'male': sex_value,
        'age': age,
        'education': education_value,
        'currentSmoker': smoker_value,
        'cigsPerDay': cigs_per_day,
        'BPMeds': 0,
        'prevalentStroke': 0,
        'prevalentHyp': hyp_value,
        'diabetes': 0,
        'totChol': tot_chol,
        'sysBP': sys_bp,
        'diaBP': dia_bp,
        'BMI': float(bmi),
        'heartRate': 72,
        'glucose': float(glucose)
    }])
    
    try:
        # Preprocess input
        X_input, _ = st.session_state.preprocessor.preprocess(
            input_data, fit=False
        )
        
        # Make prediction
        proba = st.session_state.model.predict_proba(X_input)[0, 1]
        risk_percentage = proba * 100
        
        # Get SHAP explanation
        shap_values = st.session_state.shap_analyzer.explain(X_input)
        
        # Get feature contributions
        if shap_values is not None:
            shap_single = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            feature_contributions = {}
            for i, name in enumerate(st.session_state.feature_names):
                if i < len(shap_single):
                    clean_name = name.replace('num__', '').replace('cat__', '')
                    feature_contributions[clean_name] = float(shap_single[i])
        
        with col2:
            st.markdown("### 📈 Risk Analysis")
            
            # Risk gauge
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
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk category
            if proba < 0.1:
                risk_text = "🟢 Very Low Risk"
                risk_color = "green"
            elif proba < 0.2:
                risk_text = "🟡 Low Risk"
                risk_color = "orange"
            elif proba < 0.3:
                risk_text = "🟠 Moderate Risk"
                risk_color = "darkorange"
            elif proba < 0.5:
                risk_text = "🔴 High Risk"
                risk_color = "red"
            else:
                risk_text = "🔴 Very High Risk"
                risk_color = "darkred"
            
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 10px 0;
            ">
                <h2 style="color: {risk_color};">{risk_text}</h2>
                <p>Risk Score: <strong>{risk_percentage:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature contribution bar chart
            st.markdown("### 🔍 Feature Contributions")
            
            if shap_values is not None:
                # Get top contributing features
                sorted_features = sorted(
                    feature_contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
                
                df_contrib = pd.DataFrame(
                    sorted_features,
                    columns=['Feature', 'SHAP Value']
                )
                
                fig = px.bar(
                    df_contrib,
                    x='SHAP Value',
                    y='Feature',
                    orientation='h',
                    title='SHAP Feature Contributions',
                    color='SHAP Value',
                    color_continuous_scale='RdBu_r',
                    labels={'SHAP Value': 'Impact on Risk'},
                    height=400
                )
                
                fig.update_layout(
                    xaxis_title="Contribution to Risk",
                    yaxis_title="Feature",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show what-if analysis insights
            st.markdown("### 💡 Insights")
            
            # Identify the biggest risk factors
            if shap_values is not None:
                top_risk = sorted(
                    feature_contributions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                st.markdown("**Top Risk Contributors:**")
                for feature, value in top_risk:
                    if value > 0.1:
                        st.markdown(f"- ⚠️ **{feature}**: +{value:.2f} contribution to risk")
                
                # Recommendations based on modifiable factors
                st.markdown("**Recommended Actions:**")
                recommendations = []
                
                if smoker_value == 1:
                    recommendations.append("🚭 **Quit Smoking** - Reduces risk significantly")
                
                if sys_bp > 140:
                    recommendations.append("💊 **Manage Blood Pressure** - Target < 140/90 mmHg")
                
                if bmi > 30:
                    recommendations.append("🏃 **Weight Management** - Target BMI < 25")
                
                if tot_chol > 240:
                    recommendations.append("🥗 **Dietary Changes** - Lower cholesterol intake")
                
                if glucose > 100:
                    recommendations.append("🍬 **Monitor Glucose** - Check for prediabetes")
                
                if not recommendations:
                    recommendations.append("✅ Your risk factors are well managed! Continue healthy lifestyle.")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
    
    except Exception as e:
        st.error(f"Error in what-if analysis: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
