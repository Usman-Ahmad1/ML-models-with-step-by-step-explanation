"""
Reusable UI components for Streamlit app.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any

class UIComponents:
    """Reusable Streamlit UI components."""
    
    @staticmethod
    def render_metric_card(title: str, value: str, 
                           subtitle: str = "", 
                           icon: str = "📊",
                           color: str = "blue"):
        """
        Render a metric card.
        
        Args:
            title: Card title
            value: Main value
            subtitle: Subtitle text
            icon: Emoji icon
            color: Color theme
        """
        colors = {
            'blue': '#007bff',
            'green': '#28a745',
            'red': '#dc3545',
            'yellow': '#ffc107',
            'purple': '#6f42c1'
        }
        
        color_hex = colors.get(color, '#007bff')
        
        st.markdown(f"""
        <div style="
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid {color_hex};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        ">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">{icon}</span>
                <div>
                    <div style="font-size: 14px; color: #6c757d;">{title}</div>
                    <div style="font-size: 24px; font-weight: bold;">{value}</div>
                    {f'<div style="font-size: 12px; color: #6c757d;">{subtitle}</div>' if subtitle else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_risk_gauge(risk_score: float):
        """
        Render a risk gauge chart.
        
        Args:
            risk_score: Risk probability (0-1)
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            gauge = {
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
                    'value': risk_score * 100
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
