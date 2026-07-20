"""
Visualization utilities.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

class Visualizer:
    """Visualization utilities for EDA and model evaluation."""
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, 
                              class_names: List[str] = ['Low Risk', 'High Risk'],
                              title: str = 'Confusion Matrix',
                              figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix using matplotlib.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            title: Plot title
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        return fig
    
    @staticmethod
    def plot_roc_curves(models_data: List[Tuple[str, np.ndarray, np.ndarray]],
                       title: str = 'ROC Curves'):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_data: List of (model_name, fpr, tpr) tuples
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for name, fpr, tpr in models_data:
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=name,
                mode='lines'
            ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame,
                               title: str = 'Feature Importance',
                               n_features: int = 10):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            title: Plot title
            n_features: Number of top features to show
        
        Returns:
            Plotly figure
        """
        df_top = importance_df.head(n_features)
        
        fig = px.bar(
            df_top,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            color='importance',
            color_continuous_scale='Viridis',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=400)
        
        return fig
