"""
Metrics calculation utilities.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple

class MetricsCalculator:
    """Calculate and aggregate model metrics."""
    
    @staticmethod
    def calculate_binary_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive binary classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auprc': average_precision_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> str:
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred)
