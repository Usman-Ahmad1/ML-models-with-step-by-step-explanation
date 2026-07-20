"""
Base model class and interfaces.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from src.logger import project_logger

class BaseModel(ABC):
    """
    Abstract base class for all models.
    """
    
    def __init__(self, name: str):
        """
        Initialize base model.
        
        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.logger = project_logger
        
    @abstractmethod
    def build(self, **kwargs) -> BaseEstimator:
        """
        Build the model with given parameters.
        
        Args:
            **kwargs: Model parameters
        
        Returns:
            Built model instance
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
        
        Returns:
            Prediction probabilities
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        if self.model and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        
        Returns:
            Self
        """
        if self.model and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self