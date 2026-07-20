"""
Random Forest model implementation.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator
from src.models.base import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest classifier.
    """
    
    def __init__(self, name: str = "RandomForest"):
        """
        Initialize Random Forest model.
        
        Args:
            name: Model name
        """
        super().__init__(name)
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
    def build(self, **kwargs) -> BaseEstimator:
        """
        Build Random Forest classifier.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            RandomForestClassifier instance
        """
        params = {**self.default_params, **kwargs}
        self.model = RandomForestClassifier(**params)
        self.logger.info(f"Built Random Forest model with {len(params)} parameters")
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build()
        
        self.logger.info(f"Training Random Forest model with {len(X_train)} samples...")
        
        self.model.fit(X_train, y_train)
        
        metrics = {
            'train_score': self.model.score(X_train, y_train)
        }
        
        if X_val is not None and y_val is not None:
            metrics['val_score'] = self.model.score(X_val, y_val)
        
        self.logger.info(f"Training complete. Metrics: {metrics}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
        
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)