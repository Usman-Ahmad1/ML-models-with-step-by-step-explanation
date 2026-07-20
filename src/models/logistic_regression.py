import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator
from src.models.base import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier.
    """
    
    def __init__(self, name: str = "LogisticRegression"):
        super().__init__(name)
        self.default_params = {
            'C': 1.0,
            'penalty': None,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'l1_ratio': None
        }
        self.scaler = None
        
    def build(self, **kwargs) -> BaseEstimator:
        params = {**self.default_params, **kwargs}
        self.model = LogisticRegression(**params)
        self.scaler = StandardScaler()
        self.logger.info(f"Built Logistic Regression model with {len(params)} parameters")
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if self.model is None:
            self.build()
        
        self.logger.info(f"Training Logistic Regression model with {len(X_train)} samples...")
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        
        metrics = {
            'train_score': self.model.score(X_train_scaled, y_train)
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            metrics['val_score'] = self.model.score(X_val_scaled, y_val)
        
        self.logger.info(f"Training complete. Metrics: {metrics}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
