import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator
from src.models.base import BaseModel
from src.logger import project_logger

class XGBoostModel(BaseModel):
    """
    XGBoost classifier with advanced features.
    """
    
    def __init__(self, name: str = "XGBoost"):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
        """
        super().__init__(name)
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1,
            'eval_metric': 'logloss'  # Moved to constructor
        }
        self.best_iteration = None
        
    def build(self, **kwargs) -> BaseEstimator:
        """
        Build XGBoost classifier.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            XGBClassifier instance
        """
        params = {**self.default_params, **kwargs}
        
        # Handle scale_pos_weight for imbalance
        if 'scale_pos_weight' not in kwargs:
            params['scale_pos_weight'] = 1
        
        self.model = xgb.XGBClassifier(**params)
        self.logger.info(f"Built XGBoost model with {len(params)} parameters")
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the XGBoost model with early stopping.
        
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
        
        self.logger.info(f"Training XGBoost model with {len(X_train)} samples...")
        
        # Prepare early stopping
        eval_set = []
        
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            self.logger.info(f"Using {len(X_val)} validation samples for early stopping")
        
        # Fit the model - eval_metric is already in the model constructor
        if eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        # Store best iteration
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
            self.logger.info(f"Best iteration: {self.best_iteration}")
        
        # Calculate training metrics
        metrics = {
            'train_score': self.model.score(X_train, y_train),
            'best_iteration': self.best_iteration,
            'n_features': X_train.shape[1]
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = self.model.feature_importances_
        return dict(zip(self.model.feature_names_in_, importance))
