import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator
from src.models.base import BaseModel
from src.logger import project_logger

class LightGBMModel(BaseModel):
    """
    LightGBM classifier with advanced features.
    """
    
    def __init__(self, name: str = "LightGBM"):
        super().__init__(name)
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
            'class_weight': 'balanced'
        }
        self.best_iteration = None
        
    def build(self, **kwargs) -> BaseEstimator:
        params = {**self.default_params, **kwargs}
        self.model = lgb.LGBMClassifier(**params)
        self.logger.info(f"Built LightGBM model with {len(params)} parameters")
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if self.model is None:
            self.build()
        
        self.logger.info(f"Training LightGBM model with {len(X_train)} samples...")
        
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            self.logger.info(f"Using {len(X_val)} validation samples for early stopping")
        
        if eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='auc',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        if hasattr(self.model, 'best_iteration_'):
            self.best_iteration = self.model.best_iteration_
            self.logger.info(f"Best iteration: {self.best_iteration}")
        
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
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        importance = self.model.feature_importances_
        return dict(zip(self.model.feature_names_in_, importance))
