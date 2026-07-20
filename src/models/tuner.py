"""
Hyperparameter tuning using Optuna.
"""
import optuna
import numpy as np
from typing import Dict, Any, Optional, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.logger import project_logger
from src.config import config
from src.models.xgboost_model import XGBoostModel

class HyperparameterTuner:
    """
    Hyperparameter optimization using Optuna.
    """
    
    def __init__(self, model_type: str = 'xgboost', n_trials: int = 50):
        """
        Initialize HyperparameterTuner.
        
        Args:
            model_type: Type of model to tune
            n_trials: Number of optimization trials
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.logger = project_logger
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def objective_xgboost(self, trial, X_train, y_train, cv_folds: int = 5):
        """
        Optuna objective function for XGBoost.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of CV folds
        
        Returns:
            CV score
        """
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
        
        # Create and evaluate model
        model = XGBoostModel()
        model.build(**params)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            model.model, X_train, y_train,
            cv=skf, scoring='roc_auc', n_jobs=-1
        )
        
        return scores.mean()
    
    def tune_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                    cv_folds: int = 5) -> Dict[str, Any]:
        """
        Tune XGBoost hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of CV folds
        
        Returns:
            Best parameters and score
        """
        self.logger.info(f"Starting XGBoost hyperparameter tuning with {self.n_trials} trials...")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            study_name='xgboost_tuning',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective_xgboost(trial, X_train, y_train, cv_folds),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV AUC: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study
        }
    
    def tune_models(self, X_train: np.ndarray, y_train: np.ndarray,
                   model_types: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """
        Tune multiple models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_types: List of model types to tune
        
        Returns:
            Dictionary of tuning results
        """
        results = {}
        
        if model_types is None:
            model_types = ['xgboost']
        
        for model_type in model_types:
            if model_type == 'xgboost':
                results['xgboost'] = self.tune_xgboost(X_train, y_train)
            # Add other model types as needed
        
        return results