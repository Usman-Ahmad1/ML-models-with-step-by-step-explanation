import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                            precision_score, recall_score, confusion_matrix,
                            classification_report)
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from src.logger import project_logger
from src.exceptions import ModelTrainingError
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.lightgbm_model import LightGBMModel

class ModelTrainer:
    """
    Handles training, evaluation, and comparison of multiple models.
    """
    
    def __init__(self, random_state: int = 42, use_smote: bool = True):
        self.random_state = random_state
        self.use_smote = use_smote
        self.logger = project_logger
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_models(self) -> Dict[str, Any]:
        self.logger.info("Creating models for comparison...")
        
        models = {
            'logistic_regression': LogisticRegressionModel(),
            'random_forest': RandomForestModel(),
            'xgboost': XGBoostModel(),
            'lightgbm': LightGBMModel()
        }
        
        for name, model in models.items():
            model.build()
            self.logger.info(f"Created {name} model")
        
        self.models = models
        return models
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          n_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        self.logger.info("Starting model training and evaluation...")
        
        # Apply SMOTE to balance the training data
        if self.use_smote:
            self.logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            self.logger.info(f"Balanced training data: {len(X_train_balanced)} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        results = {}
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model.model, X_train_balanced, y_train_balanced,
                    cv=skf, scoring='roc_auc', n_jobs=-1
                )
                
                # Train on full balanced training set
                model.train(X_train_balanced, y_train_balanced)
                
                # Test predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std(),
                    'test_auc': roc_auc_score(y_test, y_pred_proba),
                    'test_auprc': average_precision_score(y_test, y_pred_proba),
                    'test_f1': f1_score(y_test, y_pred),
                    'test_precision': precision_score(y_test, y_pred),
                    'test_recall': recall_score(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                self.logger.info(f"{name} - CV AUC: {metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.4f})")
                self.logger.info(f"{name} - Test AUC: {metrics['test_auc']:.4f}, AUPRC: {metrics['test_auprc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.results = results
        
        # Select best model based on test AUC
        if results:
            best_name = max(results.keys(), key=lambda x: results[x]['metrics']['test_auc'])
            self.best_model = results[best_name]['model']
            self.best_model_name = best_name
            self.logger.info(f"Best model: {best_name} with AUC: {results[best_name]['metrics']['test_auc']:.4f}")
        
        return results
    
    def save_model(self, model: Any, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(model, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise ModelTrainingError(f"Failed to save model: {str(e)}")
    
    def load_model(self, path: str) -> Any:
        try:
            model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelTrainingError(f"Failed to load model: {str(e)}")
