"""
Advanced feature engineering module.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import project_logger
from src.config import config

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer.
    """
    
    def __init__(self, create_interactions: bool = True, create_bins: bool = True):
        """
        Initialize FeatureEngineer.
        
        Args:
            create_interactions: Create interaction features
            create_bins: Create binning features
        """
        self.create_interactions = create_interactions
        self.create_bins = create_bins
        self.feature_names_ = []
        self.logger = project_logger
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        self.feature_names_ = list(X.columns)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data with engineered features.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Transformed DataFrame with engineered features
        """
        X_transformed = X.copy()
        
        if self.create_interactions:
            X_transformed = self._create_interactions(X_transformed)
        
        if self.create_bins:
            X_transformed = self._create_bins(X_transformed)
        
        return X_transformed
    
    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features.
        
        Args:
            X: Input DataFrame
        
        Returns:
            DataFrame with interaction features
        """
        # Age-related interactions
        if 'age' in X.columns:
            if 'currentSmoker' in X.columns:
                X['age_smoker'] = X['age'] * X['currentSmoker']
                self.logger.debug("Created age_smoker interaction")
            
            if 'diabetes' in X.columns:
                X['age_diabetes'] = X['age'] * X['diabetes']
                self.logger.debug("Created age_diabetes interaction")
        
        # Blood pressure interactions
        if 'sysBP' in X.columns and 'diaBP' in X.columns:
            X['bp_ratio'] = X['sysBP'] / (X['diaBP'] + 1)
            X['mean_bp'] = (2 * X['diaBP'] + X['sysBP']) / 3
            self.logger.debug("Created bp_ratio and mean_bp features")
        
        # BMI interactions
        if 'BMI' in X.columns:
            if 'currentSmoker' in X.columns:
                X['bmi_smoker'] = X['BMI'] * X['currentSmoker']
                self.logger.debug("Created bmi_smoker interaction")
            
            if 'diabetes' in X.columns:
                X['bmi_diabetes'] = X['BMI'] * X['diabetes']
                self.logger.debug("Created bmi_diabetes interaction")
        
        # Cholesterol and glucose interactions
        if 'totChol' in X.columns and 'glucose' in X.columns:
            X['chol_glucose'] = X['totChol'] * X['glucose'] / 1000
            self.logger.debug("Created chol_glucose interaction")
        
        return X
    
    def _create_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create binning features.
        
        Args:
            X: Input DataFrame
        
        Returns:
            DataFrame with binning features
        """
        # Age bins
        if 'age' in X.columns:
            bins = [0, 40, 50, 60, 70, 100]
            labels = [0, 1, 2, 3, 4]  # 0: <40, 4: 70+
            X['age_bin'] = pd.cut(X['age'], bins=bins, labels=labels).astype(float)
            self.logger.debug("Created age_bin feature")
        
        # BMI bins
        if 'BMI' in X.columns:
            bins = [0, 18.5, 25, 30, 100]
            labels = [0, 1, 2, 3]  # 0: Underweight, 3: Obese
            X['bmi_bin'] = pd.cut(X['BMI'], bins=bins, labels=labels).astype(float)
            self.logger.debug("Created bmi_bin feature")
        
        # Blood pressure bins
        if 'sysBP' in X.columns:
            bins = [0, 120, 130, 140, 180, 300]
            labels = [0, 1, 2, 3, 4]  # 0: Normal, 4: Crisis
            X['bp_bin'] = pd.cut(X['sysBP'], bins=bins, labels=labels).astype(float)
            self.logger.debug("Created bp_bin feature")
        
        # Glucose bins
        if 'glucose' in X.columns:
            bins = [0, 70, 100, 126, 200]
            labels = [0, 1, 2, 3]  # 0: Low, 3: High
            X['glucose_bin'] = pd.cut(X['glucose'], bins=bins, labels=labels).astype(float)
            self.logger.debug("Created glucose_bin feature")
        
        return X
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        return self.feature_names_