"""
Custom transformers for the ML pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Tuple

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Select specific columns from DataFrame.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize ColumnSelector.
        
        Args:
            columns: List of columns to select
        """
        self.columns = columns
        
    def fit(self, X, y=None):
        """Fit the selector."""
        return self
    
    def transform(self, X):
        """Transform by selecting columns."""
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                return X
            return X[self.columns]
        else:
            if self.columns is None:
                return X
            return X[:, self.columns] if hasattr(X, 'shape') else X
    
    def get_feature_names_out(self):
        """Get feature names."""
        return self.columns

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Convert array to DataFrame with column names.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize DataFrameTransformer.
        
        Args:
            feature_names: Names for the columns
        """
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    def transform(self, X):
        """Transform to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names)