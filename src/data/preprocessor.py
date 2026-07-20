import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Tuple, List, Optional, Dict, Any
from src.logger import project_logger
from src.exceptions import PreprocessingError
from src.config import config

class DataPreprocessor:
    '''
    Handles all data preprocessing including missing values,
    encoding, and scaling. NO feature engineering - uses raw data only.
    '''
    
    def __init__(self):
        '''Initialize preprocessor with default configurations.'''
        self.logger = project_logger
        self.preprocessor = None
        self.feature_names = None
        self.numeric_features = []
        self.categorical_features = []
        
    def identify_feature_types(self, df: pd.DataFrame, target_col: str = 'TenYearCHD') -> Tuple[List[str], List[str]]:
        '''
        Identify numeric and categorical features.
        '''
        exclude_cols = [target_col]
        
        # Define which columns should ALWAYS be numeric
        ALWAYS_NUMERIC = [
            'male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
            'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
            'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
        ]
        
        numeric_features = []
        categorical_features = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            # Force certain columns to be numeric
            if col in ALWAYS_NUMERIC:
                numeric_features.append(col)
                continue
            
            # Check if column contains mixed types
            if df[col].dtype == 'object':
                categorical_features.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # If numeric but has few unique values, treat as categorical
                unique_count = df[col].nunique()
                if unique_count <= 10 and col not in ALWAYS_NUMERIC:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                df[col] = df[col].astype(str)
                categorical_features.append(col)
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        self.logger.info(f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        self.logger.debug(f"Numeric: {numeric_features}")
        self.logger.debug(f"Categorical: {categorical_features}")
        
        return numeric_features, categorical_features
    
    def create_preprocessing_pipeline(self, 
                                     numeric_features: List[str],
                                     categorical_features: List[str],
                                     use_robust_scaler: bool = True) -> ColumnTransformer:
        '''
        Create a preprocessing pipeline with ColumnTransformer.
        '''
        self.logger.info("Creating preprocessing pipeline...")
        
        # Numeric pipeline with median imputation (handles nans)
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler() if use_robust_scaler else StandardScaler())
        ])
        
        # Categorical pipeline - only for true categorical columns
        if categorical_features:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            categorical_transformer = 'drop'
        
        # Combine transformers
        transformers = [('num', numeric_transformer, numeric_features)]
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(
            transformers,
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        self.feature_names = numeric_features + categorical_features
        
        return preprocessor
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Handle missing values with appropriate strategies.
        WARNING: This modifies the DataFrame in place.
        '''
        self.logger.info("Handling missing values...")
        
        # Numeric columns: median imputation
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self.logger.debug(f"Imputed {col} with median {median_val:.2f}")
        
        # Categorical columns: mode imputation
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                self.logger.debug(f"Imputed {col} with mode {mode_val}")
        
        return df
    
    def preprocess(self, 
                  df: pd.DataFrame, 
                  fit: bool = True,
                  target_col: str = 'TenYearCHD') -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        '''
        Complete preprocessing pipeline - RAW DATA ONLY, no feature engineering.
        '''
        self.logger.info("Starting preprocessing pipeline...")
        
        # Validate input
        if df.empty:
            raise PreprocessingError("Input DataFrame is empty")
        
        try:
            # Make a copy to avoid modifying original
            df_processed = df.copy()
            
            # Handle missing values first
            df_processed = self.handle_missing_values(df_processed)
            
            # Separate features and target
            if target_col in df_processed.columns:
                y = df_processed[target_col].values
                X = df_processed.drop(columns=[target_col])
            else:
                y = None
                X = df_processed
            
            # Convert all categorical columns to string type
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].astype(str)
            
            # Identify feature types
            self.identify_feature_types(X)
            
            # Create and apply preprocessing pipeline
            if self.preprocessor is None or fit:
                self.create_preprocessing_pipeline(
                    numeric_features=self.numeric_features,
                    categorical_features=self.categorical_features
                )
            
            if fit:
                X_transformed = self.preprocessor.fit_transform(X)
            else:
                X_transformed = self.preprocessor.transform(X)
            
            # Get feature names after preprocessing
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                self.feature_names = self.preprocessor.get_feature_names_out()
            else:
                self.feature_names = self.numeric_features + self.categorical_features
            
            self.logger.info(f"Preprocessing complete. Shape: {X_transformed.shape}")
            
            return X_transformed, y
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise PreprocessingError(f"Preprocessing failed: {str(e)}")
    
    def create_smote_pipeline(self, 
                             numeric_features: List[str],
                             categorical_features: List[str],
                             random_state: int = 42) -> ImbPipeline:
        '''
        Create a pipeline with SMOTE for handling class imbalance.
        '''
        self.logger.info("Creating pipeline with SMOTE...")
        
        preprocessor = self.create_preprocessing_pipeline(
            numeric_features, categorical_features
        )
        
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=random_state)),
            ('classifier', 'passthrough')
        ])
        
        return pipeline


__all__ = ['DataPreprocessor']
