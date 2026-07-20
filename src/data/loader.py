import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from src.logger import project_logger
from src.exceptions import DataLoadingError, DataValidationError
from src.config import config

class DataLoader:
    """Handles loading and validation of the Framingham dataset."""
    
    # Expected columns for validation
    EXPECTED_COLUMNS = [
        'male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
        'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
        'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose',
        'TenYearCHD'
    ]
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or config.data.raw_data_path
        self.logger = project_logger
        
    def load_data(self) -> pd.DataFrame:
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            
            if not Path(self.data_path).exists():
                raise DataLoadingError(f"Data file not found: {self.data_path}")
            
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise DataLoadingError(f"Data loading failed: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        try:
            self.logger.info("Validating data...")
            
            missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
            if missing_cols:
                raise DataValidationError(f"Missing columns: {missing_cols}")
            
            if 'TenYearCHD' not in df.columns:
                raise DataValidationError("Target column 'TenYearCHD' not found")
            
            valid_targets = df['TenYearCHD'].isin([0, 1])
            if not valid_targets.all():
                invalid_count = (~valid_targets).sum()
                raise DataValidationError(f"Target contains invalid values: {invalid_count} rows")
            
            self.logger.info("Data validation passed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise DataValidationError(f"Data validation failed: {str(e)}")
    
    def load_and_validate(self) -> pd.DataFrame:
        df = self.load_data()
        self.validate_data(df)
        return df
    
    def generate_sample_data(self, n_samples: int = 4240, random_seed: int = 42) -> pd.DataFrame:
        """Generate realistic Framingham-like data for demonstration."""
        self.logger.info(f"Generating {n_samples} sample data points...")
        np.random.seed(random_seed)
        
        # Generate realistic distributions based on Framingham statistics
        data = {
            'male': np.random.binomial(1, 0.52, n_samples),
            'age': np.random.normal(50, 10, n_samples).clip(30, 80).astype(int),
            'education': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'currentSmoker': np.random.binomial(1, 0.25, n_samples),
            'cigsPerDay': np.random.poisson(10, n_samples).clip(0, 40),
            'BPMeds': np.random.binomial(1, 0.08, n_samples),
            'prevalentStroke': np.random.binomial(1, 0.03, n_samples),
            'prevalentHyp': np.random.binomial(1, 0.25, n_samples),
            'diabetes': np.random.binomial(1, 0.05, n_samples),
            'totChol': np.random.normal(200, 40, n_samples).clip(100, 400),
            'sysBP': np.random.normal(125, 15, n_samples).clip(90, 200),
            'diaBP': np.random.normal(80, 10, n_samples).clip(60, 120),
            'BMI': np.random.normal(26, 4, n_samples).clip(15, 45),
            'heartRate': np.random.normal(72, 10, n_samples).clip(50, 100),
            'glucose': np.random.normal(85, 15, n_samples).clip(50, 200),
            'TenYearCHD': np.random.binomial(1, 0.15, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic correlations with the target
        # Age and CHD risk
        age_risk = 1 / (1 + np.exp(-(df['age'] - 50) / 10))
        df['TenYearCHD'] = np.random.binomial(1, 0.1 + 0.4 * age_risk, n_samples)
        
        # Blood pressure and CHD risk
        bp_risk = 1 / (1 + np.exp(-(df['sysBP'] - 120) / 20))
        df['TenYearCHD'] = np.maximum(
            df['TenYearCHD'],
            np.random.binomial(1, 0.1 + 0.3 * bp_risk, n_samples)
        )
        
        self.logger.info(f"Generated sample data with {len(df)} rows")
        return df
