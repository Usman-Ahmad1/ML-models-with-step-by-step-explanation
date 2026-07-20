"""
Configuration management for the project.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Data configuration"""
    raw_data_path: str = os.getenv('RAW_DATA_PATH', 'data/raw/framingham.csv')
    processed_data_path: str = os.getenv('PROCESSED_DATA_PATH', 'data/processed/')
    train_size: float = float(os.getenv('TRAIN_SIZE', 0.8))
    test_size: float = float(os.getenv('TEST_SIZE', 0.2))
    val_size: float = float(os.getenv('VAL_SIZE', 0.1))
    random_seed: int = int(os.getenv('RANDOM_SEED', 42))
    
@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = os.getenv('MODEL_TYPE', 'xgboost')
    n_trials: int = int(os.getenv('N_TRIALS', 100))
    n_jobs: int = int(os.getenv('N_JOBS', -1))
    early_stopping_rounds: int = int(os.getenv('EARLY_STOPPING_ROUNDS', 50))
    verbosity: int = int(os.getenv('VERBOSITY', 1))
    
@dataclass
class ExplainabilityConfig:
    """Explainability configuration"""
    shap_sample_size: int = int(os.getenv('SHAP_SAMPLE_SIZE', 100))
    
@dataclass
class AppConfig:
    """Application configuration"""
    title: str = os.getenv('APP_TITLE', 'Cardiovascular Risk Predictor')
    icon: str = os.getenv('APP_ICON', '❤️')
    
class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.explainability = ExplainabilityConfig()
        self.app = AppConfig()
        self.project_name = os.getenv('PROJECT_NAME', 'Healthcare Risk Predictor')
        self.version = os.getenv('VERSION', '1.0.0')
        self.seed = int(os.getenv('RANDOM_SEED', 42))
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'project_name': self.project_name,
            'version': self.version,
            'seed': self.seed,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'explainability': self.explainability.__dict__,
            'app': self.app.__dict__
        }

# Singleton config instance
config = Config()