"""
Complete pipeline runner script.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.logger import project_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.models.tuner import HyperparameterTuner
from src.explainability.shap_analyzer import SHAPAnalyzer
from src.explainability.rules_generator import RulesGenerator
from src.config import config

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run the complete pipeline."""
    logger = project_logger
    logger.info("Starting complete pipeline...")
    
    # 1. Load data
    logger.info("Loading data...")
    loader = DataLoader()
    
    # Generate sample data if needed
    if not Path(config.data.raw_data_path).exists():
        logger.info("Generating sample data...")
        df = loader.generate_sample_data()
        df.to_csv(config.data.raw_data_path, index=False)
    else:
        df = loader.load_and_validate()
    
    logger.info(f"Loaded {len(df)} samples")
    
    # 2. Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.preprocess(df)
    
    logger.info(f"Processed data shape: {X_processed.shape}")
    
    # 3. Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 4. Hyperparameter tuning
    logger.info("Running hyperparameter tuning...")
    tuner = HyperparameterTuner(n_trials=20)  # Reduce trials for speed
    tuning_results = tuner.tune_models(X_train, y_train)
    
    # 5. Train models
    logger.info("Training models...")
    trainer = ModelTrainer()
    models = trainer.create_models()
    
    # Update best parameters for XGBoost
    if 'xgboost' in models and 'xgboost' in tuning_results:
        best_params = tuning_results['xgboost']['best_params']
        models['xgboost'].model.set_params(**best_params)
        logger.info(f"Updated XGBoost with best params: {best_params}")
    
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # 6. SHAP Analysis
    logger.info("Performing SHAP analysis...")
    best_model = trainer.best_model
    
    # Use a sample of training data for SHAP background
    background_sample = X_train[:100]
    feature_names = preprocessor.feature_names
    
    shap_analyzer = SHAPAnalyzer(
        best_model.model,
        feature_names,
        background_sample
    )
    shap_analyzer.create_explainer()
    
    # Calculate SHAP values
    shap_values = shap_analyzer.explain(X_test)
    
    # Generate global importance
    importance_df = shap_analyzer.global_importance()
    logger.info(f"Top 5 features:\n{importance_df.head(5)}")
    
    # 7. Generate rules
    logger.info("Generating clinical rules...")
    rules_generator = RulesGenerator(feature_names, shap_analyzer)
    rules = rules_generator.generate_rules(X_train)
    
    # 8. Save models and artifacts
    logger.info("Saving artifacts...")
    
    # Save model
    model_path = "models/best_model.pkl"
    trainer.save_model(best_model.model, model_path)
    
    # Save preprocessor
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    
    # Save SHAP analyzer
    joblib.dump(shap_analyzer, "models/shap_analyzer.pkl")
    
    # Save results
    results_df = pd.DataFrame({
        'model': list(results.keys()),
        'cv_auc_mean': [results[m]['metrics']['cv_auc_mean'] for m in results],
        'test_auc': [results[m]['metrics']['test_auc'] for m in results],
        'test_auprc': [results[m]['metrics']['test_auprc'] for m in results],
        'test_f1': [results[m]['metrics']['test_f1'] for m in results]
    })
    results_df.to_csv("models/results.csv", index=False)
    
    # 9. Generate report
    logger.info("Generating report...")
    report_data = {
        'dataset_size': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': len(feature_names),
        'best_model': trainer.best_model_name,
        'best_auc': results[trainer.best_model_name]['metrics']['test_auc'],
        'top_features': importance_df.head(5).to_dict('records')
    }
    
    report_df = pd.DataFrame([report_data])
    report_df.to_csv("models/report_summary.csv", index=False)
    
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"Best model: {trainer.best_model_name}")
    logger.info(f"Test AUC: {results[trainer.best_model_name]['metrics']['test_auc']:.4f}")
    logger.info("=" * 50)
    
    return {
        'trainer': trainer,
        'preprocessor': preprocessor,
        'shap_analyzer': shap_analyzer,
        'rules_generator': rules_generator
    }

if __name__ == "__main__":
    main()