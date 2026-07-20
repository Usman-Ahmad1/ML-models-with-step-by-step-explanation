import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from src.logger import project_logger
from src.config import config

class SHAPAnalyzer:
    """
    Handles SHAP-based model explainability.
    Supports both tree-based and linear models.
    """
    
    def __init__(self, model, feature_names: List[str], background_data: np.ndarray):
        """
        Initialize SHAPAnalyzer.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            background_data: Background data for SHAP (sample of training data)
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self.shap_values = None
        self.logger = project_logger
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """
        Detect the model type to use the appropriate SHAP explainer.
        
        Returns:
            'tree', 'linear', or 'unknown'
        """
        model_class = str(type(self.model))
        
        # Tree-based models
        tree_models = ['XGBClassifier', 'XGBRegressor', 'RandomForest', 
                       'DecisionTree', 'LGBM', 'CatBoost']
        if any(name in model_class for name in tree_models):
            return 'tree'
        
        # Linear models
        linear_models = ['LogisticRegression', 'LinearRegression', 
                         'Ridge', 'Lasso', 'ElasticNet']
        if any(name in model_class for name in linear_models):
            return 'linear'
        
        return 'unknown'
    
    def create_explainer(self, **kwargs) -> Optional[shap.Explainer]:
        """
        Create appropriate SHAP explainer based on model type.
        
        Args:
            **kwargs: Additional arguments for the explainer
        
        Returns:
            SHAP explainer instance
        """
        self.logger.info(f"Creating SHAP explainer for {self.model_type} model...")
        
        try:
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(
                    self.model,
                    self.background_data,
                    **kwargs
                )
                self.logger.info("Created TreeExplainer")
                
            elif self.model_type == 'linear':
                # For linear models, use LinearExplainer
                self.explainer = shap.LinearExplainer(
                    self.model,
                    self.background_data,
                    **kwargs
                )
                self.logger.info("Created LinearExplainer")
                
            else:
                # Fallback to KernelExplainer (slower but works for any model)
                self.logger.warning("Using KernelExplainer (slower but universal)")
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self.background_data,
                    **kwargs
                )
                self.logger.info("Created KernelExplainer")
                
            return self.explainer
            
        except Exception as e:
            self.logger.error(f"Failed to create explainer: {str(e)}")
            return None
    
    def explain(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Data to explain
        
        Returns:
            SHAP values
        """
        if self.explainer is None:
            self.create_explainer()
        
        if self.explainer is None:
            self.logger.error("Explainer creation failed")
            return None
        
        self.logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        try:
            if self.model_type == 'linear':
                # Linear explainer returns SHAP values directly
                self.shap_values = self.explainer.shap_values(X)
            else:
                # Tree and Kernel explainers
                shap_output = self.explainer.shap_values(X)
                # Handle different output formats
                if isinstance(shap_output, list) and len(shap_output) == 2:
                    # Binary classification: use the positive class
                    self.shap_values = shap_output[1]
                elif isinstance(shap_output, list) and len(shap_output) > 0:
                    # If it's a list of arrays, use the first one
                    self.shap_values = shap_output[0]
                else:
                    self.shap_values = shap_output
            
            self.logger.info("SHAP values computed successfully")
            return self.shap_values
            
        except Exception as e:
            self.logger.error(f"SHAP computation failed: {str(e)}")
            return None
    
    def global_importance(self, X: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Args:
            X: Data to explain (if None, use background_data)
        
        Returns:
            DataFrame with feature importance
        """
        if X is None:
            X = self.background_data
        
        if self.shap_values is None:
            self.explain(X)
        
        if self.shap_values is None:
            self.logger.error("No SHAP values available")
            return pd.DataFrame({'feature': [], 'mean_shap': []})
        
        # Ensure shap_values is 2D
        if len(self.shap_values.shape) == 1:
            # If 1D, reshape to 2D
            self.shap_values = self.shap_values.reshape(-1, 1)
        
        # Calculate mean absolute SHAP values
        if len(self.shap_values.shape) == 2:
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        else:
            mean_abs_shap = np.abs(self.shap_values).flatten()
        
        # Ensure we have the right number of features
        if len(mean_abs_shap) != len(self.feature_names):
            # If mismatch, pad or truncate
            if len(mean_abs_shap) < len(self.feature_names):
                # Pad with zeros
                mean_abs_shap = np.pad(mean_abs_shap, (0, len(self.feature_names) - len(mean_abs_shap)))
            else:
                # Truncate
                mean_abs_shap = mean_abs_shap[:len(self.feature_names)]
        
        # Create DataFrame properly
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_abs_shap
        }).sort_values('mean_shap', ascending=False)
        
        return importance_df
    
    def plot_summary(self, X: Optional[np.ndarray] = None, 
                    save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create SHAP summary plot (beeswarm).
        
        Args:
            X: Data to explain
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if X is None:
            X = self.background_data
        
        if self.shap_values is None:
            self.explain(X)
        
        if self.shap_values is None:
            self.logger.error("No SHAP values available for plotting")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Ensure shap_values is 2D
            if len(self.shap_values.shape) == 1:
                shap_values_2d = self.shap_values.reshape(-1, 1)
            else:
                shap_values_2d = self.shap_values
            
            shap.summary_plot(
                shap_values_2d, X,
                feature_names=self.feature_names,
                show=False,
                max_display=20
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Summary plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create summary plot: {str(e)}")
            return None
    
    def plot_bar(self, X: Optional[np.ndarray] = None,
                save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create SHAP bar plot.
        
        Args:
            X: Data to explain
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if X is None:
            X = self.background_data
        
        if self.shap_values is None:
            self.explain(X)
        
        if self.shap_values is None:
            self.logger.error("No SHAP values available for plotting")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Ensure shap_values is 2D
            if len(self.shap_values.shape) == 1:
                shap_values_2d = self.shap_values.reshape(-1, 1)
            else:
                shap_values_2d = self.shap_values
            
            shap.summary_plot(
                shap_values_2d, X,
                feature_names=self.feature_names,
                plot_type='bar',
                show=False
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Bar plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create bar plot: {str(e)}")
            return None
    
    def plot_waterfall(self, X: np.ndarray, idx: int,
                      save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X: Data to explain
            idx: Index of the sample to explain
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.explain(X)
        
        if self.shap_values is None:
            self.logger.error("No SHAP values available for plotting")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get SHAP value for the specific sample
            shap_single = self.shap_values[idx]
            
            # Ensure shap_single is 1D
            if len(shap_single.shape) > 1:
                shap_single = shap_single.flatten()
            
            # Handle expected value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
                    expected_value = expected_value[1]
            else:
                expected_value = 0.0
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_single,
                    base_values=expected_value,
                    data=X[idx],
                    feature_names=self.feature_names
                ),
                show=False
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Waterfall plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create waterfall plot: {str(e)}")
            return None
    
    def get_top_features(self, X: Optional[np.ndarray] = None, n: int = 10) -> List[str]:
        """
        Get top n features by SHAP importance.
        
        Args:
            X: Data to explain
            n: Number of top features to return
        
        Returns:
            List of top feature names
        """
        importance_df = self.global_importance(X)
        if importance_df.empty:
            return []
        return importance_df['feature'].head(n).tolist()
    
    def explain_prediction(self, X: np.ndarray, idx: int) -> Dict[str, Any]:
        """
        Get comprehensive explanation for a single prediction.
        
        Args:
            X: Data to explain
            idx: Index of the sample to explain
        
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.explain(X)
        
        if self.shap_values is None:
            return {
                'prediction_probability': 0.0,
                'base_value': 0.0,
                'feature_contributions': {},
                'top_contributors': [],
                'risk_category': 'Unknown'
            }
        
        shap_single = self.shap_values[idx]
        
        # Handle expected value for different explainers
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
                expected_value = expected_value[1]
        else:
            expected_value = 0.0
        
        # Get prediction
        try:
            pred_proba = self.model.predict_proba(X[idx:idx+1])[0, 1]
        except:
            pred_proba = float(self.model.predict(X[idx:idx+1])[0])
        
        # Get feature contributions
        feature_contributions = {
            feature: value for feature, value in zip(self.feature_names, shap_single)
        }
        
        # Sort contributions by absolute value
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'prediction_probability': float(pred_proba),
            'base_value': float(expected_value) if expected_value else 0.0,
            'feature_contributions': feature_contributions,
            'top_contributors': sorted_contributions[:5],
            'risk_category': self._get_risk_category(float(pred_proba))
        }
    
    def _get_risk_category(self, probability: float) -> str:
        """
        Map probability to risk category.
        
        Args:
            probability: Prediction probability
        
        Returns:
            Risk category string
        """
        if probability < 0.1:
            return "Very Low Risk"
        elif probability < 0.2:
            return "Low Risk"
        elif probability < 0.3:
            return "Moderate Risk"
        elif probability < 0.5:
            return "High Risk"
        else:
            return "Very High Risk"
