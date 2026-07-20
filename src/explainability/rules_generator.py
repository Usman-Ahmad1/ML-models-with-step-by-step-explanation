"""
Clinical decision support rules generator from SHAP analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.logger import project_logger

class RulesGenerator:
    """
    Generates clinical decision support rules from SHAP analysis.
    """
    
    def __init__(self, feature_names: List[str], shap_analyzer):
        """
        Initialize RulesGenerator.
        
        Args:
            feature_names: List of feature names
            shap_analyzer: SHAPAnalyzer instance
        """
        self.feature_names = feature_names
        self.shap_analyzer = shap_analyzer
        self.logger = project_logger
        self.rules = []
        
    def generate_rules(self, X: np.ndarray, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Generate decision rules from SHAP analysis.
        
        Args:
            X: Data to analyze
            threshold: Contribution threshold for rule generation
        
        Returns:
            List of rules
        """
        self.logger.info("Generating clinical decision rules...")
        
        # Get global SHAP contributions
        contributions = self.shap_analyzer.global_importance(X)
        
        rules = []
        
        # Create clinical rules based on features
        for feature in self.feature_names:
            # Define rules based on common clinical knowledge
            if 'age' in feature.lower():
                rules.append({
                    'feature': feature,
                    'condition': 'age > 60',
                    'recommendation': 'Consider cardiovascular risk assessment and preventive care',
                    'clinical_evidence': 'Age is a major risk factor for CVD'
                })
            
            if 'sysbp' in feature.lower() or 'blood_pressure' in feature.lower():
                rules.append({
                    'feature': feature,
                    'condition': 'sysBP > 140',
                    'recommendation': 'Monitor blood pressure, consider antihypertensive therapy',
                    'clinical_evidence': 'Elevated BP is a leading CVD risk factor'
                })
            
            if 'bmi' in feature.lower():
                rules.append({
                    'feature': feature,
                    'condition': 'BMI > 30',
                    'recommendation': 'Weight management program, dietary consultation',
                    'clinical_evidence': 'Obesity increases cardiovascular risk'
                })
            
            if 'smoker' in feature.lower() or 'currentsmoker' in feature.lower():
                rules.append({
                    'feature': feature,
                    'condition': 'currentSmoker == 1',
                    'recommendation': 'Smoking cessation program recommended',
                    'clinical_evidence': 'Smoking significantly increases CVD risk'
                })
            
            if 'diabetes' in feature.lower():
                rules.append({
                    'feature': feature,
                    'condition': 'diabetes == 1',
                    'recommendation': 'Diabetes management, regular glucose monitoring',
                    'clinical_evidence': 'Diabetes is a major CVD risk factor'
                })
            
            if 'totchol' in feature.lower() or 'cholesterol' in feature.lower():
                rules.append({
                    'feature': feature,
                    'condition': 'totChol > 240',
                    'recommendation': 'Lipid management, consider statin therapy',
                    'clinical_evidence': 'High cholesterol increases CVD risk'
                })
            
            if 'glucose' in feature.lower():
                rules.append({
                    'feature': feature,
                    'condition': 'glucose > 100',
                    'recommendation': 'Check for diabetes, glucose monitoring',
                    'clinical_evidence': 'Impaired fasting glucose indicates prediabetes'
                })
        
        self.rules = rules
        self.logger.info(f"Generated {len(rules)} clinical rules")
        
        return rules
    
    def apply_rules(self, X: np.ndarray, feature_values: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Apply rules to a specific patient.
        
        Args:
            X: Full feature array
            feature_values: Dictionary of feature values for the patient
        
        Returns:
            List of applicable rules
        """
        applicable_rules = []
        
        for rule in self.rules:
            try:
                # Evaluate rule condition (simplified)
                condition = rule['condition']
                
                # Split condition into feature and threshold
                if '>' in condition:
                    feature, threshold = condition.split('>')
                    feature = feature.strip()
                    threshold = float(threshold.strip())
                    
                    # Check if condition applies
                    if feature in feature_values and feature_values[feature] > threshold:
                        applicable_rules.append(rule)
                
                elif '==' in condition:
                    feature, value = condition.split('==')
                    feature = feature.strip()
                    value = int(value.strip())
                    
                    if feature in feature_values and feature_values[feature] == value:
                        applicable_rules.append(rule)
            
            except Exception as e:
                self.logger.warning(f"Error applying rule {rule}: {str(e)}")
                continue
        
        return applicable_rules
    
    def get_rule_impact(self, shap_values: np.ndarray, feature_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Get impact of rules for a specific patient.
        
        Args:
            shap_values: SHAP values for the patient
            feature_values: Feature values for the patient
        
        Returns:
            Dictionary with rule impacts
        """
        # Map features to SHAP contributions
        feature_shap = dict(zip(self.feature_names, shap_values))
        
        rule_impacts = {}
        
        for rule in self.rules:
            feature = rule['feature']
            if feature in feature_shap:
                rule_impacts[rule['recommendation']] = {
                    'shap_contribution': feature_shap[feature],
                    'feature_value': feature_values.get(feature, None),
                    'rule': rule
                }
        
        return rule_impacts
    
    def generate_health_report(self, X: np.ndarray, idx: int) -> Dict[str, Any]:
        """
        Generate a comprehensive health report for a patient.
        
        Args:
            X: Data to analyze
            idx: Index of the patient
        
        Returns:
            Dictionary with health report
        """
        # Get SHAP explanation
        explanation = self.shap_analyzer.explain_prediction(X, idx)
        
        # Get feature values
        feature_values = dict(zip(self.feature_names, X[idx]))
        
        # Get applicable rules
        applicable_rules = self.apply_rules(X, feature_values)
        
        # Get rule impacts
        rule_impacts = self.get_rule_impact(
            self.shap_analyzer.shap_values[idx],
            feature_values
        )
        
        return {
            'risk_score': explanation['prediction_probability'],
            'risk_category': explanation['risk_category'],
            'top_risk_factors': [
                {
                    'feature': feature,
                    'contribution': value,
                    'current_value': feature_values.get(feature, None)
                }
                for feature, value in explanation['top_contributors']
            ],
            'applicable_rules': applicable_rules,
            'rule_impacts': rule_impacts,
            'recommendations': [
                rule['recommendation'] for rule in applicable_rules
            ],
            'feature_values': feature_values
        }