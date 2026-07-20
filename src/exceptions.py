"""
Custom exceptions for the Healthcare Risk Predictor project.
"""

class HealthcarePredictorError(Exception):
    """Base exception for all project exceptions."""
    pass

class DataLoadingError(HealthcarePredictorError):
    """Raised when data loading fails."""
    pass

class DataValidationError(HealthcarePredictorError):
    """Raised when data validation fails."""
    pass

class PreprocessingError(HealthcarePredictorError):
    """Raised when data preprocessing fails."""
    pass

class ModelTrainingError(HealthcarePredictorError):
    """Raised when model training fails."""
    pass

class ModelLoadingError(HealthcarePredictorError):
    """Raised when model loading fails."""
    pass

class ExplainabilityError(HealthcarePredictorError):
    """Raised when explainability analysis fails."""
    pass