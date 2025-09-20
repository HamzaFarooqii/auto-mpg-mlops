import pandas as pd
import numpy as np
from typing import Dict, Any

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean input data for prediction"""
    
    # Define expected features and their types (matching CSV column names)
    expected_features = {
        'cylinders': int,
        'displacement': float,
        'horsepower': float,
        'weight': float,
        'acceleration': float,
        'model year': int,  # Note: space in column name
        'origin': int
    }
    
    validated_data = {}
    
    for feature, expected_type in expected_features.items():
        if feature not in data:
            raise ValueError(f"Missing required feature: {feature}")
        
        try:
            # Convert to expected type
            validated_data[feature] = expected_type(data[feature])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {feature}: {data[feature]}")
    
    # Additional validation rules
    if validated_data['cylinders'] not in [3, 4, 5, 6, 8]:
        raise ValueError("Cylinders must be 3, 4, 5, 6, or 8")
    
    if validated_data['displacement'] <= 0:
        raise ValueError("Displacement must be positive")
    
    if validated_data['horsepower'] <= 0:
        raise ValueError("Horsepower must be positive")
    
    if validated_data['weight'] <= 0:
        raise ValueError("Weight must be positive")
    
    if validated_data['acceleration'] <= 0:
        raise ValueError("Acceleration must be positive")
    
    if validated_data['model year'] < 1970 or validated_data['model year'] > 2025:
        raise ValueError("Model year must be between 1970 and 2025")
    
    if validated_data['origin'] not in [1, 2, 3]:
        raise ValueError("Origin must be 1, 2, or 3")
    
    return validated_data

def format_prediction_response(prediction: float, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format the prediction response"""
    return {
        'predicted_mpg': round(prediction, 2),
        'input_features': input_data,
        'status': 'success'
    }

def calculate_model_metrics(y_true, y_pred):
    """Calculate various model performance metrics"""
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2_score': round(r2, 4)
    }