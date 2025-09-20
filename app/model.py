import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class AutoMPGModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_columns = None
        
    def load_data(self, data_path):
        """Load and preprocess the auto-mpg dataset"""
        try:
            # Load the CSV dataset
            df = pd.read_csv(data_path)
            
            # Replace '?' with NaN for proper handling
            df = df.replace('?', np.nan)
            
            # Convert numeric columns to proper types
            numeric_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values by dropping rows with NaN
            df = df.dropna()
            
            # Remove car_name as it's not useful for prediction
            if 'car name' in df.columns:
                df = df.drop('car name', axis=1)
            elif 'car_name' in df.columns:
                df = df.drop('car_name', axis=1)
            
            # Separate features and target
            self.feature_columns = [col for col in df.columns if col != 'mpg']
            X = df[self.feature_columns]
            y = df['mpg']
            
            return X, y
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def train(self, X, y):
        """Train the model"""
        try:
            # Set feature columns if not already set
            if self.feature_columns is None:
                self.feature_columns = [col for col in X.columns if col != 'mpg']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            return {
                'mse': mse,
                'r2_score': r2,
                'test_size': len(X_test)
            }
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict(self, features):
        """Make predictions on new data"""
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        if self.feature_columns is None:
            raise Exception("Feature columns not set. Model must be trained first.")
        
        try:
            # Ensure features are in the correct format
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, list):
                features = pd.DataFrame([features])
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in features.columns:
                    raise Exception(f"Missing feature: {col}")
            
            # Reorder columns to match training data
            features = features[self.feature_columns]
            
            prediction = self.model.predict(features)
            return float(prediction[0])
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def save_model(self, model_path):
        """Save the trained model"""
        if not self.is_trained:
            raise Exception("Model must be trained before saving")
        
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, model_path)
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
