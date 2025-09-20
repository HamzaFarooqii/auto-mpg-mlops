import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from app.model import AutoMPGModel

class TestAutoMPGModel:
    
    def setup_method(self):
        """Setup test data"""
        self.model = AutoMPGModel()
        
        # Create sample test data with more samples for better training
        self.sample_data = pd.DataFrame({
            'cylinders': [4, 6, 8, 4, 6, 8, 4, 6, 8, 4],
            'displacement': [150.0, 250.0, 350.0, 160.0, 260.0, 360.0, 170.0, 270.0, 370.0, 180.0],
            'horsepower': [100.0, 150.0, 200.0, 110.0, 160.0, 210.0, 120.0, 170.0, 220.0, 130.0],
            'weight': [3000.0, 3500.0, 4000.0, 3100.0, 3600.0, 4100.0, 3200.0, 3700.0, 4200.0, 3300.0],
            'acceleration': [15.0, 12.0, 10.0, 14.0, 11.0, 9.0, 13.0, 10.0, 8.0, 12.0],
            'model year': [1980, 1990, 2000, 1981, 1991, 2001, 1982, 1992, 2002, 1983],
            'origin': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        })
        
        self.sample_target = pd.Series([25.0, 20.0, 15.0, 24.0, 19.0, 14.0, 23.0, 18.0, 13.0, 22.0])
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert not self.model.is_trained
        assert self.model.feature_columns is None
    
    def test_load_data(self):
        """Test data loading functionality"""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("mpg,cylinders,displacement,horsepower,weight,acceleration,model year,origin,car name\n")
            f.write("25.0,4,150.0,100.0,3000.0,15.0,1980,1,chevrolet\n")
            f.write("20.0,6,250.0,150.0,3500.0,12.0,1990,2,ford\n")
            temp_path = f.name
        
        try:
            X, y = self.model.load_data(temp_path)
            assert len(X) == 2
            assert len(y) == 2
            assert 'mpg' not in X.columns
        finally:
            os.unlink(temp_path)
    
    def test_train_model(self):
        """Test model training"""
        results = self.model.train(self.sample_data, self.sample_target)
        
        assert self.model.is_trained
        assert 'mse' in results
        assert 'r2_score' in results
        assert 'test_size' in results
        # Check if R2 score is valid (not NaN)
        assert not np.isnan(results['r2_score'])
        assert results['r2_score'] >= -1.0  # R2 can be negative but should be reasonable
    
    def test_predict(self):
        """Test prediction functionality"""
        # Train model first
        self.model.train(self.sample_data, self.sample_target)
        
        # Test prediction with dictionary input
        test_input = {
            'cylinders': 4,
            'displacement': 150.0,
            'horsepower': 100.0,
            'weight': 3000.0,
            'acceleration': 15.0,
            'model year': 1980,
            'origin': 1
        }
        
        prediction = self.model.predict(test_input)
        assert isinstance(prediction, float)
        assert prediction > 0
    
    def test_predict_without_training(self):
        """Test prediction without training should raise exception"""
        with pytest.raises(Exception, match="Model must be trained"):
            self.model.predict({'cylinders': 4, 'displacement': 150.0, 'horsepower': 100.0, 
                              'weight': 3000.0, 'acceleration': 15.0, 'model year': 1980, 'origin': 1})
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Train model
        self.model.train(self.sample_data, self.sample_target)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.model.save_model(temp_path)
            assert os.path.exists(temp_path)
            
            # Create new model and load
            new_model = AutoMPGModel()
            new_model.load_model(temp_path)
            
            assert new_model.is_trained
            assert new_model.feature_columns == self.model.feature_columns
        finally:
            os.unlink(temp_path)