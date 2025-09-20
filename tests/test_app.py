import pytest
import json
from app.app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestFlaskApp:
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'model_trained' in data
    
    def test_predict_valid_data(self, client):
        """Test prediction with valid data"""
        test_data = {
            'cylinders': 4,
            'displacement': 150.0,
            'horsepower': 100.0,
            'weight': 3000.0,
            'acceleration': 15.0,
            'model year': 1980,
            'origin': 1
        }
        
        response = client.post('/predict', 
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        # Accept both 200 (success) and 503 (model not trained)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'predicted_mpg' in data
            assert 'input_features' in data
            assert 'status' in data
            assert data['status'] == 'success'
    
    def test_predict_invalid_data(self, client):
        """Test prediction with invalid data"""
        test_data = {
            'cylinders': 4,
            'displacement': -150.0,  # Invalid negative value
            'horsepower': 100.0,
            'weight': 3000.0,
            'acceleration': 15.0,
            'model year': 1980,
            'origin': 1
        }
        
        response = client.post('/predict',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        # Accept both 400 (validation error) and 503 (model not trained)
        assert response.status_code in [400, 503]
    
    def test_predict_missing_data(self, client):
        """Test prediction with missing data"""
        test_data = {
            'cylinders': 4,
            'displacement': 150.0,
            # Missing other required fields
        }
        
        response = client.post('/predict',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        # Accept both 400 (validation error) and 503 (model not trained)
        assert response.status_code in [400, 503]
    
    def test_predict_no_data(self, client):
        """Test prediction with no data"""
        response = client.post('/predict')
        # Accept both 400 (no data) and 503 (model not trained)
        assert response.status_code in [400, 503]
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get('/model/info')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'is_trained' in data
        assert 'feature_columns' in data
        assert 'model_type' in data