from flask import Flask, request, jsonify
import os
import logging
from app.model import AutoMPGModel
from app.utils import validate_input_data, format_prediction_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model
model = AutoMPGModel()
model_path = 'models/auto_mpg_model.pkl'
data_path = 'data/auto-mpg.csv'

# Train model on startup if not already trained
if not os.path.exists(model_path):
    os.makedirs('models', exist_ok=True)
    try:
        X, y = model.load_data(data_path)
        training_results = model.train(X, y)
        model.save_model(model_path)
        logger.info(f"Model trained successfully. R2 Score: {training_results['r2_score']}")
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
else:
    try:
        model.load_model(model_path)
        logger.info("Pre-trained model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model.is_trained
    })

@app.route('/predict', methods=['POST'])
def predict_mpg():
    """Predict MPG for given car features"""
    try:
        # Check if model is trained
        if not model.is_trained:
            return jsonify({'error': 'Model not trained yet. Please wait for training to complete.'}), 503
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input data
        validated_data = validate_input_data(data)
        
        # Make prediction
        prediction = model.predict(validated_data)
        
        # Format response
        response = format_prediction_response(prediction, validated_data)
        
        logger.info(f"Prediction made: {prediction} MPG")
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with current data"""
    try:
        # Load and train model
        X, y = model.load_data(data_path)
        training_results = model.train(X, y)
        
        # Save updated model
        model.save_model(model_path)
        
        logger.info(f"Model retrained successfully. R2 Score: {training_results['r2_score']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully',
            'metrics': training_results
        })
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'is_trained': model.is_trained,
        'feature_columns': model.feature_columns if model.feature_columns else [],
        'model_type': 'RandomForestRegressor'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
