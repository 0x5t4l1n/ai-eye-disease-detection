from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from PIL import Image
import os
import logging
import traceback
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_predictor():
    """Get predictor from main app"""
    try:
        predictor = current_app.config.get('predictor')
        if predictor is None:
            logger.error("Predictor is None in prediction route")
        else:
            logger.debug("Predictor successfully retrieved in prediction route")
        return predictor
    except Exception as e:
        logger.error(f"Error retrieving predictor: {str(e)}")
        traceback.print_exc()
        return None

def serialize_floats(data):
    """Convert NumPy float32 values to Python float for JSON serialization"""
    if isinstance(data, dict):
        return {k: serialize_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_floats(item) for item in data]
    elif isinstance(data, np.floating):
        return float(data)
    return data

@prediction_bp.route('/predict', methods=['POST'])
def predict_glaucoma():
    """Endpoint to predict glaucoma severity from a single image"""
    logger.debug("Received request for /api/predict")
    try:
        predictor = get_predictor()
        if predictor is None:
            logger.error("Model not loaded for /api/predict")
            return jsonify({
                'success': False,
                'error': 'Model not loaded.'
            }), 500
        
        if 'image' not in request.files:
            logger.error("No image file provided in request")
            return jsonify({
                'success': False,
                'error': 'No image file provided.'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("No file selected in request")
            return jsonify({
                'success': False,
                'error': 'No file selected.'
            }), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        return_probabilities = request.form.get('return_probabilities', 'false').lower() == 'true'
        logger.debug(f"Return probabilities: {return_probabilities}")
        
        try:
            image = Image.open(file.stream).convert('RGB')
            result = predictor.predict(image, return_probabilities=return_probabilities)
            logger.debug(f"Prediction result: {result}")
            
            # Serialize float32 values
            result = serialize_floats(result)
            
            return jsonify({
                'success': True,
                'data': {
                    'cdr': result['cdr'],
                    'severity': result['severity_label'],
                    'severity_description': result['severity_description'],
                    'confidence': result['confidence'],
                    'risk_level': result['risk_level'],
                    **({'probabilities': result['class_probabilities']} if return_probabilities else {})
                },
                'filename': secure_filename(file.filename)
            }), 200
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 500
    except Exception as e:
        logger.error(f"Server error in /api/predict: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@prediction_bp.route('/predict-batch', methods=['POST'])
def predict_glaucoma_batch():
    """Endpoint to predict glaucoma severity for multiple images"""
    logger.debug("Received request for /api/predict-batch")
    try:
        predictor = get_predictor()
        if predictor is None:
            logger.error("Model not loaded for /api/predict-batch")
            return jsonify({
                'success': False,
                'error': 'Model not loaded.'
            }), 500
        
        uploaded_files = [request.files[key] for key in request.files if key.startswith('image') and request.files[key].filename != '' and allowed_file(request.files[key].filename)]
        if not uploaded_files:
            logger.error("No valid image files provided in batch request")
            return jsonify({
                'success': False,
                'error': 'No valid image files provided.'
            }), 400
        
        return_probabilities = request.form.get('return_probabilities', 'false').lower() == 'true'
        logger.debug(f"Return probabilities: {return_probabilities}")
        
        results = []
        for i, file in enumerate(uploaded_files):
            try:
                image = Image.open(file.stream).convert('RGB')
                result = predictor.predict(image, return_probabilities=return_probabilities)
                result = serialize_floats(result)
                results.append({
                    'filename': secure_filename(file.filename),
                    'image_index': i,
                    'success': True,
                    'cdr': result['cdr'],
                    'severity': result['severity_label'],
                    'severity_description': result['severity_description'],
                    'confidence': result['confidence'],
                    'risk_level': result['risk_level'],
                    **({'probabilities': result['class_probabilities']} if return_probabilities else {})
                })
            except Exception as e:
                logger.error(f"Error processing batch image {file.filename}: {str(e)}")
                traceback.print_exc()
                results.append({
                    'filename': secure_filename(file.filename),
                    'image_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        successful_predictions = sum(1 for r in results if r['success'])
        logger.debug(f"Batch prediction completed: {successful_predictions}/{len(uploaded_files)} successful")
        return jsonify({
            'success': True,
            'data': results,
            'total_images': len(uploaded_files),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(uploaded_files) - successful_predictions
        }), 200
    except Exception as e:
        logger.error(f"Server error in /api/predict-batch: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@prediction_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    logger.debug("Received request for /api/model-info")
    try:
        predictor = get_predictor()
        if predictor is None:
            logger.error("Model not loaded for /api/model-info")
            return jsonify({
                'success': False,
                'error': 'Model not loaded.'
            }), 500
        
        try:
            device = str(predictor.device)
            severity_labels = getattr(predictor, 'severity_labels', {})
            severity_descriptions = getattr(predictor, 'severity_descriptions', {})
        except AttributeError as e:
            logger.error(f"Attribute error accessing predictor properties: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Invalid predictor configuration: {str(e)}'
            }), 500

        logger.debug("Model info retrieved successfully")
        return jsonify({
            'success': True,
            'data': {
                'model_loaded': True,
                'device': device,
                'severity_classes': list(severity_labels.values()),
                'severity_descriptions': serialize_floats(severity_descriptions),
                'supported_formats': list(ALLOWED_EXTENSIONS),
                'model_type': 'GlaucomaSeverityModel',
                'architecture': 'ResNet101 with SE Blocks'
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error getting model info: {str(e)}'
        }), 500