from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import os
import sys
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the predictor
from app.utils.glaucoma_predictor import GlaucomaPredictor

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Configure upload settings
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['predictor'] = None
    
    # Initialize predictor
    init_predictor(app)
    
    # Register blueprints
    from app.routes.prediction import prediction_bp
    app.register_blueprint(prediction_bp, url_prefix='/api')
    
    @app.route('/')
    def home():
        """Home endpoint with API information"""
        return jsonify({
            'message': 'AI Glaucoma Detection API',
            'status': 'running',
            'model_loaded': app.config['predictor'] is not None,
            'endpoints': [
                'GET / - API information',
                'POST /api/predict - Single image prediction',
                'POST /api/predict-batch - Batch image prediction',
                'GET /api/model-info - Model information'
            ]
        })
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': app.config['predictor'] is not None
        })
    
    @app.errorhandler(413)
    def too_large(e):
        """Handle file too large error"""
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 16MB.'
        }), 413
    
    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors"""
        return jsonify({
            'success': False,
            'error': 'Endpoint not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        """Handle 500 errors"""
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    
    return app

def init_predictor(app):
    """Initialize the glaucoma predictor"""
    logger.debug("🚀 Starting Glaucoma Detection Server...")
    
    # Model path
    model_path = os.path.join(project_root, 'model', 'best_glaucoma_model.pth')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found at: {model_path}")
        logger.error("Please ensure the model file exists in the correct location.")
        return
    
    # Check file permissions
    if not os.access(model_path, os.R_OK):
        logger.error(f"❌ Model file at {model_path} is not readable. Check permissions.")
        return
    
    try:
        logger.debug(f"📦 Loading model from: {model_path}")
        app.config['predictor'] = GlaucomaPredictor(model_path)
        logger.debug("✅ Glaucoma predictor initialized successfully!")
        logger.debug(f"🔧 Using device: {app.config['predictor'].device}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {str(e)}")
        logger.error("Full traceback:")
        traceback.print_exc()
        logger.error("Please check if the model file is compatible with GlaucomaPredictor.")
        app.config['predictor'] = None

def get_predictor():
    """Get the predictor instance"""
    if current_app.config.get('predictor') is None:
        logger.error("⚠️ Predictor is None, model not loaded.")
    return current_app.config['predictor']

if __name__ == '__main__':
    app = create_app()
    
    if app.config.get('predictor') is not None:
        logger.debug("🌟 Server ready! Available endpoints:")
        logger.debug("   - http://localhost:5000/ (API info)")
        logger.debug("   - http://localhost:5000/api/predict (Single prediction)")
        logger.debug("   - http://localhost:5000/api/predict-batch (Batch prediction)")
        logger.debug("   - http://localhost:5000/api/model-info (Model info)")
        logger.debug("   - http://localhost:5000/health (Health check)")
        logger.debug("="*50)
    else:
        logger.error("⚠️ Server starting without model loaded!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)