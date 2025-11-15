import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys
CURRENT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parents[1] if len(CURRENT_DIR.parents) > 1 else CURRENT_DIR
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DATA_DIR = (PROJECT_ROOT / 'data' / 'raw')
TEMPLATE_DIR = PROJECT_ROOT / 'templates'
STATIC_DIR = PROJECT_ROOT / 'static'
from ocr.image_preprocessor import ImagePreprocessor
from models.code_analyzer import CodeQualityAnalyzer
from utils.feedback_generator import FeedbackGenerator

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = str(DATA_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Initialize components
preprocessor = ImagePreprocessor()
analyzer = CodeQualityAnalyzer()
feedback_generator = FeedbackGenerator()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Code Quality Assessment API'})


@app.route('/api/analyze', methods=['POST'])
def analyze_code_image():
    """
    Main endpoint to analyze code quality from uploaded image
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        # Process the image
        ocr_result = preprocessor.process_code_image(file_path)

        if not ocr_result['success']:
            return jsonify({
                'error': 'Failed to process image',
                'details': ocr_result.get('error', 'Unknown error')
            }), 500

        # Analyze extracted code
        extracted_text = ocr_result['extracted_text']
        if not extracted_text.strip():
            return jsonify({
                'error': 'No code text could be extracted from the image',
                'suggestion': 'Please ensure the image contains clear, readable code'
            }), 400

        # Perform code quality analysis
        analysis_result = analyzer.analyze_code(extracted_text)

        # Generate educational feedback
        feedback = feedback_generator.generate_feedback(analysis_result)

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass  # File cleanup failed, but don't fail the request

        # Return comprehensive response
        return jsonify({
            'success': True,
            'extracted_text': extracted_text,
            'code_regions': ocr_result['code_regions'],
            'quality_analysis': analysis_result,
            'educational_feedback': feedback,
            'processing_info': {
                'filename': filename,
                'text_length': len(extracted_text),
                'regions_detected': len(ocr_result['code_regions'])
            }
        })

    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text_directly():
    """
    Endpoint to analyze code quality from direct text input
    """
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'No code text provided'}), 400

        code_text = data['code']
        if not code_text.strip():
            return jsonify({'error': 'Empty code text provided'}), 400

        # Perform code quality analysis
        analysis_result = analyzer.analyze_code(code_text)

        # Generate educational feedback
        feedback = feedback_generator.generate_feedback(analysis_result)

        return jsonify({
            'success': True,
            'quality_analysis': analysis_result,
            'educational_feedback': feedback
        })

    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/feedback/<quality_score>', methods=['GET'])
def get_contextual_feedback(quality_score):
    """
    Get contextual feedback based on quality score
    """
    try:
        score = float(quality_score)
        if not 0 <= score <= 100:
            return jsonify({'error': 'Quality score must be between 0 and 100'}), 400

        feedback = feedback_generator.get_score_based_feedback(score)
        return jsonify({'feedback': feedback})

    except ValueError:
        return jsonify({'error': 'Invalid quality score format'}), 400
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print(f"Starting Code Quality Assessment API on port {port}")
    print(f"Debug mode: {debug}")

    app.run(host='0.0.0.0', port=port, debug=debug)
