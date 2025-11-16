"""
Advanced API with transformer model support.

This is an enhanced version of the API that supports both rule-based
and transformer-based code quality analysis.
"""
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
MODEL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'models'
TRANSFORMER_MODEL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'transformer_models'

from ocr.image_preprocessor import ImagePreprocessor
from models.hybrid_analyzer import HybridCodeAnalyzer
from utils.feedback_generator import FeedbackGenerator

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = str(DATA_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Initialize components
preprocessor = ImagePreprocessor()
feedback_generator = FeedbackGenerator()

# Initialize hybrid analyzer with ML models if available
transformer_path = TRANSFORMER_MODEL_DIR
baseline_path = MODEL_DIR / "baseline_quality_model.joblib"

analyzer = HybridCodeAnalyzer(
    use_transformers=True,
    transformer_model_path=transformer_path if transformer_path.exists() else None,
    use_baseline_ml=True,
    baseline_model_path=baseline_path if baseline_path.exists() else None,
)

print("\n" + "=" * 80)
print("CODE QUALITY ASSESSMENT API - ADVANCED MODE")
print("=" * 80)
model_info = analyzer.get_model_info()
print(f"Rule-based analyzer: ✓")
print(f"Transformer model: {'✓' if model_info['transformer_available'] else '✗'}")
print(f"Baseline ML model: {'✓' if model_info['baseline_ml_available'] else '✗'}")
print(f"Recommended strategy: {model_info['recommended_strategy']}")
print("=" * 80 + "\n")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with model info"""
    return jsonify({
        'status': 'healthy',
        'service': 'Code Quality Assessment API (Advanced)',
        'models': analyzer.get_model_info(),
    })


@app.route('/api/models', methods=['GET'])
def get_model_info():
    """Get information about available models"""
    return jsonify(analyzer.get_model_info())


@app.route('/api/analyze', methods=['POST'])
def analyze_code_image():
    """
    Main endpoint to analyze code quality from uploaded image.
    Supports advanced ML models.
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

        # Get analysis strategy from request
        ensemble_strategy = request.form.get('strategy', 'weighted_average')
        valid_strategies = ['rule_only', 'ml_only', 'weighted_average', 'ml_override']
        if ensemble_strategy not in valid_strategies:
            return jsonify({
                'error': f'Invalid strategy. Must be one of: {valid_strategies}'
            }), 400

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

        # Perform code quality analysis with hybrid analyzer
        analysis_result = analyzer.analyze_code(
            extracted_text,
            ensemble_strategy=ensemble_strategy,
        )

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
                'regions_detected': len(ocr_result['code_regions']),
                'analysis_strategy': ensemble_strategy,
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
    Endpoint to analyze code quality from direct text input.
    Supports advanced ML models.
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'code' not in data:
            return jsonify({'error': 'No code text provided'}), 400

        code_text = data['code']
        if not code_text.strip():
            return jsonify({'error': 'Empty code text provided'}), 400

        # Get analysis strategy from request
        ensemble_strategy = data.get('strategy', 'weighted_average')
        valid_strategies = ['rule_only', 'ml_only', 'weighted_average', 'ml_override']
        if ensemble_strategy not in valid_strategies:
            return jsonify({
                'error': f'Invalid strategy. Must be one of: {valid_strategies}'
            }), 400

        # Get attention visualization flag
        return_attention = data.get('return_attention', False)

        # Perform code quality analysis
        analysis_result = analyzer.analyze_code(
            code_text,
            ensemble_strategy=ensemble_strategy,
        )

        # Generate educational feedback
        feedback = feedback_generator.generate_feedback(analysis_result)

        response = {
            'success': True,
            'quality_analysis': analysis_result,
            'educational_feedback': feedback,
        }

        # Add attention visualization if requested and transformer model is used
        if return_attention and analyzer.transformer_model:
            try:
                attention_pred = analyzer.transformer_model.predict(
                    [code_text],
                    return_attention=True,
                )
                if 'attention_weights' in attention_pred:
                    response['attention_available'] = True
                    # Note: Actual attention weights are large, so we just indicate availability
                    # Client can call separate endpoint to get attention visualization
            except Exception as e:
                print(f"Attention extraction failed: {e}")

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/attention', methods=['POST'])
def get_attention_weights():
    """
    Get attention weights and visualization data for transformer models.
    """
    try:
        if not analyzer.transformer_model:
            return jsonify({
                'error': 'Transformer model not available',
                'suggestion': 'Train a transformer model first'
            }), 400

        data = request.get_json(silent=True)
        if not data or 'code' not in data:
            return jsonify({'error': 'No code text provided'}), 400

        code_text = data['code']
        layer_idx = data.get('layer', -1)
        head_idx = data.get('head', 0)

        # Get predictions with attention
        predictions = analyzer.transformer_model.predict(
            [code_text],
            return_attention=True,
        )

        if 'attention_weights' not in predictions:
            return jsonify({
                'error': 'Attention weights not available'
            }), 500

        # Get tokenizer
        tokenizer = analyzer.transformer_model.tokenizer
        tokens = tokenizer.tokenize(code_text)[:50]  # Limit for response size

        # Extract CLS attention for visualization
        attention_weights = predictions['attention_weights'][0]  # First sample
        # attention_weights shape: (layers, heads, seq_len, seq_len)

        # Get CLS token attention
        cls_attention = attention_weights[layer_idx, :, 0, :].mean(axis=0)
        cls_attention = cls_attention[:len(tokens)].tolist()

        # Get top attended tokens
        top_k = min(20, len(tokens))
        top_indices = sorted(
            range(len(cls_attention)),
            key=lambda i: cls_attention[i],
            reverse=True,
        )[:top_k]

        top_tokens = [
            {
                'token': tokens[i],
                'position': i,
                'attention_score': cls_attention[i],
            }
            for i in top_indices
        ]

        return jsonify({
            'success': True,
            'tokens': tokens,
            'cls_attention': cls_attention,
            'top_attended_tokens': top_tokens,
            'layer': layer_idx,
            'head': head_idx,
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


@app.route('/api/compare', methods=['POST'])
def compare_strategies():
    """
    Compare different analysis strategies on the same code.
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'code' not in data:
            return jsonify({'error': 'No code text provided'}), 400

        code_text = data['code']
        strategies = ['rule_only', 'ml_only', 'weighted_average', 'ml_override']

        results = {}
        for strategy in strategies:
            try:
                analysis = analyzer.analyze_code(code_text, ensemble_strategy=strategy)
                results[strategy] = {
                    'quality_score': analysis['quality_score'],
                    'total_issues': analysis['total_issues'],
                    'model_used': analysis.get('model_used', 'unknown'),
                }
            except Exception as e:
                results[strategy] = {'error': str(e)}

        return jsonify({
            'success': True,
            'code_length': len(code_text),
            'comparison': results,
        })

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

    print(f"\nStarting Advanced Code Quality Assessment API on port {port}")
    print(f"Debug mode: {debug}\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
