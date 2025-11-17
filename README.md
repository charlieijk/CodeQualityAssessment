---
title: Code Quality Assessment
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
tags:
- code-quality
- computer-vision
- ocr
- machine-learning
- education
- python
- code-analysis
- deep-learning
- gradio
language:
- en
library_name: transformers
pipeline_tag: text-classification
short_description: AI-powered code quality analysis with computer vision
---

# 🔍 Computer Vision Code Quality Assessment

An advanced system that uses computer vision and machine learning to analyze programming code screenshots and provide automated code quality assessment with educational feedback.

## 🌟 Features [![Python application](https://github.com/charlieijk/CodeQualityAssessment/actions/workflows/python-app.yml/badge.svg)](https://github.com/charlieijk/CodeQualityAssessment/actions/workflows/python-app.yml)

- **📸 Image OCR Processing**: Upload screenshots of code for automatic text extraction using OpenCV and Tesseract
- **🤖 Advanced ML Models**: State-of-the-art transformer models (CodeBERT, GraphCodeBERT) for deep code understanding
- **🎯 Multi-Task Learning**: Simultaneous quality scoring and issue detection with attention mechanisms
- **📊 Hybrid Analysis**: Combines rule-based heuristics with deep learning for robust predictions
- **📚 Educational Feedback**: Constructive suggestions and learning paths for code improvement
- **🌐 Web Interface**: User-friendly Flask web application with drag-and-drop functionality
- **🔗 RESTful API**: Complete API for integration with other tools and services
- **👁️ Attention Visualization**: Understand which code parts affect quality predictions
- **⚡ Real-time Analysis**: Instant quality assessment with detailed reporting
- **🔄 Fine-tuning Support**: Train custom models on your own code datasets

## 🛠️ Technical Stack

- **Computer Vision**: OpenCV, Tesseract OCR, PIL/Pillow
- **Deep Learning**: PyTorch, Transformers (HuggingFace)
- **Transformer Models**: CodeBERT, GraphCodeBERT for code understanding
- **Machine Learning**: scikit-learn, TensorFlow (optional)
- **Web Framework**: Flask, Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Matplotlib, Seaborn for attention heatmaps
- **Image Processing**: Advanced preprocessing pipeline for better OCR accuracy
- **Text Analysis**: NLTK, spaCy for code pattern recognition

## 📋 Prerequisites

- Python 3.8 or higher (3.10–3.11 recommended for full ML stack)
- Tesseract OCR engine
- Git
- Optional ML dependencies (PyTorch, TensorFlow, spaCy, etc.) currently publish wheels only for Python < 3.12. Install them with `pip install -r requirements-ml-optional.txt` inside a Python 3.11 virtualenv if you need those advanced features.

### Installing Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## 🚀 Quick Start

### Web Interface (Gradio)

**Try it live on Hugging Face Spaces**: [Code Quality Assessment Space](https://huggingface.co/spaces/charlieijk/code-quality-assessment)

**Run locally:**

```bash
# Basic interface (code analysis only)
pip install -r requirements.txt
python gradio_app.py

# Advanced interface (with model parameters & fine-tuning)
pip install -r requirements-ml-optional.txt
python gradio_app_advanced.py

# Smart launcher (auto-detects available features)
python launch_gradio.py
```

Open your browser: http://localhost:7860

### Flask API

1. **Clone the repository:**
```bash
git clone <repository-url>
cd CodeQualityAssessment
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
   - (Optional) Install developer tools with `pip install -r requirements-dev.txt` if you have network access.
   - (Optional) Install the heavy ML stack with `pip install -r requirements-ml-optional.txt` when using Python 3.11.

4. **Run the application:**
```bash
python run.py
```

5. **Open your browser and navigate to:**
```
http://localhost:5000
```

## 🧠 Running in Jupyter Notebook

1. **Launch Jupyter from the project root** so the relative imports keep working:
   ```bash
   cd CodeQualityAssessment
   jupyter lab          # or: jupyter notebook
   ```
2. **Install dependencies inside the notebook kernel** (Tesseract still needs to be installed system-wide via Homebrew/apt/Windows installer):
   ```python
   %pip install -r requirements.txt
   ```
3. **Open `run.ipynb` and execute the server cell** (the one containing the former `run.py` entry point). Keep that cell running while you test; stop it via the red ■ button or `Kernel → Interrupt` when you want to shut the server down.
4. **Optional background run:** if you need the server detached from the main kernel, convert the notebook once and launch it in a background bash cell:
   ```python
   !jupyter nbconvert --to python run.ipynb

   %%bash --bg
   source venv/bin/activate
   python run.py --host 127.0.0.1 --port 5000
   ```
   (Use the printed URL to access the app, and stop the background job with `jobs`/`kill` in the same notebook.)

## 📖 Usage

### Web Interface

1. **Upload Code Screenshot:**
   - Drag and drop an image file containing code
   - Or click "Choose Image File" to select manually
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF

2. **Direct Text Analysis:**
   - Paste your code directly into the text area
   - Click "Analyze Code Text" for instant analysis

3. **View Results:**
   - Quality score (0-100)
   - Detailed issue breakdown
   - Educational feedback and suggestions
   - Learning path recommendations

### API Endpoints

#### Health Check
```bash
GET /api/health
```

#### Analyze Code Image
```bash
POST /api/analyze
Content-Type: multipart/form-data

# Upload image file
curl -X POST -F "file=@code_screenshot.png" http://localhost:5000/api/analyze
```

#### Analyze Code Text
```bash
POST /api/analyze-text
Content-Type: application/json

{
    "code": "def hello():\n    print('Hello, World!')"
}
```

#### Get Contextual Feedback
```bash
GET /api/feedback/{quality_score}

# Example
curl http://localhost:5000/api/feedback/85.5
```

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)

### Command Line Options

```bash
python run.py --help

Options:
  --port PORT     Port to run the server on (default: 5000)
  --host HOST     Host to bind the server to (default: 127.0.0.1)
  --debug         Run in debug mode
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 🤖 Advanced ML Models

This system supports three levels of code quality analysis:

### 1. Rule-Based Analysis (Default)
Fast, lightweight analysis using heuristics and AST parsing. No ML dependencies required.

### 2. Baseline ML Model
TF-IDF + scikit-learn models for improved accuracy.

**Training:**
```bash
# Generate dataset
python -m src.data_pipeline.dataset_builder --code-dirs src tests --output data/processed/dataset.jsonl

# Train baseline model
python -m src.models.baseline_trainer --dataset data/processed/dataset.jsonl --model-dir data/processed/models
```

### 3. Transformer Models (CodeBERT/GraphCodeBERT) 🚀

State-of-the-art deep learning models for code understanding with attention mechanisms.

**Features:**
- **Multi-task learning**: Simultaneous quality scoring + issue detection
- **Attention visualization**: See which code parts affect predictions
- **Transfer learning**: Fine-tune on your specific codebase
- **Ensemble strategies**: Combine with rule-based analysis

**Training:**
```bash
# Install transformer dependencies (Python 3.11 recommended)
pip install -r requirements-ml-optional.txt

# Generate training dataset
python -m src.data_pipeline.dataset_builder \
  --code-dirs src tests \
  --output data/processed/dataset.jsonl \
  --limit 1000

# Train transformer model
python -m src.models.train_transformer \
  --dataset data/processed/dataset.jsonl \
  --model-name microsoft/codebert-base \
  --output-dir data/processed/transformer_models \
  --num-epochs 10 \
  --batch-size 16
```

**Available Models:**
- `microsoft/codebert-base` - General purpose code model
- `microsoft/graphcodebert-base` - Graph-aware code model
- `huggingface/CodeBERTa-small-v1` - Smaller, faster variant

**Using Advanced API:**
```bash
# Run with ML support
python src/api/app_advanced.py
```

**API Strategies:**
- `rule_only`: Traditional heuristic analysis
- `ml_only`: Pure ML predictions
- `weighted_average`: Combine rule-based (30%) + ML (70%)
- `ml_override`: ML score with detailed rule-based issues

**Example API Request:**
```bash
curl -X POST http://localhost:5000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello():\n    print(\"Hello\")",
    "strategy": "weighted_average",
    "return_attention": true
  }'
```

**Attention Visualization:**
```python
from src.utils.attention_viz import AttentionVisualizer
from src.models.transformer_model import TransformerQualityTrainer

# Load model
trainer = TransformerQualityTrainer.load("data/processed/transformer_models")

# Get predictions with attention
predictions = trainer.predict(["def hello():\n    print('hi')"], return_attention=True)

# Visualize
viz = AttentionVisualizer(trainer.tokenizer)
viz.visualize_cls_attention(
    "def hello():\n    print('hi')",
    predictions['attention_weights'][0],
    save_path="attention.png"
)
```

**Model Parameter Analysis:**
```bash
# Analyze base transformer model parameters
python scripts/analyze_model_params.py --model microsoft/codebert-base

# Analyze full multi-task model with custom heads
python scripts/analyze_model_params.py --model microsoft/codebert-base --with-heads --num-issues 15

# Compare different architectures
python scripts/analyze_model_params.py --compare microsoft/codebert-base microsoft/graphcodebert-base
```

**Learn more**:
- `docs/ML_PIPELINE.md` covers dataset schema, training details, and extending the pipeline
- `docs/MODEL_PARAMETERS.md` explains parameter analysis and model optimization strategies

## 📁 Project Structure

```
CodeQualityAssessment/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py              # Flask application
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── image_preprocessor.py # OCR and image processing
│   ├── models/
│   │   ├── __init__.py
│   │   └── code_analyzer.py     # Code quality analysis
│   └── utils/
│       ├── __init__.py
│       └── feedback_generator.py # Educational feedback
├── tests/
│   ├── test_code_analyzer.py    # Analyzer tests
│   └── test_api.py             # API tests
├── templates/
│   └── index.html              # Web interface
├── static/                     # Static files (CSS, JS)
├── data/
│   ├── raw/                    # Uploaded images
│   └── processed/              # Processed data
├── requirements.txt            # Python dependencies
├── run.py                     # Application runner
└── README.md                  # This file
```

## 🎯 Code Quality Metrics

The system analyzes code for:

- **Syntax Errors**: Python syntax validation
- **Naming Conventions**: PEP 8 compliance
- **Code Structure**: Line length, nesting depth
- **Documentation**: Missing docstrings
- **Code Smells**: Magic numbers, empty exception blocks
- **Best Practices**: General programming principles

### Quality Scoring

- **90-100**: Excellent (🟢)
- **75-89**: Good (🔵)
- **60-74**: Fair (🟡)
- **40-59**: Poor (🟠)
- **0-39**: Critical (🔴)

## 🎓 Educational Features

- **Issue-Specific Feedback**: Detailed explanations for each type of problem
- **Improvement Suggestions**: Actionable advice for fixing issues
- **Learning Paths**: Personalized recommendations based on detected issues
- **Resource Links**: References to documentation and best practices
- **Progress Tracking**: Quality score evolution over time

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- Support for multiple programming languages
- Advanced ML models for more sophisticated analysis
- Integration with popular IDEs
- Batch processing capabilities
- User authentication and progress tracking
- Advanced metrics and analytics
- Mobile app support

## 🆘 Troubleshooting

### Common Issues

1. **Tesseract not found error:**
   - Ensure Tesseract is installed and in your PATH
   - Set `TESSDATA_PREFIX` environment variable if needed

2. **OCR accuracy issues:**
   - Use high-quality images with good contrast
   - Ensure text is clearly visible and not too small
   - Avoid cluttered backgrounds

3. **Performance issues:**
   - Large images may take longer to process
   - Consider resizing images before upload
   - Check available system memory

### Getting Help

- Check the Issues section on GitHub
- Review the documentation
- Contact the maintainers

---

Built with ❤️ for better code quality and learning
