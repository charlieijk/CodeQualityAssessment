# 🔍 Computer Vision Code Quality Assessment

An advanced system that uses computer vision and machine learning to analyze programming code screenshots and provide automated code quality assessment with educational feedback.

## 🌟 Features [![Python application](https://github.com/charlieijk/CodeQualityAssessment/actions/workflows/python-app.yml/badge.svg)](https://github.com/charlieijk/CodeQualityAssessment/actions/workflows/python-app.yml)

- **📸 Image OCR Processing**: Upload screenshots of code for automatic text extraction using OpenCV and Tesseract
- **🤖 ML-Powered Analysis**: Automated detection of code quality issues using custom algorithms
- **📚 Educational Feedback**: Constructive suggestions and learning paths for code improvement
- **🌐 Web Interface**: User-friendly Flask web application with drag-and-drop functionality
- **🔗 RESTful API**: Complete API for integration with other tools and services
- **⚡ Real-time Analysis**: Instant quality assessment with detailed reporting

## 🛠️ Technical Stack

- **Computer Vision**: OpenCV, Tesseract OCR, PIL/Pillow
- **Machine Learning**: TensorFlow, PyTorch, scikit-learn
- **Web Framework**: Flask, Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript
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

## 🤖 ML Training Pipeline

Weak labels are produced by the same analyzers that power the API. You can use them to bootstrap machine-learning models that generalize beyond the heuristic rules:

1. **Generate a dataset** from any Python repositories and optional screenshots:
   ```bash
   python -m src.data_pipeline.dataset_builder --code-dirs src tests --output data/processed/dataset.jsonl
   ```
   This writes JSONL samples (one per snippet) under `data/processed/` with issue-level annotations.
2. **Train the baseline quality model** (requires the optional ML dependencies from `requirements-ml-optional.txt`):
   ```bash
   python -m src.models.baseline_trainer --dataset data/processed/dataset.jsonl --model-dir data/processed/models
   ```
   The trainer reports mean absolute error for the quality score and micro-F1 for multi-label issue detection, then exports a reusable `baseline_quality_model.joblib`.
3. **Learn more**: `docs/ML_PIPELINE.md` covers environment setup, dataset schema, trainer outputs, and ideas for extending the pipeline.

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
