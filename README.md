# 🔍 Computer Vision Code Quality Assessment

An advanced system that uses computer vision and machine learning to analyze programming code screenshots and provide automated code quality assessment with educational feedback.

## 🌟 Features

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

- Python 3.8 or higher
- Tesseract OCR engine
- Git

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

4. **Run the application:**
```bash
python run.py
```

5. **Open your browser and navigate to:**
```
http://localhost:5000
```

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