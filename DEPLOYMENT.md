# Deploying to Hugging Face

This guide explains how to run your Code Quality Assessment model on Hugging Face.

## Option 1: Hugging Face Spaces (Web App Deployment) 🚀

Deploy your Flask web application directly on Hugging Face infrastructure.

### Steps:

1. **Create a Hugging Face Space:**
   - Go to https://huggingface.co/new-space
   - Name: `code-quality-assessment`
   - SDK: Select **Docker**
   - Hardware: Select **CPU basic** (free) or **GPU** for transformer models
   - Visibility: Public or Private

2. **The files are already prepared:**
   - `Dockerfile` - Docker configuration
   - `app.py` - Entry point for Spaces
   - `requirements-spaces.txt` - Lightweight dependencies
   - `.spacesconfig.yaml` - Spaces metadata

3. **Push to your Space:**
   ```bash
   # Add the Space as a remote (replace USERNAME with your HF username)
   git remote add space https://huggingface.co/spaces/charlieijk/code-quality-assessment

   # Push to the Space
   git push space master:main
   ```

4. **Your Space will automatically build and deploy!**
   - URL: `https://huggingface.co/spaces/charlieijk/code-quality-assessment`
   - It will install dependencies, build the Docker image, and start the app

### For Advanced ML Models on Spaces:

If you want to use transformer models on Spaces:

1. **Use GPU hardware** (select in Space settings)
2. **Modify `requirements-spaces.txt`** to include:
   ```bash
   torch>=2.0.0
   transformers>=4.30.0
   ```
3. **Pre-download models** in Dockerfile:
   ```dockerfile
   RUN python -c "from transformers import AutoModel, AutoTokenizer; \
       AutoModel.from_pretrained('microsoft/codebert-base'); \
       AutoTokenizer.from_pretrained('microsoft/codebert-base')"
   ```

---

## Option 2: Model Hub (For Trained Models) 🤖

Upload your trained transformer models to the Hugging Face Model Hub.

### Steps:

1. **Your model is already on the Hub:**
   - Repository: https://huggingface.co/charlieijk/CodeQualityAssessment

2. **To use the model from the Hub:**

   ```python
   from transformers import AutoModel, AutoTokenizer
   import torch

   # Load from Hugging Face Hub
   model_name = "charlieijk/CodeQualityAssessment"

   # Note: You'll need to upload the actual trained model files first
   # See "Uploading Trained Models" below
   ```

3. **Uploading Trained Models:**

   After training your transformer model:

   ```bash
   # Install huggingface_hub
   pip install huggingface_hub

   # Login
   huggingface-cli login

   # Upload model files
   python -c "
   from huggingface_hub import HfApi
   api = HfApi()

   api.upload_folder(
       folder_path='data/processed/transformer_models',
       repo_id='charlieijk/CodeQualityAssessment',
       repo_type='model',
       path_in_repo='models/transformer'
   )
   "
   ```

4. **Create a Model Card:**

   Add model information to your README.md on the Hub with model card metadata:

   ```yaml
   ---
   license: mit
   datasets:
   - code-quality-dataset
   metrics:
   - accuracy
   - f1
   model-index:
   - name: CodeQualityAssessment-Transformer
     results:
     - task:
         type: text-classification
       metrics:
       - type: f1
         value: 0.85
   ---
   ```

---

## Option 3: Inference API 🔌

Use Hugging Face's Inference API to run models without deploying.

### Steps:

1. **Upload your model to the Hub** (see Option 2)

2. **Use the Inference API:**

   ```python
   import requests

   API_URL = "https://api-inference.huggingface.co/models/charlieijk/CodeQualityAssessment"
   headers = {"Authorization": f"Bearer {YOUR_HF_TOKEN}"}

   def query(payload):
       response = requests.post(API_URL, headers=headers, json=payload)
       return response.json()

   output = query({
       "inputs": "def hello():\n    print('Hello World')",
   })
   print(output)
   ```

3. **Note:** The Inference API requires your model to follow standard HuggingFace formats.

---

## Option 4: Gradio App (Interactive Demo) 🎨

Create an interactive Gradio interface on Hugging Face Spaces.

### Create `gradio_app.py`:

```python
import gradio as gr
from src.models.code_analyzer import CodeQualityAnalyzer
from src.utils.feedback_generator import FeedbackGenerator

analyzer = CodeQualityAnalyzer()
feedback_gen = FeedbackGenerator()

def analyze_code(code_text):
    if not code_text.strip():
        return "Please enter some code to analyze."

    # Analyze code
    result = analyzer.analyze_code(code_text)
    feedback = feedback_gen.generate_feedback(result)

    # Format output
    output = f"""
    ## Quality Score: {result['quality_score']}/100

    ### Issues Found: {result['total_issues']}
    - High Severity: {result['severity_breakdown']['high']}
    - Medium Severity: {result['severity_breakdown']['medium']}
    - Low Severity: {result['severity_breakdown']['low']}

    ### Feedback:
    {feedback.get('overall_assessment', '')}

    ### Suggestions:
    """

    for suggestion in feedback.get('suggestions', [])[:5]:
        output += f"\n- {suggestion}"

    return output

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_code,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your Python code here...",
        label="Code Input"
    ),
    outputs=gr.Markdown(label="Analysis Results"),
    title="🔍 Code Quality Assessment",
    description="AI-powered code quality analysis. Paste your Python code to get instant feedback!",
    examples=[
        ["def hello():\n    print('Hello World')"],
        ["class MyClass:\n    def __init__(self):\n        pass"],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    iface.launch()
```

Then:
1. Replace `app.py` with `gradio_app.py`
2. Update `requirements-spaces.txt` to include `gradio`
3. Push to your Space

---

## Recommended Approach

**For your use case, I recommend:**

### Starting with Gradio on Spaces (Easiest):
- ✅ Quick setup
- ✅ Beautiful UI out of the box
- ✅ Free hosting
- ✅ Automatic GPU support
- ✅ Easy sharing

### Steps to Deploy with Gradio:

```bash
# 1. Install gradio locally to test
pip install gradio

# 2. Create and test the gradio app
python gradio_app.py

# 3. Commit the changes
git add gradio_app.py requirements-spaces.txt
git commit -m "Add Gradio interface for Hugging Face Spaces"

# 4. Create a Space at https://huggingface.co/new-space
# Select: SDK = Gradio

# 5. Push to your Space
git remote add space https://huggingface.co/spaces/charlieijk/code-quality-assessment
git push space master:main
```

---

## Testing Locally Before Deployment

### Test Docker build locally:
```bash
# Build the Docker image
docker build -t code-quality-app .

# Run the container
docker run -p 7860:7860 code-quality-app

# Visit http://localhost:7860
```

### Test Gradio locally:
```bash
python gradio_app.py
```

---

## Cost & Hardware Options

| Hardware | Cost | Use Case |
|----------|------|----------|
| CPU basic | FREE | Rule-based analysis, lightweight |
| CPU upgrade | $0.03/hr | Faster processing |
| T4 Small GPU | $0.60/hr | Transformer models |
| A10G Small GPU | $1.05/hr | Large transformer models |

For rule-based analysis only: **FREE CPU basic is sufficient**
For transformer models: **Requires GPU** (T4 Small minimum)

---

## Troubleshooting

### Space fails to build:
- Check logs in Space settings
- Verify all dependencies are in requirements-spaces.txt
- Ensure Tesseract is installed in Dockerfile

### Out of memory:
- Reduce batch size in model configs
- Use CPU basic if only using rule-based analysis
- Upgrade to GPU hardware for transformer models

### Model not loading:
- Ensure model files are in the repository
- Check file paths in code
- Verify model files aren't in .gitignore

---

## Next Steps

1. Create a Hugging Face Space
2. Choose SDK (Gradio recommended)
3. Push your code
4. Share your Space URL!

Your Space will be live at:
`https://huggingface.co/spaces/charlieijk/code-quality-assessment`
