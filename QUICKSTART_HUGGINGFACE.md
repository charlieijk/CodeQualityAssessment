# 🚀 Quick Start: Deploy to Hugging Face Spaces

## Option 1: Create a Gradio Space (Recommended - Easiest!)

### Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `code-quality-assessment`
   - **License**: MIT
   - **Select the SDK**: **Gradio**
   - **Hardware**: **CPU basic** (FREE)
   - **Visibility**: Public

3. Click **Create Space**

### Step 2: Push Your Code

```bash
# Add your Space as a remote
git remote add space https://huggingface.co/spaces/charlieijk/code-quality-assessment

# Push your code
git push space master:main
```

### Step 3: Wait for Build

Your Space will automatically:
1. Install dependencies from `requirements.txt`
2. Launch `gradio_app.py`
3. Become available at: `https://huggingface.co/spaces/charlieijk/code-quality-assessment`

**That's it!** 🎉 Your app is now live!

---

## Option 2: Use Docker SDK (For Custom Setup)

If you need more control (e.g., system dependencies like Tesseract):

### Step 1: Create a Docker Space

1. Go to https://huggingface.co/new-space
2. Select **SDK**: **Docker**
3. Hardware: **CPU basic** (FREE)

### Step 2: Push Your Code

```bash
git remote add space https://huggingface.co/spaces/charlieijk/code-quality-assessment
git push space master:main
```

The `Dockerfile` will automatically be used to build your Space.

---

## Quick Test Locally

Before deploying, test the Gradio app locally:

```bash
# Install dependencies
pip install -r requirements.txt
pip install gradio

# Run the app
python gradio_app.py
```

Visit http://localhost:7860 to test!

---

## What Gets Deployed

### With Gradio SDK (Recommended):
- ✅ Beautiful interactive UI
- ✅ Code editor with syntax highlighting
- ✅ Real-time analysis
- ✅ Example code snippets
- ✅ Automatic deployment from `gradio_app.py`

### Files Used:
- `gradio_app.py` - Main application
- `requirements.txt` - Python dependencies
- `src/` - Source code modules

---

## Troubleshooting

### "Application startup failed"
- Check logs in your Space's "Logs" tab
- Ensure all dependencies are in `requirements.txt`
- Test locally first with `python gradio_app.py`

### "Module not found"
- Make sure `src/` directory is included in the repository
- Check that `requirements.txt` includes all necessary packages

### Out of memory
- Use CPU basic hardware (rule-based analysis only)
- Don't load ML models on free tier
- For transformer models, upgrade to GPU hardware

---

## Cost

| Setup | Cost |
|-------|------|
| Gradio on CPU basic | **FREE** ✅ |
| Docker on CPU basic | **FREE** ✅ |
| With GPU for ML models | ~$0.60/hr |

**Recommendation**: Start with FREE Gradio on CPU basic!

---

## Your Space URL

Once deployed, share your Space at:

**https://huggingface.co/spaces/charlieijk/code-quality-assessment**

---

## Next Steps

1. **Customize the UI**: Edit `gradio_app.py` to add features
2. **Add more examples**: Update the example code snippets
3. **Enable ML models**: Upgrade to GPU hardware and enable transformer models
4. **Share your Space**: Get feedback and improve!

Happy deploying! 🚀
