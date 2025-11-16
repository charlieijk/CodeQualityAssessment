"""
Hugging Face Spaces entry point.
This file is automatically detected and run by Hugging Face Spaces.
"""
import os
import sys
from pathlib import Path

# Set environment variables for Hugging Face Spaces
os.environ.setdefault('PORT', '7860')  # Hugging Face Spaces default port

# Import and run the application
from src.api.app import app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))
    print(f"Starting Code Quality Assessment on Hugging Face Spaces (port {port})")
    app.run(host='0.0.0.0', port=port)
