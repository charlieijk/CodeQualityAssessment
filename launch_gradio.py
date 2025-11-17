#!/usr/bin/env python3
"""
Smart launcher for Gradio interface.
Automatically detects available dependencies and launches the appropriate version.
"""
import sys
import importlib.util

def check_ml_available():
    """Check if ML dependencies are available."""
    required_packages = ['torch', 'transformers']
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            return False
    return True

def main():
    """Launch appropriate Gradio interface based on available dependencies."""

    ml_available = check_ml_available()

    if ml_available:
        print("✅ ML dependencies detected - launching advanced interface with model parameters and fine-tuning")
        print("📊 Features: Code Analysis + Model Parameters + Fine-Tuning")
        print("")
        from gradio_app_advanced import demo
    else:
        print("ℹ️  ML dependencies not available - launching basic interface")
        print("📊 Features: Code Analysis")
        print("")
        print("💡 To enable advanced features (model parameters, fine-tuning), install:")
        print("   pip install -r requirements-ml-optional.txt")
        print("")
        from gradio_app import demo

    # Launch the demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

if __name__ == "__main__":
    main()
