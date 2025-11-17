#!/usr/bin/env python3
"""
Script to create and configure a Hugging Face Space for Code Quality Assessment.
"""
import sys
from huggingface_hub import HfApi, create_repo, upload_folder, get_token
from pathlib import Path

def create_hf_space():
    """Create and configure the Hugging Face Space."""

    # Initialize API
    api = HfApi()
    token = get_token()

    if not token:
        print("❌ Error: No Hugging Face token found.")
        print("Please login first with: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        return False

    # Get username
    try:
        user_info = api.whoami(token=token)
        username = user_info['name']
        print(f"✅ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Error getting user info: {e}")
        return False

    # Space configuration
    space_name = "code-quality-assessment"
    repo_id = f"{username}/{space_name}"

    print(f"\n📦 Creating Space: {repo_id}")

    try:
        # Create the Space
        url = create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        print(f"✅ Space created/found: {url}")

    except Exception as e:
        print(f"❌ Error creating Space: {e}")
        return False

    # Files to upload
    print(f"\n📤 Preparing to upload files...")

    # First, let's create a proper app.py that uses gradio_app
    app_content = '''"""
Hugging Face Spaces entry point.
This file is automatically detected and run by Hugging Face Spaces.
"""
if __name__ == "__main__":
    from gradio_app import demo
    demo.launch()
'''

    with open("app.py", "w") as f:
        f.write(app_content)

    print("✅ Updated app.py for Gradio")

    try:
        # Upload the entire folder
        print(f"📤 Uploading files to Space...")

        # Define patterns to ignore
        ignore_patterns = [
            ".git/*",
            ".github/*",
            "__pycache__/*",
            "*.pyc",
            ".env",
            "venv/*",
            ".venv/*",
            "data/raw/*",
            "data/processed/models/*",
            "*.ipynb",
            ".pytest_cache/*",
            "tests/*",
            "*.log",
            "create_space.py",  # Don't upload this script itself
        ]

        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            token=token,
            ignore_patterns=ignore_patterns,
            commit_message="🚀 Deploy Code Quality Assessment to Hugging Face Spaces",
        )

        print(f"\n✅ Successfully uploaded to Hugging Face Space!")
        print(f"🌐 Space URL: https://huggingface.co/spaces/{repo_id}")
        print(f"\n⏳ The Space is building... This may take a few minutes.")
        print(f"   Visit the URL above to check the build status and see your app!")

        return True

    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_hf_space()
    sys.exit(0 if success else 1)
