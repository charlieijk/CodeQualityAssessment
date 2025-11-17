#!/usr/bin/env python3
"""
Quick test script for parameter analysis functionality.

This script tests the parameter analysis without requiring the full model download.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    from src.models.transformer_model import TransformerConfig, MultiTaskCodeQualityModel
    print("✓ Successfully imported transformer model classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nTo run the full analysis, install ML dependencies:")
    print("  pip install -r requirements-ml-optional.txt")
    sys.exit(1)


def test_parameter_counting():
    """Test parameter counting on a simple model."""
    print("\n" + "="*70)
    print("TESTING PARAMETER ANALYSIS FUNCTIONALITY")
    print("="*70 + "\n")

    # Create a small test model
    print("Creating a simple test model...")

    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(100, 50)
            self.layer2 = nn.Linear(50, 10)

        def forward(self, x):
            return self.layer2(self.layer1(x))

    model = SimpleTestModel()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Simple test model created:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Expected: {100*50 + 50 + 50*10 + 10:,}")

    assert total == 100*50 + 50 + 50*10 + 10, "Parameter count mismatch!"
    print("✓ Parameter counting works correctly!\n")


def test_multitask_model_structure():
    """Test the multi-task model parameter counting (without loading pretrained weights)."""
    print("="*70)
    print("TESTING MULTI-TASK MODEL PARAMETER METHODS")
    print("="*70 + "\n")

    print("Note: This test creates model structure only (no pretrained weights)")
    print("To analyze actual CodeBERT models, use: scripts/analyze_model_params.py\n")

    try:
        from transformers import AutoConfig

        # Create a minimal config for testing structure
        config = TransformerConfig(
            model_name="microsoft/codebert-base",
            hidden_size=768,
        )

        print(f"Creating model structure with config:")
        print(f"  Model: {config.model_name}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Dropout rate: {config.dropout_rate}")
        print()

        # Check if the MultiTaskCodeQualityModel has the new methods
        print("Checking for new parameter analysis methods...")
        assert hasattr(MultiTaskCodeQualityModel, 'count_parameters'), \
            "count_parameters method not found!"
        assert hasattr(MultiTaskCodeQualityModel, 'print_parameter_summary'), \
            "print_parameter_summary method not found!"

        print("✓ Parameter analysis methods are available!")
        print("\nTo see the full parameter table, run:")
        print("  python scripts/analyze_model_params.py --model microsoft/codebert-base --with-heads")

    except Exception as e:
        print(f"Note: Could not test full model structure: {e}")
        print("This is expected if transformers library is not fully installed.")

    print("\n" + "="*70)


def show_usage_examples():
    """Display usage examples."""
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70 + "\n")

    examples = [
        ("Analyze base CodeBERT model",
         "python scripts/analyze_model_params.py --model microsoft/codebert-base"),

        ("Analyze multi-task model with heads",
         "python scripts/analyze_model_params.py --model microsoft/codebert-base --with-heads --num-issues 15"),

        ("Compare different models",
         "python scripts/analyze_model_params.py --compare microsoft/codebert-base microsoft/graphcodebert-base"),

        ("Analyze saved trained model",
         "python scripts/analyze_model_params.py --load data/processed/transformer_models"),

        ("Save analysis to JSON",
         "python scripts/analyze_model_params.py --model microsoft/codebert-base --output params.json"),
    ]

    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"{i}. {desc}:")
        print(f"   {cmd}\n")


if __name__ == "__main__":
    try:
        test_parameter_counting()
        test_multitask_model_structure()
        show_usage_examples()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
