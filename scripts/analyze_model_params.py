#!/usr/bin/env python3
"""
Analyze and display parameter tables for transformer models.

This script provides detailed parameter analysis for:
1. Pre-trained transformer models (CodeBERT, GraphCodeBERT, etc.)
2. The multi-task code quality model with custom heads

Usage:
    # Analyze base CodeBERT model
    python scripts/analyze_model_params.py --model microsoft/codebert-base

    # Analyze the full multi-task model
    python scripts/analyze_model_params.py --model microsoft/codebert-base --with-heads --num-issues 10

    # Analyze a saved model
    python scripts/analyze_model_params.py --load data/processed/transformer_models

    # Save analysis to file
    python scripts/analyze_model_params.py --model microsoft/codebert-base --output model_params.json
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from src.models.transformer_model import (
        MultiTaskCodeQualityModel,
        TransformerConfig,
        TransformerQualityTrainer,
    )
    from src.utils.model_analyzer import (
        ModelParameterAnalyzer,
        analyze_codebert_model,
    )
except ImportError as exc:
    print(f"Error: {exc}")
    print("\nPlease install ML dependencies:")
    print("  pip install -r requirements-ml-optional.txt")
    sys.exit(1)


def analyze_base_transformer(model_name: str, output_path: Path = None):
    """Analyze a base transformer model (CodeBERT, etc.)."""
    print(f"\n{'='*80}")
    print(f"ANALYZING BASE TRANSFORMER MODEL: {model_name}")
    print(f"{'='*80}\n")

    summary = analyze_codebert_model(model_name)
    analyzer = ModelParameterAnalyzer()
    analyzer.print_summary_table(summary, show_layers=False)

    if output_path:
        analyzer.save_summary(summary, output_path)


def analyze_multitask_model(model_name: str, num_issue_types: int, output_path: Path = None):
    """Analyze the full multi-task code quality model."""
    print(f"\n{'='*80}")
    print(f"ANALYZING MULTI-TASK CODE QUALITY MODEL")
    print(f"Base Model: {model_name}")
    print(f"Number of Issue Types: {num_issue_types}")
    print(f"{'='*80}\n")

    config = TransformerConfig(model_name=model_name)

    print("Loading model components...")
    model = MultiTaskCodeQualityModel(config, num_issue_types=num_issue_types)

    # Print custom summary
    model.print_parameter_summary()

    # Detailed analysis
    analyzer = ModelParameterAnalyzer()
    summary = analyzer.analyze_model(model, detailed=True)
    analyzer.print_summary_table(summary, show_layers=False)

    if output_path:
        analyzer.save_summary(summary, output_path)


def analyze_saved_model(model_dir: Path, output_path: Path = None):
    """Analyze a saved trained model."""
    print(f"\n{'='*80}")
    print(f"ANALYZING SAVED MODEL FROM: {model_dir}")
    print(f"{'='*80}\n")

    try:
        trainer = TransformerQualityTrainer.load(model_dir)
        print(f"Model loaded successfully!")
        print(f"Model type: {trainer.config.model_name}")

        if trainer.model:
            trainer.model.print_parameter_summary()

            analyzer = ModelParameterAnalyzer()
            summary = analyzer.analyze_model(trainer.model, detailed=True)
            analyzer.print_summary_table(summary, show_layers=False)

            if output_path:
                analyzer.save_summary(summary, output_path)
        else:
            print("Error: Model not properly loaded.")

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def compare_models(models: list[str]):
    """Compare parameter counts across multiple models."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}\n")

    results = []
    for model_name in models:
        print(f"Loading {model_name}...")
        summary = analyze_codebert_model(model_name)
        results.append((model_name, summary))

    # Print comparison table
    print(f"\n{'Model Name':<50} {'Total Params':>15} {'Size (MB)':>12}")
    print("-"*80)

    for model_name, summary in results:
        params_str = ModelParameterAnalyzer.format_number(summary.total_params)
        print(f"{model_name:<50} {params_str:>15} {summary.total_size_mb:>12.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze transformer model parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CodeBERT base model
  python scripts/analyze_model_params.py --model microsoft/codebert-base

  # Analyze full multi-task model with custom heads
  python scripts/analyze_model_params.py --model microsoft/codebert-base --with-heads --num-issues 15

  # Analyze saved trained model
  python scripts/analyze_model_params.py --load data/processed/transformer_models

  # Compare multiple models
  python scripts/analyze_model_params.py --compare microsoft/codebert-base microsoft/graphcodebert-base

  # Save analysis to JSON
  python scripts/analyze_model_params.py --model microsoft/codebert-base --output analysis.json
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace model name (default: microsoft/codebert-base)",
    )

    parser.add_argument(
        "--with-heads",
        action="store_true",
        help="Analyze full model including quality and issue classification heads",
    )

    parser.add_argument(
        "--num-issues",
        type=int,
        default=10,
        help="Number of issue types for multi-task model (default: 10)",
    )

    parser.add_argument(
        "--load",
        type=Path,
        help="Load and analyze a saved model from directory",
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple models",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save parameter analysis to JSON file",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed layer-by-layer analysis",
    )

    args = parser.parse_args()

    # Handle different modes
    if args.compare:
        compare_models(args.compare)
    elif args.load:
        analyze_saved_model(args.load, args.output)
    elif args.with_heads:
        analyze_multitask_model(args.model, args.num_issues, args.output)
    else:
        analyze_base_transformer(args.model, args.output)


if __name__ == "__main__":
    main()
