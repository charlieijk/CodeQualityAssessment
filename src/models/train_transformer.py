"""
Fine-tuning script for transformer-based code quality models.

This script trains CodeBERT or other transformer models on code quality datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.transformer_model import (
        TransformerConfig,
        TransformerQualityTrainer,
    )
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(
        "Error: Missing dependencies. Install requirements-ml-optional.txt "
        "inside a Python 3.11 virtualenv."
    )
    raise exc


def load_dataset(dataset_path: Path) -> tuple[List[str], np.ndarray, List[List[str]]]:
    """
    Load dataset from JSONL file.

    Args:
        dataset_path: Path to JSONL dataset file

    Returns:
        Tuple of (texts, quality_scores, issue_labels)
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    texts = []
    quality_scores = []
    issue_labels = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            texts.append(entry.get("text", ""))
            quality_scores.append(float(entry.get("quality_score", 0.0)))

            # Extract issue types
            issues = entry.get("issues", [])
            issue_types = [
                issue.get("issue_type") for issue in issues if issue.get("issue_type")
            ]
            issue_labels.append(issue_types)

    if not texts:
        raise ValueError("No data found in dataset file")

    return texts, np.array(quality_scores), issue_labels


def plot_training_history(history: Dict[str, List[float]], save_dir: Path) -> None:
    """
    Plot and save training metrics.

    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot losses
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE plot
    axes[1].plot(history["train_quality_mae"], label="Train MAE", marker="o")
    axes[1].plot(history["val_quality_mae"], label="Val MAE", marker="s")
    axes[1].set_title("Quality Score MAE", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 plot
    axes[2].plot(history["train_issue_f1"], label="Train F1", marker="o")
    axes[2].plot(history["val_issue_f1"], label="Val F1", marker="s")
    axes[2].set_title("Issue Classification F1 Score", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / "training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nTraining plots saved to {plot_path}")
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer models for code quality assessment"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset file",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/codebert-base",
        choices=[
            "microsoft/codebert-base",
            "microsoft/graphcodebert-base",
            "huggingface/CodeBERTa-small-v1",
            "neulab/codebert-javascript",
        ],
        help="Pre-trained model to use (default: microsoft/codebert-base)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/transformer_models",
        help="Directory to save trained model (default: data/processed/transformer_models)",
    )
    parser.add_argument(
        "--issue-threshold",
        type=float,
        default=0.5,
        help="Threshold for issue classification (default: 0.5)",
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("TRANSFORMER MODEL TRAINING")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Max length: {args.max_length}")
    print(f"Output directory: {args.output_dir}")
    print("\n" + "=" * 80 + "\n")

    # Load dataset
    print("Loading dataset...")
    dataset_path = Path(args.dataset)
    texts, quality_scores, issue_labels = load_dataset(dataset_path)
    print(f"Loaded {len(texts)} samples")
    print(f"Quality scores range: [{quality_scores.min():.2f}, {quality_scores.max():.2f}]")

    # Count unique issue types
    unique_issues = set()
    for issues in issue_labels:
        unique_issues.update(issues)
    print(f"Unique issue types: {len(unique_issues)}")
    print(f"Issue types: {sorted(unique_issues)}\n")

    # Create configuration
    config = TransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        dropout_rate=args.dropout_rate,
        issue_threshold=args.issue_threshold,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = TransformerQualityTrainer(config)

    # Prepare data
    print("Preparing data loaders...")
    train_loader, val_loader = trainer.prepare_data(
        texts,
        quality_scores,
        issue_labels,
        test_size=args.test_size,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    # Train model
    print("Starting training...")
    print("-" * 80)
    history = trainer.train(train_loader, val_loader)

    # Save model
    output_dir = Path(args.output_dir)
    print(f"\nSaving model to {output_dir}...")
    model_path = trainer.save(output_dir)
    print(f"Model saved successfully to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Plot training curves
    print("\nGenerating training plots...")
    plot_training_history(history, output_dir)

    # Final metrics
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation MAE: {history['val_quality_mae'][-1]:.4f}")
    print(f"Final validation F1: {history['val_issue_f1'][-1]:.4f}")
    print("=" * 80)

    # Save final metrics
    final_metrics = {
        "dataset": str(dataset_path),
        "model_name": args.model_name,
        "num_samples": len(texts),
        "num_train": len(train_loader.dataset),
        "num_val": len(val_loader.dataset),
        "final_val_loss": history["val_loss"][-1],
        "final_val_mae": history["val_quality_mae"][-1],
        "final_val_f1": history["val_issue_f1"][-1],
        "best_val_loss": min(history["val_loss"]),
        "config": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
        },
    }

    metrics_path = output_dir / "final_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nFinal metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
