"""
Model architecture and parameter analysis utilities.

This module provides tools for analyzing transformer model architectures,
counting parameters, and generating detailed parameter tables.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "PyTorch and transformers are required for model analysis. "
        "Install requirements-ml-optional.txt inside a Python 3.11 virtualenv."
    ) from exc


@dataclass
class LayerInfo:
    """Information about a model layer."""
    name: str
    type: str
    num_params: int
    trainable_params: int
    frozen_params: int
    shape: str
    dtype: str


@dataclass
class ModelParameterSummary:
    """Complete summary of model parameters."""
    total_params: int
    trainable_params: int
    frozen_params: int
    total_size_mb: float
    layers: List[LayerInfo]
    param_breakdown: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "total_size_mb": self.total_size_mb,
            "param_breakdown": self.param_breakdown,
            "num_layers": len(self.layers),
        }


class ModelParameterAnalyzer:
    """Analyzes and generates parameter tables for PyTorch models."""

    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        Count total and trainable parameters in a model.

        Args:
            model: PyTorch model

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    @staticmethod
    def get_parameter_size_mb(model: nn.Module) -> float:
        """
        Calculate model size in megabytes.

        Args:
            model: PyTorch model

        Returns:
            Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    @staticmethod
    def analyze_layer(name: str, module: nn.Module) -> LayerInfo:
        """
        Analyze a single layer/module.

        Args:
            name: Layer name
            module: PyTorch module

        Returns:
            LayerInfo object with layer details
        """
        total_params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable_params = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        # Get shape information
        shapes = []
        dtypes = set()
        for p in module.parameters(recurse=False):
            shapes.append(str(tuple(p.shape)))
            dtypes.add(str(p.dtype))

        shape_str = ", ".join(shapes) if shapes else "N/A"
        dtype_str = ", ".join(dtypes) if dtypes else "N/A"

        return LayerInfo(
            name=name,
            type=type(module).__name__,
            num_params=total_params,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            shape=shape_str,
            dtype=dtype_str,
        )

    @classmethod
    def analyze_model(cls, model: nn.Module, detailed: bool = True) -> ModelParameterSummary:
        """
        Generate comprehensive parameter analysis for a model.

        Args:
            model: PyTorch model to analyze
            detailed: Whether to include per-layer analysis

        Returns:
            ModelParameterSummary object
        """
        total_params, trainable_params = cls.count_parameters(model)
        frozen_params = total_params - trainable_params
        size_mb = cls.get_parameter_size_mb(model)

        # Analyze layers
        layers = []
        if detailed:
            for name, module in model.named_modules():
                # Skip the model itself and containers without direct parameters
                if name and list(module.parameters(recurse=False)):
                    layer_info = cls.analyze_layer(name, module)
                    if layer_info.num_params > 0:
                        layers.append(layer_info)

        # Parameter breakdown by component
        param_breakdown = {}
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                param_breakdown[name] = module_params

        return ModelParameterSummary(
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            total_size_mb=size_mb,
            layers=layers,
            param_breakdown=param_breakdown,
        )

    @staticmethod
    def format_number(num: int) -> str:
        """Format large numbers with commas and SI suffixes."""
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.2f}B ({num:,})"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.2f}M ({num:,})"
        elif num >= 1_000:
            return f"{num/1_000:.2f}K ({num:,})"
        else:
            return f"{num:,}"

    @classmethod
    def print_summary_table(cls, summary: ModelParameterSummary, show_layers: bool = False):
        """
        Print a formatted parameter summary table.

        Args:
            summary: ModelParameterSummary object
            show_layers: Whether to show detailed layer information
        """
        print("\n" + "="*80)
        print("MODEL PARAMETER SUMMARY")
        print("="*80)

        # Overall statistics
        print(f"\n{'Metric':<30} {'Value':>30}")
        print("-"*80)
        print(f"{'Total Parameters:':<30} {cls.format_number(summary.total_params):>30}")
        print(f"{'Trainable Parameters:':<30} {cls.format_number(summary.trainable_params):>30}")
        print(f"{'Frozen Parameters:':<30} {cls.format_number(summary.frozen_params):>30}")
        print(f"{'Model Size (MB):':<30} {summary.total_size_mb:>30.2f}")
        print(f"{'Trainable Percentage:':<30} {(summary.trainable_params/summary.total_params*100):>29.2f}%")

        # Component breakdown
        if summary.param_breakdown:
            print("\n" + "="*80)
            print("PARAMETER BREAKDOWN BY COMPONENT")
            print("="*80)
            print(f"\n{'Component':<40} {'Parameters':>20} {'Percentage':>15}")
            print("-"*80)

            for component, params in sorted(
                summary.param_breakdown.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (params / summary.total_params) * 100
                print(f"{component:<40} {cls.format_number(params):>20} {percentage:>14.2f}%")

        # Layer details
        if show_layers and summary.layers:
            print("\n" + "="*80)
            print("DETAILED LAYER ANALYSIS")
            print("="*80)
            print(f"\n{'Layer Name':<40} {'Type':<20} {'Parameters':>15}")
            print("-"*80)

            for layer in summary.layers[:50]:  # Show first 50 layers
                params_str = cls.format_number(layer.num_params)
                print(f"{layer.name:<40} {layer.type:<20} {params_str:>15}")
                if layer.frozen_params > 0:
                    print(f"{'  └─ Frozen:':<40} {'':<20} {cls.format_number(layer.frozen_params):>15}")

            if len(summary.layers) > 50:
                print(f"\n... and {len(summary.layers) - 50} more layers")

        print("\n" + "="*80 + "\n")

    @staticmethod
    def save_summary(summary: ModelParameterSummary, output_path: Path):
        """
        Save parameter summary to JSON file.

        Args:
            summary: ModelParameterSummary object
            output_path: Path to save JSON file
        """
        data = summary.to_dict()

        # Add layer details
        data["layers"] = [
            {
                "name": layer.name,
                "type": layer.type,
                "num_params": layer.num_params,
                "trainable_params": layer.trainable_params,
                "frozen_params": layer.frozen_params,
            }
            for layer in summary.layers
        ]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Parameter summary saved to: {output_path}")


def analyze_codebert_model(model_name: str = "microsoft/codebert-base") -> ModelParameterSummary:
    """
    Analyze a CodeBERT or similar transformer model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        ModelParameterSummary object
    """
    print(f"\nLoading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)

    print("Analyzing model architecture...")
    analyzer = ModelParameterAnalyzer()
    summary = analyzer.analyze_model(model, detailed=True)

    return summary


if __name__ == "__main__":
    # Example usage
    summary = analyze_codebert_model("microsoft/codebert-base")
    ModelParameterAnalyzer.print_summary_table(summary, show_layers=True)
