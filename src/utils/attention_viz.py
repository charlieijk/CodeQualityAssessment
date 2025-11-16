"""
Attention visualization utilities for transformer models.

Visualizes attention weights to understand which code parts affect quality predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as exc:
    raise ImportError(
        "Visualization dependencies missing. Install requirements-ml-optional.txt "
        "to enable attention visualization."
    ) from exc


class AttentionVisualizer:
    """Visualize attention weights from transformer models."""

    def __init__(self, tokenizer: Any):
        """
        Initialize the attention visualizer.

        Args:
            tokenizer: HuggingFace tokenizer used by the model
        """
        self.tokenizer = tokenizer
        self.colormap = self._create_colormap()

    def _create_colormap(self) -> LinearSegmentedColormap:
        """Create a custom colormap for attention visualization."""
        colors = ["#ffffff", "#e3f2fd", "#90caf9", "#42a5f5", "#1976d2", "#0d47a1"]
        return LinearSegmentedColormap.from_list("attention", colors)

    def visualize_token_attention(
        self,
        code_text: str,
        attention_weights: np.ndarray,
        layer_idx: int = -1,
        head_idx: int = 0,
        save_path: Optional[Path] = None,
        figsize: tuple = (14, 8),
    ) -> None:
        """
        Visualize attention weights for tokens.

        Args:
            code_text: Original code text
            attention_weights: Attention weights from model (layers, heads, seq_len, seq_len)
            layer_idx: Which layer to visualize (-1 for last layer)
            head_idx: Which attention head to visualize
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        # Tokenize the code
        tokens = self.tokenizer.tokenize(code_text)
        tokens = ["[CLS]"] + tokens[:min(len(tokens), 510)] + ["[SEP]"]

        # Extract attention for specified layer and head
        if len(attention_weights.shape) == 4:
            # Shape: (layers, heads, seq_len, seq_len)
            attn = attention_weights[layer_idx, head_idx]
        elif len(attention_weights.shape) == 3:
            # Shape: (heads, seq_len, seq_len)
            attn = attention_weights[head_idx]
        else:
            # Shape: (seq_len, seq_len)
            attn = attention_weights

        # Trim attention matrix to match tokens
        max_len = len(tokens)
        attn = attn[:max_len, :max_len]

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(
            attn,
            cmap=self.colormap,
            xticklabels=tokens,
            yticklabels=tokens,
            cbar=True,
            square=True,
            ax=ax,
            cbar_kws={"label": "Attention Weight"},
        )

        ax.set_title(
            f"Attention Weights - Layer {layer_idx}, Head {head_idx}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Key Tokens", fontsize=12)
        ax.set_ylabel("Query Tokens", fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Attention visualization saved to {save_path}")

        plt.show()

    def visualize_cls_attention(
        self,
        code_text: str,
        attention_weights: np.ndarray,
        layer_idx: int = -1,
        save_path: Optional[Path] = None,
        top_k: int = 20,
        figsize: tuple = (12, 6),
    ) -> Dict[str, float]:
        """
        Visualize attention from [CLS] token to other tokens.

        This shows which code tokens the model focuses on for quality prediction.

        Args:
            code_text: Original code text
            attention_weights: Attention weights from model
            layer_idx: Which layer to visualize (-1 for last layer)
            save_path: Optional path to save the figure
            top_k: Number of top tokens to display
            figsize: Figure size

        Returns:
            Dictionary mapping tokens to their attention scores
        """
        # Tokenize the code
        tokens = self.tokenizer.tokenize(code_text)
        tokens = ["[CLS]"] + tokens[:min(len(tokens), 510)] + ["[SEP]"]

        # Extract attention from [CLS] token (index 0)
        # Average over all attention heads for the specified layer
        if len(attention_weights.shape) == 4:
            # Shape: (layers, heads, seq_len, seq_len)
            cls_attn = attention_weights[layer_idx, :, 0, :].mean(axis=0)
        elif len(attention_weights.shape) == 3:
            # Shape: (heads, seq_len, seq_len)
            cls_attn = attention_weights[:, 0, :].mean(axis=0)
        else:
            cls_attn = attention_weights[0, :]

        # Trim to match tokens
        cls_attn = cls_attn[: len(tokens)]

        # Get top-k tokens
        top_indices = np.argsort(cls_attn)[-top_k:][::-1]
        top_tokens = [tokens[i] for i in top_indices]
        top_scores = cls_attn[top_indices]

        # Create token->score mapping
        token_scores = {
            token: float(score) for token, score in zip(top_tokens, top_scores)
        }

        # Create bar plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Blues(top_scores / top_scores.max())
        bars = ax.barh(range(len(top_tokens)), top_scores, color=colors)

        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens)
        ax.set_xlabel("Attention Weight", fontsize=12)
        ax.set_title(
            f"Top {top_k} Tokens Attended by [CLS] - Layer {layer_idx}",
            fontsize=14,
            fontweight="bold",
        )

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(
                score,
                i,
                f" {score:.4f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"CLS attention visualization saved to {save_path}")

        plt.show()

        return token_scores

    def visualize_attention_by_layer(
        self,
        code_text: str,
        attention_weights: np.ndarray,
        save_path: Optional[Path] = None,
        figsize: tuple = (16, 12),
    ) -> None:
        """
        Visualize how attention changes across layers.

        Args:
            code_text: Original code text
            attention_weights: Attention weights (layers, heads, seq_len, seq_len)
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(attention_weights.shape) != 4:
            raise ValueError(
                "Expected attention_weights with shape (layers, heads, seq_len, seq_len)"
            )

        num_layers = attention_weights.shape[0]
        tokens = self.tokenizer.tokenize(code_text)
        tokens = ["[CLS]"] + tokens[:min(len(tokens), 50)] + ["[SEP]"]  # Limit for visualization

        # Create subplots
        n_cols = 4
        n_rows = (num_layers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if num_layers > 1 else [axes]

        for layer_idx in range(num_layers):
            # Average attention over heads and get [CLS] attention
            cls_attn = attention_weights[layer_idx, :, 0, :].mean(axis=0)
            cls_attn = cls_attn[: len(tokens)]

            ax = axes[layer_idx]
            colors = plt.cm.Blues(cls_attn / (cls_attn.max() + 1e-10))

            ax.bar(range(len(tokens)), cls_attn, color=colors)
            ax.set_title(f"Layer {layer_idx}", fontsize=10)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_ylabel("Attention", fontsize=8)

        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            "[CLS] Token Attention Across Layers",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Layer-wise attention visualization saved to {save_path}")

        plt.show()

    def create_attention_heatmap_overlay(
        self,
        code_text: str,
        attention_scores: Dict[str, float],
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Create an HTML representation of code with attention highlighting.

        Args:
            code_text: Original code text
            attention_scores: Dictionary mapping tokens to attention scores
            save_path: Optional path to save HTML file

        Returns:
            HTML string with highlighted code
        """
        # Create color mapping based on attention scores
        max_score = max(attention_scores.values()) if attention_scores else 1.0

        html_parts = ['<div style="font-family: monospace; white-space: pre-wrap;">']

        for token, score in attention_scores.items():
            # Normalize score to [0, 1]
            normalized = score / max_score
            # Create color (light yellow to dark orange)
            alpha = normalized
            color = f"rgba(255, 165, 0, {alpha})"

            html_parts.append(
                f'<span style="background-color: {color}; '
                f'padding: 2px 4px; margin: 1px; border-radius: 3px;">'
                f'{token}</span>'
            )

        html_parts.append("</div>")

        html_output = "".join(html_parts)

        if save_path:
            with open(save_path, "w") as f:
                f.write(
                    f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Code Attention Heatmap</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 20px; }}
                            h2 {{ color: #333; }}
                        </style>
                    </head>
                    <body>
                        <h2>Code Quality Attention Heatmap</h2>
                        <p>Brighter colors indicate higher attention from the model.</p>
                        {html_output}
                    </body>
                    </html>
                    """
                )
            print(f"HTML attention heatmap saved to {save_path}")

        return html_output


def aggregate_attention_heads(
    attention_weights: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregate attention across multiple heads.

    Args:
        attention_weights: Attention weights (heads, seq_len, seq_len)
        method: Aggregation method ('mean', 'max', 'min')

    Returns:
        Aggregated attention (seq_len, seq_len)
    """
    if method == "mean":
        return attention_weights.mean(axis=0)
    elif method == "max":
        return attention_weights.max(axis=0)
    elif method == "min":
        return attention_weights.min(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def extract_attention_patterns(
    attention_weights: np.ndarray,
    tokens: Sequence[str],
    pattern_type: str = "syntax",
) -> List[Dict[str, Any]]:
    """
    Extract specific attention patterns (e.g., syntax, naming, structure).

    Args:
        attention_weights: Attention matrix
        tokens: List of tokens
        pattern_type: Type of pattern to extract

    Returns:
        List of detected patterns with their scores
    """
    patterns = []

    if pattern_type == "syntax":
        # Look for attention to syntax keywords
        syntax_keywords = {"def", "class", "if", "for", "while", "return", "import"}
        for i, token in enumerate(tokens):
            if token.lower() in syntax_keywords:
                avg_attention = attention_weights[:, i].mean()
                patterns.append({
                    "token": token,
                    "position": i,
                    "attention_score": float(avg_attention),
                    "type": "syntax_keyword",
                })

    elif pattern_type == "naming":
        # Look for attention to variable/function names
        for i, token in enumerate(tokens):
            if token.isidentifier() and not token.startswith("__"):
                avg_attention = attention_weights[:, i].mean()
                patterns.append({
                    "token": token,
                    "position": i,
                    "attention_score": float(avg_attention),
                    "type": "identifier",
                })

    # Sort by attention score
    patterns.sort(key=lambda x: x["attention_score"], reverse=True)

    return patterns
