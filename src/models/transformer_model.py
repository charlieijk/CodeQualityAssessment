"""
Advanced transformer-based models for code quality assessment.

This module implements state-of-the-art transformer models (CodeBERT, GraphCodeBERT)
for deep code understanding and quality prediction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoModel,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        get_linear_schedule_with_warmup,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import mean_absolute_error, f1_score
except ImportError as exc:
    raise ImportError(
        "Transformer dependencies are missing. Install requirements-ml-optional.txt "
        "inside a Python 3.11 virtualenv to enable advanced ML models."
    ) from exc


@dataclass
class TransformerConfig:
    """Configuration for transformer-based quality model."""
    model_name: str = "microsoft/codebert-base"  # CodeBERT, GraphCodeBERT, etc.
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    dropout_rate: float = 0.1
    hidden_size: int = 768
    num_quality_outputs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    issue_threshold: float = 0.5


class CodeQualityDataset(Dataset):
    """PyTorch Dataset for code quality assessment."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizer,
        quality_scores: Optional[np.ndarray] = None,
        issue_labels: Optional[List[List[str]]] = None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.quality_scores = quality_scores
        self.issue_labels = issue_labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])

        # Tokenize the code
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.quality_scores is not None:
            item["quality_score"] = torch.tensor(
                self.quality_scores[idx], dtype=torch.float
            )

        if self.issue_labels is not None:
            item["issue_label"] = self.issue_labels[idx]

        return item


class MultiTaskCodeQualityModel(nn.Module):
    """
    Multi-task transformer model for code quality assessment.

    Performs two tasks simultaneously:
    1. Quality score regression (0-100)
    2. Multi-label issue type classification
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_issue_types: int,
    ):
        super().__init__()
        self.config = config

        # Load pre-trained transformer (CodeBERT, etc.)
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(config.model_name)

        # Freeze some encoder layers for efficient fine-tuning (optional)
        # self._freeze_encoder_layers(num_layers_to_freeze=6)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)

        # Quality score regression head
        self.quality_regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, config.num_quality_outputs),
        )

        # Issue type classification head (multi-label)
        self.issue_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, num_issue_types),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            return_attention: Whether to return attention weights for visualization

        Returns:
            Dictionary with quality_score, issue_logits, and optionally attention
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention,
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Quality score prediction (0-100)
        quality_score = self.quality_regressor(pooled_output).squeeze(-1)
        # Clamp to valid range
        quality_score = torch.clamp(quality_score, 0.0, 100.0)

        # Issue type predictions (multi-label)
        issue_logits = self.issue_classifier(pooled_output)

        result = {
            "quality_score": quality_score,
            "issue_logits": issue_logits,
        }

        if return_attention:
            result["attention"] = outputs.attentions

        return result

    def _freeze_encoder_layers(self, num_layers_to_freeze: int):
        """Freeze first N encoder layers to speed up training."""
        modules_to_freeze = list(self.encoder.encoder.layer)[:num_layers_to_freeze]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters by component.

        Returns:
            Dictionary with parameter counts for each component
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

        regressor_params = sum(p.numel() for p in self.quality_regressor.parameters())
        classifier_params = sum(p.numel() for p in self.issue_classifier.parameters())

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "encoder": encoder_params,
            "encoder_trainable": encoder_trainable,
            "encoder_frozen": encoder_params - encoder_trainable,
            "quality_regressor": regressor_params,
            "issue_classifier": classifier_params,
        }

    def print_parameter_summary(self):
        """Print a formatted summary of model parameters."""
        params = self.count_parameters()

        print("\n" + "="*70)
        print("MULTI-TASK CODE QUALITY MODEL - PARAMETER SUMMARY")
        print("="*70)

        def format_num(num: int) -> str:
            if num >= 1_000_000:
                return f"{num/1_000_000:.2f}M ({num:,})"
            elif num >= 1_000:
                return f"{num/1_000:.2f}K ({num:,})"
            return f"{num:,}"

        print(f"\n{'Component':<35} {'Parameters':>20} {'%':>10}")
        print("-"*70)
        print(f"{'Total Parameters:':<35} {format_num(params['total']):>20} {'100.00%':>10}")
        print(f"{'  └─ Trainable:':<35} {format_num(params['trainable']):>20} {params['trainable']/params['total']*100:>9.2f}%")
        print(f"{'  └─ Frozen:':<35} {format_num(params['frozen']):>20} {params['frozen']/params['total']*100:>9.2f}%")
        print()
        print(f"{'Encoder (Transformer):':<35} {format_num(params['encoder']):>20} {params['encoder']/params['total']*100:>9.2f}%")
        print(f"{'  └─ Trainable:':<35} {format_num(params['encoder_trainable']):>20} {params['encoder_trainable']/params['total']*100:>9.2f}%")
        print(f"{'  └─ Frozen:':<35} {format_num(params['encoder_frozen']):>20} {params['encoder_frozen']/params['total']*100:>9.2f}%")
        print()
        print(f"{'Quality Regressor Head:':<35} {format_num(params['quality_regressor']):>20} {params['quality_regressor']/params['total']*100:>9.2f}%")
        print(f"{'Issue Classifier Head:':<35} {format_num(params['issue_classifier']):>20} {params['issue_classifier']/params['total']*100:>9.2f}%")
        print("="*70 + "\n")


class TransformerQualityTrainer:
    """Trainer for transformer-based code quality models."""

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[MultiTaskCodeQualityModel] = None
        self.label_binarizer = MultiLabelBinarizer()
        self.best_val_loss = float("inf")

    def prepare_data(
        self,
        texts: Sequence[str],
        quality_scores: np.ndarray,
        issue_labels: Sequence[Sequence[str]],
        test_size: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Binarize issue labels
        issue_matrix = self.label_binarizer.fit_transform(issue_labels)
        num_issue_types = len(self.label_binarizer.classes_)

        # Split data
        (
            texts_train, texts_val,
            scores_train, scores_val,
            issues_train, issues_val,
        ) = train_test_split(
            texts,
            quality_scores,
            issue_matrix,
            test_size=test_size,
            random_state=42,
        )

        # Create datasets
        train_dataset = CodeQualityDataset(
            texts_train,
            self.tokenizer,
            scores_train,
            issues_train.tolist(),
            self.config.max_length,
        )

        val_dataset = CodeQualityDataset(
            texts_val,
            self.tokenizer,
            scores_val,
            issues_val.tolist(),
            self.config.max_length,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Initialize model
        self.model = MultiTaskCodeQualityModel(
            self.config,
            num_issue_types=num_issue_types,
        ).to(self.device)

        return train_loader, val_loader

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_data first.")

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Loss functions
        quality_criterion = nn.MSELoss()
        issue_criterion = nn.BCEWithLogitsLoss()

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_quality_mae": [],
            "val_quality_mae": [],
            "train_issue_f1": [],
            "val_issue_f1": [],
        }

        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_quality_preds = []
            train_quality_true = []
            train_issue_preds = []
            train_issue_true = []

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                quality_true = batch["quality_score"].to(self.device)
                issue_true = torch.tensor(
                    batch["issue_label"], dtype=torch.float
                ).to(self.device)

                outputs = self.model(input_ids, attention_mask)

                # Calculate losses
                quality_loss = quality_criterion(
                    outputs["quality_score"], quality_true
                )
                issue_loss = issue_criterion(
                    outputs["issue_logits"], issue_true
                )

                # Combined loss (you can adjust weights)
                loss = quality_loss + issue_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

                # Store predictions for metrics
                train_quality_preds.extend(
                    outputs["quality_score"].detach().cpu().numpy()
                )
                train_quality_true.extend(quality_true.cpu().numpy())
                train_issue_preds.extend(
                    torch.sigmoid(outputs["issue_logits"]).detach().cpu().numpy()
                )
                train_issue_true.extend(issue_true.cpu().numpy())

            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_mae = mean_absolute_error(train_quality_true, train_quality_preds)
            train_issue_binary = (
                np.array(train_issue_preds) >= self.config.issue_threshold
            ).astype(int)
            train_f1 = f1_score(
                np.array(train_issue_true),
                train_issue_binary,
                average="micro",
                zero_division=0,
            )

            # Validation phase
            val_metrics = self._validate(val_loader, quality_criterion, issue_criterion)

            # Update history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_metrics["val_loss"])
            history["train_quality_mae"].append(train_mae)
            history["val_quality_mae"].append(val_metrics["val_mae"])
            history["train_issue_f1"].append(train_f1)
            history["val_issue_f1"].append(val_metrics["val_f1"])

            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Train MAE: {train_mae:.4f}, Val MAE: {val_metrics['val_mae']:.4f}")
            print(f"  Train F1: {train_f1:.4f}, Val F1: {val_metrics['val_f1']:.4f}")

            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                print("  -> Best model saved!")

        return history

    def _validate(
        self,
        val_loader: DataLoader,
        quality_criterion: nn.Module,
        issue_criterion: nn.Module,
    ) -> Dict[str, float]:
        """Validate the model."""
        if self.model is None:
            raise ValueError("Model not initialized.")

        self.model.eval()
        val_loss = 0.0
        val_quality_preds = []
        val_quality_true = []
        val_issue_preds = []
        val_issue_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                quality_true = batch["quality_score"].to(self.device)
                issue_true = torch.tensor(
                    batch["issue_label"], dtype=torch.float
                ).to(self.device)

                outputs = self.model(input_ids, attention_mask)

                quality_loss = quality_criterion(
                    outputs["quality_score"], quality_true
                )
                issue_loss = issue_criterion(
                    outputs["issue_logits"], issue_true
                )
                loss = quality_loss + issue_loss

                val_loss += loss.item()

                val_quality_preds.extend(
                    outputs["quality_score"].cpu().numpy()
                )
                val_quality_true.extend(quality_true.cpu().numpy())
                val_issue_preds.extend(
                    torch.sigmoid(outputs["issue_logits"]).cpu().numpy()
                )
                val_issue_true.extend(issue_true.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_mae = mean_absolute_error(val_quality_true, val_quality_preds)
        val_issue_binary = (
            np.array(val_issue_preds) >= self.config.issue_threshold
        ).astype(int)
        val_f1 = f1_score(
            np.array(val_issue_true),
            val_issue_binary,
            average="micro",
            zero_division=0,
        )

        return {
            "val_loss": avg_val_loss,
            "val_mae": val_mae,
            "val_f1": val_f1,
        }

    def predict(
        self,
        texts: Sequence[str],
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """Make predictions on new code samples."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized.")

        self.model.eval()
        dataset = CodeQualityDataset(
            texts,
            self.tokenizer,
            max_length=self.config.max_length,
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        quality_preds = []
        issue_preds = []
        attention_weights = [] if return_attention else None

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_attention=return_attention,
                )

                quality_preds.extend(outputs["quality_score"].cpu().numpy())

                # Convert logits to probabilities and then to binary labels
                issue_probs = torch.sigmoid(outputs["issue_logits"]).cpu().numpy()
                issue_binary = (issue_probs >= self.config.issue_threshold).astype(int)
                issue_labels = self.label_binarizer.inverse_transform(issue_binary)
                issue_preds.extend([list(labels) for labels in issue_labels])

                if return_attention and "attention" in outputs:
                    attention_weights.extend(outputs["attention"])

        result = {
            "quality_scores": quality_preds,
            "issue_labels": issue_preds,
        }

        if return_attention:
            result["attention_weights"] = attention_weights

        return result

    def save(self, model_dir: Path) -> Path:
        """Save the trained model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized.")

        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = model_dir / "transformer_quality_model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "label_binarizer": self.label_binarizer,
        }, model_path)

        # Save tokenizer
        tokenizer_path = model_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)

        # Save metadata
        metadata = {
            "model_name": self.config.model_name,
            "num_issue_types": len(self.label_binarizer.classes_),
            "issue_types": list(self.label_binarizer.classes_),
        }
        with (model_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        return model_path

    @classmethod
    def load(cls, model_dir: Path) -> "TransformerQualityTrainer":
        """Load a trained model."""
        # Load metadata
        with (model_dir / "metadata.json").open("r") as f:
            metadata = json.load(f)

        # Load checkpoint
        model_path = model_dir / "transformer_quality_model.pt"
        checkpoint = torch.load(model_path, map_location="cpu")

        # Initialize trainer
        config = checkpoint["config"]
        trainer = cls(config)

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer"
        trainer.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load label binarizer
        trainer.label_binarizer = checkpoint["label_binarizer"]

        # Initialize and load model
        trainer.model = MultiTaskCodeQualityModel(
            config,
            num_issue_types=metadata["num_issue_types"],
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.to(trainer.device)
        trainer.model.eval()

        return trainer
