"""
Baseline ML training utilities for code quality prediction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import classification_report, f1_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError as exc:  # pragma: no cover - optional deps
    raise ImportError(
        "Optional ML dependencies are missing. Install requirements-ml-optional.txt "
        "inside a Python 3.11 virtualenv to enable training."
    ) from exc


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise ValueError("Dataset file is empty.")
    return entries


def _extract_training_arrays(entries: Sequence[Dict[str, Any]]) -> Tuple[List[str], np.ndarray, List[List[str]]]:
    texts: List[str] = []
    quality_scores: List[float] = []
    issue_labels: List[List[str]] = []

    for entry in entries:
        texts.append(entry.get("text", ""))
        quality_scores.append(float(entry.get("quality_score", 0.0)))
        issues = entry.get("issues", [])
        issue_labels.append([issue.get("issue_type") for issue in issues if issue.get("issue_type")])

    return texts, np.array(quality_scores), issue_labels


@dataclass
class TrainingConfig:
    dataset_path: Path = Path("data/processed/dataset.jsonl")
    model_dir: Path = Path("data/processed/models")
    test_size: float = 0.2
    random_state: int = 42
    issue_threshold: float = 0.5
    max_features: int = 5000


class BaselineQualityModel:
    """Simple TF-IDF + linear models for regression and multi-label classification."""

    def __init__(
        self,
        max_features: int = 5000,
        issue_threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
        )
        self.regressor = Ridge(alpha=1.0)
        self.issue_classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                random_state=random_state,
            )
        )
        self.label_binarizer = MultiLabelBinarizer()
        self.issue_threshold = issue_threshold
        self.fitted_issue_classifier = False

    def fit(
        self,
        texts: Sequence[str],
        quality_scores: np.ndarray,
        issue_labels: Sequence[Sequence[str]],
    ) -> "BaselineQualityModel":
        X = self.vectorizer.fit_transform(texts)
        self.regressor.fit(X, quality_scores)

        label_matrix = self.label_binarizer.fit_transform(issue_labels)
        if label_matrix.size == 0 or label_matrix.sum() == 0:
            self.fitted_issue_classifier = False
        else:
            self.issue_classifier.fit(X, label_matrix)
            self.fitted_issue_classifier = True

        return self

    def predict(
        self,
        texts: Sequence[str],
        issue_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        threshold = issue_threshold if issue_threshold is not None else self.issue_threshold
        X = self.vectorizer.transform(texts)
        quality_predictions = self.regressor.predict(X).tolist()
        issue_predictions: List[List[str]] = [[] for _ in texts]

        if self.fitted_issue_classifier:
            probs = self.issue_classifier.predict_proba(X)
            binary = (probs >= threshold).astype(int)
            decoded = self.label_binarizer.inverse_transform(binary)
            issue_predictions = [list(labels) for labels in decoded]

        return {"quality_scores": quality_predictions, "issue_labels": issue_predictions}

    def evaluate(
        self,
        texts: Sequence[str],
        true_scores: np.ndarray,
        true_issue_labels: Sequence[Sequence[str]],
        issue_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        predictions = self.predict(texts, issue_threshold=issue_threshold)
        mae = mean_absolute_error(true_scores, predictions["quality_scores"])
        metrics: Dict[str, Any] = {"mae": float(mae)}

        if self.fitted_issue_classifier and self.label_binarizer.classes_.size > 0:
            y_true = self.label_binarizer.transform(true_issue_labels)
            y_pred_binary = self.label_binarizer.transform(predictions["issue_labels"])
            metrics["issue_f1_micro"] = float(
                f1_score(y_true, y_pred_binary, average="micro", zero_division=0)
            )
            metrics["issue_report"] = classification_report(
                y_true,
                y_pred_binary,
                target_names=[str(label) for label in self.label_binarizer.classes_],
                zero_division=0,
                output_dict=True,
            )

        return metrics

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "regressor": self.regressor,
            "issue_classifier": self.issue_classifier,
            "label_binarizer": self.label_binarizer,
            "issue_threshold": self.issue_threshold,
            "fitted_issue_classifier": self.fitted_issue_classifier,
        }
        model_path = model_dir / "baseline_quality_model.joblib"
        joblib.dump(payload, model_path)
        return model_path

    @classmethod
    def load(cls, model_path: Path) -> "BaselineQualityModel":
        payload = joblib.load(model_path)
        instance = cls()
        instance.vectorizer = payload["vectorizer"]
        instance.regressor = payload["regressor"]
        instance.issue_classifier = payload["issue_classifier"]
        instance.label_binarizer = payload["label_binarizer"]
        instance.issue_threshold = payload["issue_threshold"]
        instance.fitted_issue_classifier = payload["fitted_issue_classifier"]
        return instance


def train_baseline(config: TrainingConfig) -> Dict[str, Any]:
    entries = _load_dataset(config.dataset_path)
    texts, scores, issue_labels = _extract_training_arrays(entries)

    (
        texts_train,
        texts_test,
        scores_train,
        scores_test,
        issues_train,
        issues_test,
    ) = train_test_split(
        texts,
        scores,
        issue_labels,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )

    model = BaselineQualityModel(
        max_features=config.max_features,
        issue_threshold=config.issue_threshold,
        random_state=config.random_state,
    )
    model.fit(texts_train, scores_train, issues_train)
    metrics = model.evaluate(texts_test, scores_test, issues_test)

    # Fit on full dataset before saving for production use.
    final_model = BaselineQualityModel(
        max_features=config.max_features,
        issue_threshold=config.issue_threshold,
        random_state=config.random_state,
    )
    final_model.fit(texts, scores, issue_labels)
    model_path = final_model.save(config.model_dir)
    metrics_path = config.model_dir / "metrics.json"
    metrics_with_paths = {
        **metrics,
        "model_path": str(model_path),
        "dataset_path": str(config.dataset_path),
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_with_paths, fh, indent=2)

    return metrics_with_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline ML quality model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset.jsonl",
        help="Path to the JSONL dataset produced by DatasetBuilder.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="data/processed/models",
        help="Directory to store the trained model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--issue-threshold", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        dataset_path=Path(args.dataset),
        model_dir=Path(args.model_dir),
        test_size=args.test_size,
        issue_threshold=args.issue_threshold,
        max_features=args.max_features,
        random_state=args.random_state,
    )
    metrics = train_baseline(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
