"""
Hybrid analyzer that combines rule-based and ML-based code quality analysis.

This module provides a unified interface for using both traditional rule-based
analysis and advanced transformer models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.code_analyzer import CodeQualityAnalyzer, QualityIssue


class HybridCodeAnalyzer:
    """
    Hybrid analyzer combining rule-based and ML-based approaches.

    Falls back gracefully to rule-based analysis if ML models are unavailable.
    """

    def __init__(
        self,
        use_transformers: bool = True,
        transformer_model_path: Optional[Path] = None,
        use_baseline_ml: bool = True,
        baseline_model_path: Optional[Path] = None,
    ):
        """
        Initialize the hybrid analyzer.

        Args:
            use_transformers: Whether to use transformer models
            transformer_model_path: Path to trained transformer model
            use_baseline_ml: Whether to use baseline ML model
            baseline_model_path: Path to baseline ML model
        """
        # Always initialize rule-based analyzer as fallback
        self.rule_analyzer = CodeQualityAnalyzer()

        # Try to load ML models
        self.transformer_model = None
        self.baseline_model = None
        self.use_transformers = use_transformers
        self.use_baseline_ml = use_baseline_ml

        if use_transformers and transformer_model_path:
            self._load_transformer_model(transformer_model_path)

        if use_baseline_ml and baseline_model_path:
            self._load_baseline_model(baseline_model_path)

    def _load_transformer_model(self, model_path: Path) -> None:
        """Load transformer model if available."""
        try:
            from src.models.transformer_model import TransformerQualityTrainer

            if model_path.exists():
                self.transformer_model = TransformerQualityTrainer.load(model_path)
                print(f"Loaded transformer model from {model_path}")
            else:
                print(f"Transformer model not found at {model_path}, using rule-based only")
        except ImportError:
            print(
                "Transformer dependencies not available. "
                "Install requirements-ml-optional.txt to enable transformer models."
            )
        except Exception as e:
            print(f"Failed to load transformer model: {e}")

    def _load_baseline_model(self, model_path: Path) -> None:
        """Load baseline ML model if available."""
        try:
            from src.models.baseline_trainer import BaselineQualityModel

            if model_path.exists():
                self.baseline_model = BaselineQualityModel.load(model_path)
                print(f"Loaded baseline ML model from {model_path}")
            else:
                print(f"Baseline model not found at {model_path}")
        except ImportError:
            print(
                "Baseline ML dependencies not available. "
                "Install requirements-ml-optional.txt to enable ML models."
            )
        except Exception as e:
            print(f"Failed to load baseline model: {e}")

    def analyze_code(
        self,
        code_text: str,
        ensemble_strategy: str = "weighted_average",
    ) -> Dict[str, Any]:
        """
        Analyze code quality using available models.

        Args:
            code_text: Source code to analyze
            ensemble_strategy: How to combine predictions
                - "rule_only": Use only rule-based analysis
                - "ml_only": Use only ML models
                - "weighted_average": Combine predictions with weights
                - "ml_override": Use ML predictions but keep rule-based issues

        Returns:
            Analysis result with quality score, issues, and model metadata
        """
        # Always get rule-based analysis
        rule_result = self.rule_analyzer.analyze_code(code_text)

        # If no ML models available or rule_only strategy, return rule-based result
        if (
            ensemble_strategy == "rule_only"
            or (not self.transformer_model and not self.baseline_model)
        ):
            return {
                **rule_result,
                "model_used": "rule_based",
                "ml_available": False,
            }

        # Get ML predictions
        ml_quality_score = None
        ml_issues = []
        model_used = "rule_based"

        # Try transformer model first (most accurate)
        if self.transformer_model and self.use_transformers:
            try:
                transformer_pred = self.transformer_model.predict([code_text])
                ml_quality_score = transformer_pred["quality_scores"][0]
                ml_issue_types = transformer_pred["issue_labels"][0]
                ml_issues = ml_issue_types
                model_used = "transformer"
            except Exception as e:
                print(f"Transformer prediction failed: {e}")

        # Fall back to baseline ML if transformer failed
        if ml_quality_score is None and self.baseline_model and self.use_baseline_ml:
            try:
                baseline_pred = self.baseline_model.predict([code_text])
                ml_quality_score = baseline_pred["quality_scores"][0]
                ml_issue_types = baseline_pred["issue_labels"][0]
                ml_issues = ml_issue_types
                model_used = "baseline_ml"
            except Exception as e:
                print(f"Baseline ML prediction failed: {e}")

        # If ML prediction failed, return rule-based result
        if ml_quality_score is None:
            return {
                **rule_result,
                "model_used": "rule_based (ML failed)",
                "ml_available": True,
                "ml_error": True,
            }

        # Combine predictions based on strategy
        if ensemble_strategy == "ml_only":
            # Use ML prediction only
            final_score = ml_quality_score
            # Map ML issue types to QualityIssue objects (simplified)
            final_issues = self._create_issues_from_ml(ml_issues, code_text)

        elif ensemble_strategy == "ml_override":
            # Use ML score but keep detailed rule-based issues
            final_score = ml_quality_score
            final_issues = rule_result["issues"]

        else:  # weighted_average (default)
            # Combine scores with weights
            rule_weight = 0.3
            ml_weight = 0.7
            final_score = (
                rule_weight * rule_result["quality_score"]
                + ml_weight * ml_quality_score
            )

            # Combine issues (union of both)
            final_issues = rule_result["issues"]

            # Add ML-detected issues that aren't in rule-based
            rule_issue_types = {issue["issue_type"] for issue in rule_result["issues"]}
            for ml_issue in ml_issues:
                if ml_issue not in rule_issue_types:
                    final_issues.append({
                        "issue_type": ml_issue,
                        "severity": "medium",
                        "line_number": 0,
                        "description": f"ML model detected: {ml_issue}",
                        "suggestion": "Review this issue identified by ML analysis",
                        "code_snippet": "",
                    })

        return {
            "issues": final_issues,
            "quality_score": round(final_score, 1),
            "total_issues": len(final_issues),
            "severity_breakdown": self._get_severity_breakdown(final_issues),
            "model_used": model_used,
            "ensemble_strategy": ensemble_strategy,
            "ml_available": True,
            "predictions": {
                "rule_based_score": rule_result["quality_score"],
                "ml_score": ml_quality_score,
                "final_score": round(final_score, 1),
            },
        }

    def _create_issues_from_ml(
        self,
        ml_issue_types: List[str],
        code_text: str,
    ) -> List[Dict[str, Any]]:
        """Convert ML issue types to QualityIssue format."""
        issues = []
        for issue_type in ml_issue_types:
            issues.append({
                "issue_type": issue_type,
                "severity": "medium",  # Default severity for ML-detected issues
                "line_number": 0,  # ML models don't provide line-level info
                "description": f"Detected by ML model: {issue_type}",
                "suggestion": "Review and address this issue",
                "code_snippet": code_text[:100],  # First 100 chars
            })
        return issues

    def _get_severity_breakdown(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of issues by severity."""
        breakdown = {"low": 0, "medium": 0, "high": 0}
        for issue in issues:
            severity = issue.get("severity", "medium")
            if severity in breakdown:
                breakdown[severity] += 1
        return breakdown

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "rule_based": True,
            "transformer_available": self.transformer_model is not None,
            "baseline_ml_available": self.baseline_model is not None,
            "transformer_enabled": self.use_transformers,
            "baseline_ml_enabled": self.use_baseline_ml,
            "recommended_strategy": "weighted_average" if (
                self.transformer_model or self.baseline_model
            ) else "rule_only",
        }
