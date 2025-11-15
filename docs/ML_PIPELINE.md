# ML Data & Training Pipeline

This document explains how to generate weakly labeled datasets from code/text/screenshots and how to train the baseline ML models that extend the rule-based analyzer.

## 1. Environment

1. Use Python 3.11 for the full ML stack.
2. Install core requirements plus the optional ML extras:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-ml-optional.txt
   ```
3. Make sure Tesseract OCR is installed (see `README.md` for platform-specific instructions) so screenshots can be processed.

## 2. Preparing Raw Data

Place inputs in the existing data folders:

- `data/raw/` – screenshots (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`) and any text snippets you want to keep as reference.
- `data/processed/` – automatically populated with JSONL datasets, trained models, and metrics.

You can also point the dataset builder at any other directory (e.g., cloned open-source projects) when generating snippets.

## 3. Dataset Generation

The dataset builder runs the production OCR and rule-based analyzer to produce weak labels that mimic runtime behavior.

```bash
python -m src.data_pipeline.dataset_builder \
  --code-dirs src tests \
  --output data/processed/dataset.jsonl \
  --limit 500
```

Flags:

- `--code-dirs`: one or more directories (or files) with Python sources.
- `--limit`: optional cap on the total number of snippets.
- `--skip-images`: ignore screenshots under `data/raw/`.
- `--max-snippets-per-file`: avoid flooding the dataset with a single large module.

Each line in the resulting JSONL file follows the schema:

```json
{
  "id": "uuid",
  "source_type": "text|image",
  "source_path": "path/to/file.py",
  "text": "def sample(...): ...",
  "issues": [
    {
      "issue_type": "missing_docstring",
      "severity": "low",
      "line_number": 10,
      "description": "...",
      "suggestion": "...",
      "code_snippet": "def sample(...):"
    }
  ],
  "quality_score": 72.0,
  "severity_breakdown": {"low": 1, "medium": 0, "high": 0},
  "metadata": {
    "snippet_type": "function",
    "text_length": 14,
    "timestamp": "...",
    "line_span": [8, 30]
  }
}
```

You can append multiple runs together (e.g., different repositories) by rerunning the builder and concatenating the JSONL files.

## 4. Training the Baseline Model

The trainer consumes the JSONL dataset, splits it into train/test folds, and fits:

1. A TF-IDF → Ridge regressor for the overall quality score.
2. A TF-IDF → One-vs-Rest logistic classifier for issue-type predictions.

```bash
python -m src.models.baseline_trainer \
  --dataset data/processed/dataset.jsonl \
  --model-dir data/processed/models \
  --test-size 0.2 \
  --issue-threshold 0.45
```

Outputs:

- `data/processed/models/baseline_quality_model.joblib` – serialized vectorizer + models + label binarizer.
- `data/processed/models/metrics.json` – MAE, multi-label F1 score (when applicable), and bookkeeping metadata.

The trainer automatically refits on the full dataset after evaluating the held-out split so the saved artifact uses all available samples.

## 5. Integrating ML Predictions

- Load the saved joblib via `BaselineQualityModel.load(...)` and call `predict(...)` with raw code text (after OCR if needed).
- Blend the ML predictions with the rule-based issues to experiment with ensemble strategies (e.g., overriding scores, flagging probable issue types not caught by heuristics).

## 6. Extending the Pipeline

- **Additional labels:** Plug linting tools (flake8, pylint, radon) into the dataset builder and merge their warnings into `QualityIssue` entries.
- **Screenshot augmentation:** Render curated snippets in multiple fonts/themes, store them under `data/raw/`, and regenerate the dataset to expose the OCR to more variation.
- **Model upgrades:** Swap the TF-IDF baseline for transformer encoders (CodeBERT, MiniLM) by replacing the vectorizer/regressor blocks inside `BaselineQualityModel`.

All steps are fully scriptable, enabling regular re-training as new labeled data becomes available.
