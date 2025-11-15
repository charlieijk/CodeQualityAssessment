"""
Utilities for turning raw code/text/screenshots into labeled training data.
"""
from __future__ import annotations

import argparse
import ast
import json
import textwrap
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from src.models.code_analyzer import CodeQualityAnalyzer
from src.ocr.image_preprocessor import ImagePreprocessor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}


@dataclass
class DatasetEntry:
    """Serializable representation of a single training example."""

    id: str
    source_type: str  # 'text' or 'image'
    source_path: str
    text: str
    issues: List[Dict[str, Any]]
    quality_score: float
    severity_breakdown: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBuilderConfig:
    """Runtime configuration for dataset creation."""

    raw_image_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    output_filename: str = "dataset.jsonl"
    min_lines: int = 3
    max_lines: int = 120
    max_snippets_per_file: Optional[int] = 20
    include_images: bool = True

    @property
    def output_path(self) -> Path:
        return self.processed_dir / self.output_filename


class CodeSnippetExtractor:
    """Extracts meaningful code samples from raw Python modules."""

    def __init__(self, min_lines: int = 3, max_lines: int = 120):
        self.min_lines = min_lines
        self.max_lines = max_lines

    def extract(self, code_text: str) -> List[Dict[str, Any]]:
        snippets: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return [{"type": "file", "text": code_text.strip()}]

        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            source_segment = ast.get_source_segment(code_text, node)
            if not source_segment:
                continue

            line_count = self._estimate_line_count(node, source_segment)
            if line_count < self.min_lines:
                continue
            if self.max_lines and line_count > self.max_lines:
                source_segment = self._truncate_segment(source_segment)

            snippets.append(
                {
                    "type": self._node_type(node),
                    "text": source_segment.strip("\n"),
                    "line_span": (getattr(node, "lineno", None), getattr(node, "end_lineno", None)),
                }
            )

        if not snippets:
            snippets.append({"type": "file", "text": code_text.strip(), "line_span": None})

        return snippets

    def _estimate_line_count(self, node: ast.AST, source_segment: str) -> int:
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            return max(1, node.end_lineno - node.lineno + 1)  # type: ignore[attr-defined]
        return source_segment.count("\n") + 1

    def _truncate_segment(self, segment: str) -> str:
        lines = segment.splitlines()
        truncated = lines[: self.max_lines]
        truncated.append("    # ... truncated for dataset generation ...")
        return "\n".join(truncated)

    def _node_type(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            return "class"
        if isinstance(node, ast.FunctionDef):
            return "function"
        if isinstance(node, ast.AsyncFunctionDef):
            return "async_function"
        return type(node).__name__


class DatasetBuilder:
    """Main orchestration class for weak-label dataset creation."""

    def __init__(
        self,
        config: Optional[DatasetBuilderConfig] = None,
        analyzer: Optional[CodeQualityAnalyzer] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.config = config or DatasetBuilderConfig()
        self.analyzer = analyzer or CodeQualityAnalyzer()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.extractor = CodeSnippetExtractor(
            min_lines=self.config.min_lines, max_lines=self.config.max_lines
        )

    def build_dataset(
        self,
        code_dirs: Sequence[Path],
        limit: Optional[int] = None,
        include_images: Optional[bool] = None,
    ) -> List[DatasetEntry]:
        include_images = (
            self.config.include_images if include_images is None else include_images
        )
        entries: List[DatasetEntry] = []
        entries.extend(self._process_code_directories(code_dirs, limit=limit))
        if include_images:
            entries.extend(self._process_image_directory(limit=limit))
        return entries

    def _process_code_directories(
        self, code_dirs: Sequence[Path], limit: Optional[int]
    ) -> List[DatasetEntry]:
        code_files = self._collect_code_files(code_dirs)
        entries: List[DatasetEntry] = []
        sample_count = 0

        for file_path in tqdm(code_files, desc="Analyzing source files", unit="file"):
            code_text = file_path.read_text(encoding="utf-8", errors="ignore")
            snippets = self.extractor.extract(code_text)

            if (
                self.config.max_snippets_per_file
                and len(snippets) > self.config.max_snippets_per_file
            ):
                snippets = snippets[: self.config.max_snippets_per_file]

            for snippet_index, snippet in enumerate(snippets):
                entry = self._create_entry(
                    text=snippet["text"],
                    source_type="text",
                    source_path=str(file_path),
                    snippet_type=snippet.get("type", "unknown"),
                    metadata_extra={
                        "line_span": snippet.get("line_span"),
                        "snippet_index": snippet_index,
                        "source_size": len(code_text.splitlines()),
                    },
                )
                entries.append(entry)
                sample_count += 1
                if limit and sample_count >= limit:
                    return entries

        return entries

    def _process_image_directory(self, limit: Optional[int]) -> List[DatasetEntry]:
        raw_dir = self.config.raw_image_dir
        if not raw_dir.exists():
            return []

        entries: List[DatasetEntry] = []
        image_files = sorted(
            [
                p
                for p in raw_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        for file_path in tqdm(image_files, desc="Processing screenshots", unit="image"):
            result = self.preprocessor.process_code_image(str(file_path))
            text = result.get("extracted_text", "").strip()
            if not text:
                continue

            entry = self._create_entry(
                text=text,
                source_type="image",
                source_path=str(file_path),
                snippet_type="ocr",
                metadata_extra={
                    "code_regions": result.get("code_regions", []),
                    "ocr_success": result.get("success", False),
                },
            )
            entries.append(entry)
            if limit and len(entries) >= limit:
                break

        return entries

    def _collect_code_files(self, code_dirs: Sequence[Path]) -> List[Path]:
        code_files: List[Path] = []
        for directory in code_dirs:
            directory = directory.expanduser().resolve()
            if directory.is_file() and directory.suffix == ".py":
                code_files.append(directory)
                continue
            if not directory.exists():
                continue
            code_files.extend(sorted(directory.rglob("*.py")))
        return code_files

    def _create_entry(
        self,
        text: str,
        source_type: str,
        source_path: str,
        snippet_type: str,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> DatasetEntry:
        analysis_result = self.analyzer.analyze_code(text)
        timestamp = datetime.utcnow().isoformat() + "Z"
        metadata = {
            "snippet_type": snippet_type,
            "text_length": len(text.splitlines()),
            "timestamp": timestamp,
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        return DatasetEntry(
            id=str(uuid.uuid4()),
            source_type=source_type,
            source_path=source_path,
            text=text,
            issues=analysis_result.get("issues", []),
            quality_score=analysis_result.get("quality_score", 0.0),
            severity_breakdown=analysis_result.get("severity_breakdown", {}),
            metadata=metadata,
        )

    def save(self, entries: Iterable[DatasetEntry], output_path: Optional[Path] = None) -> Path:
        output_path = output_path or self.config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(asdict(entry), ensure_ascii=False))
                fh.write("\n")
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate weakly labeled datasets from source code and screenshots."
    )
    parser.add_argument(
        "--code-dirs",
        nargs="+",
        default=["src"],
        help="Directories containing Python code to sample (default: src).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination JSONL path (default: data/processed/dataset.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of snippets to generate (default: unlimited).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Disable screenshot processing.",
    )
    parser.add_argument(
        "--max-snippets-per-file",
        type=int,
        default=20,
        help="Clamp snippet count per file to this number.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetBuilderConfig(max_snippets_per_file=args.max_snippets_per_file)
    builder = DatasetBuilder(config=config)

    code_dirs = [Path(p) for p in args.code_dirs]
    entries = builder.build_dataset(
        code_dirs=code_dirs,
        limit=args.limit,
        include_images=not args.skip_images,
    )
    output_path = (
        Path(args.output).expanduser().resolve() if args.output else config.output_path
    )
    builder.save(entries, output_path=output_path)

    summary = textwrap.dedent(
        f"""
        Dataset generation complete.
        Samples written: {len(entries)}
        Output: {output_path}
        """
    ).strip()
    print(summary)


if __name__ == "__main__":
    main()
