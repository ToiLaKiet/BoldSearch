from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from evaluation.metrics.ranking import normalize_k_values
from evaluation.report import render_markdown_report, render_summary_text
from evaluation.tasks.evidence_ranking import evaluate_evidence_rankings

_REQUIRED_METADATA_FIELDS = (
    "run_id",
    "corpus_version",
    "corpus_sha256",
    "collection",
    "index_config_sha256",
    "encoder",
    "encoder_revision",
    "preprocessing_version",
    "query_strategy",
)
_SHA256_METADATA_FIELDS = ("corpus_sha256", "index_config_sha256")


def run_evaluation(
    task_cards_path: Path,
    rankings_path: Path,
    report_path: Path,
    k_values: Sequence[int],
    metadata_path: Path | None = None,
    markdown_report_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate JSONL rankings against versioned task cards and save one report."""
    if metadata_path is None:
        raise ValueError("metadata path is required")
    _validate_output_paths(
        report_path,
        markdown_report_path,
        task_cards_path,
        rankings_path,
        metadata_path,
    )
    normalized_k_values = normalize_k_values(k_values)
    task_cards_content = task_cards_path.read_bytes()
    rankings_content = rankings_path.read_bytes()
    metadata_content = metadata_path.read_bytes()
    metadata = _read_json_object(metadata_path, metadata_content)
    _validate_metadata(metadata)
    report = {
        "schema_version": 1,
        "k": normalized_k_values,
        "metadata": metadata,
        "inputs": {
            "task_cards_sha256": _sha256(task_cards_content),
            "rankings_sha256": _sha256(rankings_content),
            "metadata_sha256": _sha256(metadata_content),
        },
        **evaluate_evidence_rankings(
            task_cards=_read_jsonl(task_cards_path, task_cards_content),
            rankings=_read_jsonl(rankings_path, rankings_content),
            k_values=normalized_k_values,
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_report_path:
        markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_report_path.write_text(render_markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    from app_config import app_config

    parser = argparse.ArgumentParser(description="Evaluate BoldSearch JSONL rankings.")
    parser.add_argument("--tasks", type=Path, required=True, help="Path to task cards JSONL.")
    parser.add_argument("--rankings", type=Path, required=True, help="Path to ranking results JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=app_config.EVALUATION_ARTIFACT_DIR / "report.json",
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=app_config.EVALUATION_ARTIFACT_DIR / "report.md",
        help="Optional path for a Markdown summary artifact.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="JSON object with required corpus, collection, encoder, query, and run identifiers.",
    )
    parser.add_argument(
        "--k",
        default="1,5,10",
        help="Comma-separated K values for Recall@K; nDCG uses the largest k.",
    )
    arguments = parser.parse_args()
    report = run_evaluation(
        task_cards_path=arguments.tasks,
        rankings_path=arguments.rankings,
        report_path=arguments.output,
        k_values=_parse_k_values(arguments.k),
        metadata_path=arguments.metadata,
        markdown_report_path=arguments.markdown_output,
    )
    print(render_summary_text(report))


def _read_jsonl(path: Path, content: bytes) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record in {path} at line {line_number} must be an object")
        records.append(record)
    return records


def _read_json_object(path: Path, content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"metadata in {path} must be a JSON object")
    return value


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing_fields = [
        field
        for field in _REQUIRED_METADATA_FIELDS
        if not isinstance(metadata.get(field), str) or not metadata[field].strip()
    ]
    if missing_fields:
        raise ValueError(f"metadata requires non-empty fields: {', '.join(missing_fields)}")
    malformed_hashes = [
        field
        for field in _SHA256_METADATA_FIELDS
        if not re.fullmatch(r"[0-9a-fA-F]{64}", metadata[field])
    ]
    if malformed_hashes:
        raise ValueError(f"{malformed_hashes[0]} must be a SHA-256 hex digest")


def _validate_output_paths(
    report_path: Path,
    markdown_report_path: Path | None,
    task_cards_path: Path,
    rankings_path: Path,
    metadata_path: Path | None,
) -> None:
    output_paths = [report_path]
    if markdown_report_path:
        output_paths.append(markdown_report_path)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("output paths must be distinct")
    input_paths = [task_cards_path, rankings_path]
    if metadata_path:
        input_paths.append(metadata_path)
    input_path_set = {path.resolve() for path in input_paths}
    if any(path.resolve() in input_path_set for path in output_paths):
        raise ValueError("output path must not overwrite an input")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parse_k_values(raw_k_values: str) -> list[int]:
    try:
        k_values = [int(item.strip()) for item in raw_k_values.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("k values must be comma-separated positive integers") from error
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k values must be comma-separated positive integers")
    return k_values


if __name__ == "__main__":
    main()
