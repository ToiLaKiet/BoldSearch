"""Export rankings from the live search API and evaluate them in one command.

Reads versioned case files (``case_id`` vocabulary), queries the running
backend, and writes an immutable run folder under ``evaluation/runs/<run-id>/``
with ``tasks.jsonl``/``rankings.jsonl``/``metadata.json`` plus the scored
reports from :func:`evaluation.runner.run_evaluation`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app_config import app_config
from evaluation.report import render_summary_text
from evaluation.runner import run_evaluation

CASES_DIR = Path(__file__).resolve().parent / "cases"
RUNS_DIR = Path(__file__).resolve().parent / "runs"

DEFAULT_K_VALUES = (1, 5, 10)

Poster = Callable[[str, dict[str, Any]], dict[str, Any]]


def load_cases(cases_dir: Path = CASES_DIR) -> list[dict[str, Any]]:
    """Load every JSONL case file under the cases directory, keyed by case_id."""
    case_files = sorted(cases_dir.rglob("*.jsonl"))
    if not case_files:
        raise ValueError(f"no case files found under {cases_dir}")
    cases: dict[str, dict[str, Any]] = {}
    for path in case_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"{path} line {line_number} requires a non-empty case_id")
            if case_id in cases:
                raise ValueError(f"duplicate case_id {case_id} in {path}")
            cases[case_id] = case
    return list(cases.values())


def select_cases(
    cases: Sequence[Mapping[str, Any]], requested: str | None = None
) -> list[Mapping[str, Any]]:
    """Return all cases, or the requested comma-separated case_ids in order."""
    if requested is None:
        return list(cases)
    requested_ids = list(dict.fromkeys(item.strip() for item in requested.split(",") if item.strip()))
    if not requested_ids:
        raise ValueError("--cases requires at least one case_id")
    by_id = {str(case["case_id"]): case for case in cases}
    unknown = [case_id for case_id in requested_ids if case_id not in by_id]
    if unknown:
        raise ValueError(f"unknown case_id(s): {', '.join(unknown)}")
    return [by_id[case_id] for case_id in requested_ids]


def build_query_payload(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build a POST /api/search/query body from one case's ordered text queries."""
    texts = [str(text).strip() for text in case.get("query", {}).get("texts", []) if str(text).strip()]
    if not texts:
        raise ValueError(f"case {case.get('case_id')} requires query.texts")
    return {
        "query": texts[0],
        "queries": texts,
        "task": str(case.get("task_type") or "KIS").upper(),
    }


def build_run_metadata(
    run_id: str, corpus_metadata: Mapping[str, str], collection: str
) -> dict[str, str]:
    metadata = dict(corpus_metadata)
    metadata["run_id"] = run_id
    metadata["collection"] = collection
    return metadata


def export_rankings(
    cases: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    metadata: Mapping[str, str],
    post_json: Poster,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Query the API per case, write the run folder, and return the evaluation report."""
    if output_dir.exists():
        raise ValueError(f"run folder already exists: {output_dir} — pick a new run_id")
    task_cards: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        relevant_frames = case.get("relevant_frames")
        if not isinstance(relevant_frames, list) or not relevant_frames:
            raise ValueError(f"case {case_id} requires relevant_frames")
        response = post_json("/search/query", build_query_payload(case))
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list):
            raise ValueError(f"search response for case {case_id} requires a results list")
        rankings.append({"task_id": case_id, "results": [_frame_candidate(item) for item in results]})
        task_cards.append(
            {
                "task_id": case_id,
                "task_type": str(case.get("task_type") or "KIS").upper(),
                "relevant_frames": relevant_frames,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "tasks.jsonl", task_cards)
    _write_jsonl(output_dir / "rankings.jsonl", rankings)
    (output_dir / "metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_evaluation(
        task_cards_path=output_dir / "tasks.jsonl",
        rankings_path=output_dir / "rankings.jsonl",
        report_path=output_dir / "report.json",
        k_values=list(k_values),
        metadata_path=output_dir / "metadata.json",
        markdown_report_path=output_dir / "report.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export rankings from the live search API and evaluate them."
    )
    parser.add_argument(
        "--run-id", required=True, help="Unique run name; becomes the folder under evaluation/runs/."
    )
    parser.add_argument(
        "--cases", default=None, help="Comma-separated case_ids to run; defaults to all cases."
    )
    arguments = parser.parse_args()

    base_url = _base_url()
    try:
        cases = select_cases(load_cases(), arguments.cases)
        corpus_metadata = {
            **_corpus_metadata(),
            "corpus_sha256": _corpus_sha256(),
            "index_config_sha256": _index_config_sha256(),
            **_encoder_identity(),
        }
        metadata = build_run_metadata(
            arguments.run_id, corpus_metadata, app_config.MILVUS_COLLECTION
        )
        report = export_rankings(
            cases, output_dir=RUNS_DIR / arguments.run_id, metadata=metadata, post_json=_post_json(base_url)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    except OSError as error:
        raise SystemExit(
            f"{error} — is the backend serving at {base_url}? start it with: uv run python main.py"
        ) from error
    print(render_summary_text(report))


def _base_url() -> str:
    host = "localhost" if app_config.HOST in ("", "0.0.0.0") else app_config.HOST
    return f"http://{host}:{app_config.PORT}{app_config.API_PREFIX}"


def _encoder_identity() -> dict[str, str]:
    from encoders.fg_clip import MODEL_ID, MODEL_REVISION

    return {"encoder": MODEL_ID, "encoder_revision": MODEL_REVISION}


def _corpus_sha256(keyframes_dir: Path | None = None) -> str:
    """Fingerprint the keyframes corpus from sorted relative paths and sizes."""
    root = keyframes_dir or app_config.KEYFRAMES_DIR
    if not root.is_dir():
        raise ValueError(f"keyframes directory not found: {root}")
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(f"{path.relative_to(root)}\0{path.stat().st_size}\0".encode())
    return digest.hexdigest()


_MILVUS_CONFIG_FIELDS = (
    "MILVUS_COLLECTION",
    "MILVUS_VECTOR_FIELD",
    "MILVUS_OUTPUT_FIELDS",
    "MILVUS_TEXT_SEARCH_PARAMS",
    "MILVUS_VECTOR_SEARCH_PARAMS",
    "MILVUS_RANKER_WEIGHTS",
)


def _index_config_sha256() -> str:
    payload = {
        field: str(getattr(app_config, field)) for field in _MILVUS_CONFIG_FIELDS
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _corpus_metadata() -> dict[str, str]:
    return {
        "corpus_version": app_config.EVALUATION_CORPUS_VERSION,
        "preprocessing_version": app_config.EVALUATION_PREPROCESSING_VERSION,
        "query_strategy": app_config.EVALUATION_QUERY_STRATEGY,
    }


def _frame_candidate(result: Mapping[str, Any]) -> dict[str, str]:
    video_id = result.get("video_id") or result.get("videoId")
    frame_id = result.get("frame_id") or result.get("frameId")
    if not video_id or not frame_id:
        raise ValueError("search result requires video_id and frame_id")
    return {"video_id": str(video_id), "frame_id": str(frame_id)}


def _post_json(base_url: str) -> Poster:
    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    return post


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
