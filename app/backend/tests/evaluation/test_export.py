import json

import pytest

from evaluation.export import (
    build_query_payload,
    build_run_metadata,
    export_rankings,
    select_cases,
    _corpus_sha256,
)


def _case(case_id: str, task_type: str = "KIS") -> dict:
    return {
        "case_id": case_id,
        "task_type": task_type,
        "query": {"texts": [f"query for {case_id}"]},
        "relevant_frames": [{"video_id": "L21_V001", "frame_id": "100", "relevance": 2}],
    }


def _metadata() -> dict[str, str]:
    return build_run_metadata(
        "run-1",
        {
            "corpus_version": "2026-08-25",
            "corpus_sha256": "a" * 64,
            "index_config_sha256": "b" * 64,
            "encoder": "fg-clip2-large",
            "encoder_revision": "model-revision",
            "preprocessing_version": "rgb-v1",
            "query_strategy": "translated-text",
        },
        collection="BoldSearch",
    )


def _fake_post(responses: dict[str, list[dict]]) -> object:
    def post(path: str, payload: dict) -> dict:
        return {"results": responses[payload["query"]]}

    return post


def test_export_rankings_writes_run_folder_and_scores_the_export(tmp_path) -> None:
    cases = [_case("kis-001"), _case("kis-002")]
    responses = {
        "query for kis-001": [
            {"videoId": "L21_V999", "frameId": "1"},
            {"video_id": "L21_V001", "frame_id": "100"},
        ],
        "query for kis-002": [{"video_id": "L21_V001", "frame_id": "100"}],
    }

    report = export_rankings(
        cases,
        output_dir=tmp_path / "run-1",
        metadata=_metadata(),
        post_json=_fake_post(responses),
    )

    assert report["overall"]["recall_at_1"] == 0.5
    assert report["overall"]["recall_at_5"] == 1.0
    assert (tmp_path / "run-1" / "tasks.jsonl").is_file()
    assert (tmp_path / "run-1" / "rankings.jsonl").is_file()
    assert (tmp_path / "run-1" / "report.json").is_file()
    assert (tmp_path / "run-1" / "report.md").is_file()
    metadata = json.loads((tmp_path / "run-1" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run-1"
    assert metadata["collection"] == "BoldSearch"
    rankings = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rankings[0]["task_id"] == "kis-001"
    assert rankings[0]["results"] == [
        {"video_id": "L21_V999", "frame_id": "1"},
        {"video_id": "L21_V001", "frame_id": "100"},
    ]


def test_export_rankings_refuses_an_existing_run_folder(tmp_path) -> None:
    (tmp_path / "run-1").mkdir()

    with pytest.raises(ValueError, match="run folder already exists"):
        export_rankings(
            [_case("kis-001")],
            output_dir=tmp_path / "run-1",
            metadata=_metadata(),
            post_json=_fake_post({}),
        )


def test_select_cases_defaults_to_all_and_filters_in_requested_order() -> None:
    cases = [_case("kis-001"), _case("kis-002"), _case("trake-001", "TRAKE")]

    assert [case["case_id"] for case in select_cases(cases)] == [
        "kis-001",
        "kis-002",
        "trake-001",
    ]
    assert [case["case_id"] for case in select_cases(cases, "trake-001, kis-001")] == [
        "trake-001",
        "kis-001",
    ]

    with pytest.raises(ValueError, match="unknown case_id"):
        select_cases(cases, "kis-001,missing-001")
    with pytest.raises(ValueError, match="at least one case_id"):
        select_cases(cases, " , ")


def test_build_query_payload_preserves_ordered_texts_and_task_type() -> None:
    payload = build_query_payload(
        {
            "case_id": "trake-001",
            "task_type": "trake",
            "query": {"texts": ["first event", "second event"]},
        }
    )

    assert payload == {
        "query": "first event",
        "queries": ["first event", "second event"],
        "task": "TRAKE",
    }

    with pytest.raises(ValueError, match="requires query.texts"):
        build_query_payload({"case_id": "empty-001", "query": {"texts": []}})


def test_corpus_sha256_is_deterministic_and_tracks_corpus_changes(tmp_path) -> None:
    (tmp_path / "L21_V001").mkdir()
    (tmp_path / "L21_V001" / "100.jpg").write_bytes(b"frame-100")

    first = _corpus_sha256(tmp_path)
    assert first == _corpus_sha256(tmp_path)

    (tmp_path / "L21_V001" / "104.jpg").write_bytes(b"frame-104")
    assert _corpus_sha256(tmp_path) != first

    with pytest.raises(ValueError, match="keyframes directory not found"):
        _corpus_sha256(tmp_path / "missing")
