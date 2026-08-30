import json
import sys
from hashlib import sha256

import pytest

from evaluation.runner import run_evaluation
from evaluation import runner


def test_run_evaluation_reads_jsonl_and_writes_a_reproducible_report(tmp_path) -> None:
    task_cards_path = tmp_path / "tasks.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "reports" / "baseline.json"
    markdown_report_path = tmp_path / "reports" / "baseline.md"
    task_cards_path.write_text(
        '{"task_id":"kis-001","task_type":"KIS","relevant_frames":'
        '[{"video_id":"L21_V001","frame_id":"100","relevance":2}]}\n',
        encoding="utf-8",
    )
    rankings_path.write_text(
        '{"task_id":"kis-001","results":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    metadata_path.write_text(
        '{"collection":"BoldSearch","corpus_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"corpus_version":"2026-08-25","encoder":"fg-clip2-large",'
        '"encoder_revision":"model-revision","index_config_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"preprocessing_version":"rgb-v1","query_strategy":"translated-text",'
        '"run_id":"baseline"}\n',
        encoding="utf-8",
    )

    report = run_evaluation(
        task_cards_path,
        rankings_path,
        report_path,
        k_values=[5, 1, 5],
        metadata_path=metadata_path,
        markdown_report_path=markdown_report_path,
    )

    assert report["schema_version"] == 1
    assert report["k"] == [1, 5]
    assert set(report) == {
        "schema_version",
        "k",
        "metadata",
        "inputs",
        "overall",
        "by_task_type",
        "per_task",
    }
    assert report["metadata"] == {
        "collection": "BoldSearch",
        "corpus_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "corpus_version": "2026-08-25",
        "encoder": "fg-clip2-large",
        "encoder_revision": "model-revision",
        "index_config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "preprocessing_version": "rgb-v1",
        "query_strategy": "translated-text",
        "run_id": "baseline",
    }
    assert report["inputs"] == {
        "task_cards_sha256": sha256(task_cards_path.read_bytes()).hexdigest(),
        "rankings_sha256": sha256(rankings_path.read_bytes()).hexdigest(),
        "metadata_sha256": sha256(metadata_path.read_bytes()).hexdigest(),
    }
    assert report["overall"]["recall_at_1"] == 1.0
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert "# BoldSearch Evaluation Report" in markdown_report_path.read_text(encoding="utf-8")


def test_run_evaluation_rejects_an_output_path_that_matches_an_input(tmp_path) -> None:
    task_cards_path = tmp_path / "tasks.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    metadata_path = tmp_path / "metadata.json"
    task_cards_path.write_text(
        '{"task_id":"kis-001","relevant_frames":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    rankings_path.write_text(
        '{"task_id":"kis-001","results":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    metadata_path.write_text(
        '{"collection":"BoldSearch","corpus_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"corpus_version":"2026-08-25","encoder":"fg-clip2-large",'
        '"encoder_revision":"model-revision","index_config_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"preprocessing_version":"rgb-v1","query_strategy":"translated-text",'
        '"run_id":"baseline"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output path must not overwrite an input"):
        run_evaluation(
            task_cards_path,
            rankings_path,
            rankings_path,
            k_values=[1],
            metadata_path=metadata_path,
        )


def test_run_evaluation_requires_comparable_run_metadata(tmp_path) -> None:
    task_cards_path = tmp_path / "tasks.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    report_path = tmp_path / "report.json"
    task_cards_path.write_text(
        '{"task_id":"kis-001","relevant_frames":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    rankings_path.write_text(
        '{"task_id":"kis-001","results":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata path is required"):
        run_evaluation(task_cards_path, rankings_path, report_path, k_values=[1])


def test_run_evaluation_rejects_metadata_without_immutable_versions(tmp_path) -> None:
    task_cards_path = tmp_path / "tasks.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "report.json"
    task_cards_path.write_text(
        '{"task_id":"kis-001","task_type":"KIS","relevant_frames":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    rankings_path.write_text(
        '{"task_id":"kis-001","results":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    metadata_path.write_text(
        '{"collection":"BoldSearch","corpus_version":"2026-08-25",'
        '"encoder":"fg-clip2-large","query_strategy":"translated-text",'
        '"run_id":"baseline"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corpus_sha256"):
        run_evaluation(
            task_cards_path,
            rankings_path,
            report_path,
            k_values=[1],
            metadata_path=metadata_path,
        )


def test_run_evaluation_rejects_malformed_provenance_hashes(tmp_path) -> None:
    task_cards_path = tmp_path / "tasks.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "report.json"
    task_cards_path.write_text(
        '{"task_id":"kis-001","task_type":"KIS","relevant_frames":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    rankings_path.write_text(
        '{"task_id":"kis-001","results":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "collection": "BoldSearch",
                "corpus_sha256": "not-a-sha",
                "corpus_version": "2026-08-25",
                "encoder": "fg-clip2-large",
                "encoder_revision": "model-revision",
                "index_config_sha256": "b" * 64,
                "preprocessing_version": "rgb-v1",
                "query_strategy": "translated-text",
                "run_id": "baseline",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corpus_sha256 must be a SHA-256"):
        run_evaluation(
            task_cards_path,
            rankings_path,
            report_path,
            k_values=[1],
            metadata_path=metadata_path,
        )


def test_runner_cli_renders_a_text_summary_and_writes_both_artifacts(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    task_cards_path = tmp_path / "tasks.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "report.json"
    markdown_report_path = tmp_path / "report.md"
    task_cards_path.write_text(
        '{"task_id":"kis-001","task_type":"KIS","relevant_frames":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    rankings_path.write_text(
        '{"task_id":"kis-001","results":'
        '[{"video_id":"L21_V001","frame_id":"100"}]}\n',
        encoding="utf-8",
    )
    metadata_path.write_text(
        '{"collection":"BoldSearch","corpus_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"corpus_version":"2026-08-25","encoder":"fg-clip2-large",'
        '"encoder_revision":"model-revision","index_config_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"preprocessing_version":"rgb-v1","query_strategy":"translated-text",'
        '"run_id":"baseline"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation.runner",
            "--tasks",
            str(task_cards_path),
            "--rankings",
            str(rankings_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(report_path),
            "--markdown-output",
            str(markdown_report_path),
        ],
    )

    runner.main()

    output = capsys.readouterr().out
    assert "Scope" in output
    assert "Recall@1" in output
    assert report_path.is_file()
    assert markdown_report_path.is_file()
