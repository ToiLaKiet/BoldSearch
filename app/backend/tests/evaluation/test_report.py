from evaluation.report import render_markdown_report, render_summary_text


def test_render_summary_text_renders_overall_and_task_type_metrics() -> None:
    text = render_summary_text(_report())

    assert "Recall@1" in text
    assert "0.5000" in text
    row = next(line for line in text.splitlines() if line.startswith("Overall"))
    assert row.split() == ["Overall", "2", "0.5000", "1.0000", "0.7500", "0.8000"]


def test_render_markdown_report_includes_provenance_and_summary_table() -> None:
    markdown = render_markdown_report(_report())

    assert "# BoldSearch Evaluation Report" in markdown
    assert "| Scope | Tasks | Recall@1 | Recall@5 | MRR | nDCG@5 |" in markdown
    assert "| Overall | 2 | 0.5000 | 1.0000 | 0.7500 | 0.8000 |" in markdown
    assert "## Provenance" in markdown
    assert "`encoder`: `fg-clip2-large`" in markdown


def test_render_markdown_report_escapes_dynamic_table_and_provenance_values() -> None:
    report = _report()
    report["by_task_type"] = {"KIS | <img src=x>": report["overall"]}
    report["metadata"] = {"run`id": "baseline\n## forged"}

    markdown = render_markdown_report(report)

    assert "| KIS \\| &lt;img src=x&gt; |" in markdown
    assert "``run`id``: `baseline ## forged`" in markdown
    assert "\n## forged" not in markdown
    assert "<img src=x>" not in markdown


def _report() -> dict:
    return {
        "k": [1, 5],
        "metadata": {"encoder": "fg-clip2-large", "run_id": "baseline"},
        "overall": {
            "task_count": 2,
            "recall_at_1": 0.5,
            "recall_at_5": 1.0,
            "mrr": 0.75,
            "ndcg_at_5": 0.8,
        },
        "by_task_type": {
            "KIS": {
                "task_count": 2,
                "recall_at_1": 0.5,
                "recall_at_5": 1.0,
                "mrr": 0.75,
                "ndcg_at_5": 0.8,
            }
        },
    }
