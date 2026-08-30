from __future__ import annotations

import html
import re
from typing import Any, Mapping, Sequence


def render_summary_text(report: Mapping[str, Any]) -> str:
    """Fixed-width summary — no dependency, deterministic, copy-paste friendly."""
    k_values = _k_values(report)
    headers = ["Scope", *_metric_headers(k_values)]
    rows = [[scope, *_summary_cells(summary, k_values)] for scope, summary in _scopes(report)]
    widths = [max(map(len, column)) for column in zip(headers, *rows)]

    def pad(cells: list[str]) -> str:
        return "  ".join(
            cell.rjust(width) if index else cell.ljust(width)
            for index, (cell, width) in enumerate(zip(cells, widths))
        )

    return "\n".join([pad(headers), pad(["-" * width for width in widths]), *(pad(row) for row in rows)]) + "\n"


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a portable Markdown artifact from an evaluation report."""
    k_values = _k_values(report)
    headers = ["Scope", *_metric_headers(k_values)]
    lines = [
        "# BoldSearch Evaluation Report",
        "",
        "## Summary",
        "",
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
    ]
    for scope, summary in _scopes(report):
        cells = " | ".join(_summary_cells(summary, k_values))
        lines.append(f"| {_escape_table_cell(scope)} | {cells} |")

    lines.extend(["", "## Provenance", ""])
    lines.extend(_markdown_definitions(report.get("metadata", {})))
    if report.get("inputs"):
        lines.extend(["", "## Inputs", ""])
        lines.extend(_markdown_definitions(report["inputs"]))
    return "\n".join(lines) + "\n"


def _scopes(report: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = [("Overall", report["overall"])]
    rows.extend(sorted((str(t), s) for t, s in report.get("by_task_type", {}).items()))
    return rows


def _metric_headers(k_values: Sequence[int]) -> list[str]:
    return [
        "Tasks",
        *(f"Recall@{k}" for k in k_values),
        "MRR",
        f"nDCG@{k_values[-1]}",
    ]


def _k_values(report: Mapping[str, Any]) -> list[int]:
    return [int(k) for k in report["k"]]


def _summary_cells(summary: Mapping[str, Any], k_values: Sequence[int]) -> list[str]:
    values = [str(summary["task_count"])]
    values.extend(_format_metric(summary[f"recall_at_{k}"]) for k in k_values)
    values.append(_format_metric(summary["mrr"]))
    values.append(_format_metric(summary[f"ndcg_at_{k_values[-1]}"]))
    return values


def _markdown_definitions(pairs: Mapping[str, Any]) -> list[str]:
    return [f"- {_code_span(key)}: {_code_span(value)}" for key, value in sorted(pairs.items())]


def _format_metric(value: Any) -> str:
    return f"{float(value):.4f}"


def _escape_table_cell(value: Any) -> str:
    return _normalized_markdown_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _code_span(value: Any) -> str:
    text = _normalized_markdown_text(value)
    longest_backtick_run = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    delimiter = "`" * (longest_backtick_run + 1)
    return f"{delimiter}{text}{delimiter}"


def _normalized_markdown_text(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return html.escape(text)
