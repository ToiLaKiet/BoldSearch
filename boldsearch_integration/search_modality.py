"""Config-driven Milvus search modality selection.

The MP4 pipeline publishes one embedding per frame: ``visual_embedding``.
This module keeps the search adapter honest by selecting only fields that are
both requested and present in the collection schema.  In particular, it never
silently reuses a visual vector as a caption vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any


@dataclass(frozen=True)
class SearchModality:
    """One ANN request specification, independent of pymilvus."""

    name: str
    anns_field: str
    metric_type: str = "COSINE"
    weight: float = 1.0


_FIELD_BY_NAME = {
    "visual": "visual_embedding",
    "caption": "caption_embedding",
}


def select_search_modalities(
    requested: str | Iterable[str] | None,
    available_fields: Iterable[str],
    *,
    has_query_embedding: bool,
    weights: Mapping[str, float] | None = None,
) -> tuple[SearchModality, ...]:
    """Resolve a safe, deterministic list of ANN modalities.

    ``requested`` accepts ``"visual"`` or a comma-separated/list form such as
    ``("visual", "caption")``.  An omitted value defaults to visual-only,
    which is the contract emitted by ``aic_video_pipeline_v1``.  Caption is
    allowed only when the collection really has ``caption_embedding`` and the
    caller explicitly opts in.
    """

    fields = {str(field) for field in available_fields}
    if requested is None or (isinstance(requested, str) and not requested.strip()):
        names = ["visual"]
    elif isinstance(requested, str):
        names = [item.strip().lower() for item in requested.split(",") if item.strip()]
    else:
        names = [str(item).strip().lower() for item in requested if str(item).strip()]
    if not names:
        names = ["visual"]
    if not has_query_embedding:
        raise ValueError("visual/caption search requires a query embedding")

    unknown = sorted(set(names) - set(_FIELD_BY_NAME))
    if unknown:
        raise ValueError(f"unsupported search modalities: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate search modality")

    result: list[SearchModality] = []
    for name in names:
        field = _FIELD_BY_NAME[name]
        if field not in fields:
            raise ValueError(f"collection schema missing {field} for {name} modality")
        weight = float((weights or {}).get(name, 1.0))
        if weight <= 0:
            raise ValueError(f"search modality weight must be positive: {name}")
        result.append(SearchModality(name=name, anns_field=field, weight=weight))
    return tuple(result)


def request_kwargs(
    modality: SearchModality,
    *,
    query_embedding: list[float],
    top_k: int,
    expr: str | None = None,
) -> dict[str, Any]:
    """Return kwargs suitable for ``pymilvus.AnnSearchRequest``."""

    if not query_embedding or top_k <= 0:
        raise ValueError("query_embedding and positive top_k are required")
    return {
        "data": [query_embedding],
        "anns_field": modality.anns_field,
        "param": {"metric_type": modality.metric_type, "params": {}},
        "limit": top_k,
        "expr": expr,
    }
