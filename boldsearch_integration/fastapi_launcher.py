"""Run the archived BoldSearch FastAPI app with a source-safe search overlay.

The archive's legacy service always submits both visual and caption ANN
requests.  A V1 MP4 corpus contains only ``visual_embedding``.  This launcher
imports the archived app unchanged, then replaces only the in-memory search
function so a visual-only collection can be served without editing the clone.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .search_modality import request_kwargs, select_search_modalities


def _schema_fields(client: Any, collection_name: str) -> set[str]:
    description = client.describe_collection(collection_name)
    fields = description.get("fields") if isinstance(description, Mapping) else None
    if not isinstance(fields, list):
        raise RuntimeError("Milvus collection description has no fields")
    result: set[str] = set()
    for field in fields:
        if not isinstance(field, Mapping):
            continue
        name = field.get("name") or field.get("field_name")
        if isinstance(name, str) and name:
            result.add(name)
    return result


def _output_fields(config: Any, available: set[str]) -> list[str]:
    configured = str(getattr(config, "MILVUS_OUTPUT_FIELDS", "")).split(",")
    fields = [item.strip() for item in configured if item.strip() and item.strip() in available]
    for required in ("frame_id", "shot_id", "video_id", "thumbnail"):
        if required in available and required not in fields:
            fields.append(required)
    return fields


def patch_visual_search(service_module: Any) -> None:
    """Patch an imported archived ``search.service`` module in memory."""

    schema_cache: dict[tuple[int, str], set[str]] = {}

    def visual_safe_hybrid_search(
        config: Any,
        client: Any,
        query_text: str | None,
        query_embedding: Any,
        top_k: int,
        expr: str | None = None,
    ) -> dict[str, Any]:
        del query_text
        if client is None:
            raise RuntimeError("ZILLIZ_URI is not configured")
        if query_embedding is None:
            raise ValueError("query embedding is required")
        try:
            from pymilvus import AnnSearchRequest, WeightedRanker
        except ImportError as exc:
            raise RuntimeError("pymilvus is required for Zilliz search") from exc

        collection = str(config.MILVUS_COLLECTION)
        cache_key = (id(client), collection)
        available = schema_cache.get(cache_key)
        if available is None:
            available = _schema_fields(client, collection)
            schema_cache[cache_key] = available

        requested = os.environ.get("BOLDSEARCH_SEARCH_MODALITIES", "visual")
        weight_values = [item.strip() for item in str(
            getattr(config, "MILVUS_RANKER_WEIGHTS", "1.0")
        ).split(",") if item.strip()]
        weights = {
            name: float(value)
            for name, value in zip(("visual", "caption"), weight_values)
        }
        modalities = select_search_modalities(
            requested, available, has_query_embedding=True, weights=weights
        )
        specs = [
            AnnSearchRequest(**request_kwargs(
                modality, query_embedding=list(query_embedding),
                top_k=top_k, expr=expr,
            ))
            for modality in modalities
        ]
        output_fields = _output_fields(config, available)

        if len(specs) == 1:
            raw = client.hybrid_search(
                collection_name=collection,
                reqs=specs,
                ranker=WeightedRanker(modalities[0].weight),
                limit=top_k,
                output_fields=output_fields,
            )
        else:
            raw = client.hybrid_search(
                collection_name=collection,
                reqs=specs,
                ranker=WeightedRanker(*(item.weight for item in modalities)),
                limit=top_k,
                output_fields=output_fields,
            )
        if isinstance(raw, dict):
            return raw
        flatten = getattr(service_module, "_flatten_milvus_hits", None)
        data = flatten(raw) if callable(flatten) else raw
        return {"code": 0, "data": data, "topks": [top_k]}

    service_module._hybrid_search = visual_safe_hybrid_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args(argv)
    root = args.app_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--app-root is not a directory: {root}")
    sys.path.insert(0, str(root))
    try:
        import main as archived_main
        from search import service
        import uvicorn
    except ImportError as exc:
        raise SystemExit(f"cannot import archived BoldSearch app: {exc}") from exc
    patch_visual_search(service)
    uvicorn.run(archived_main.app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
