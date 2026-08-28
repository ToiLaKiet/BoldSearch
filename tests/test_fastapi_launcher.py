import sys
from types import ModuleType, SimpleNamespace

import pytest

from boldsearch_integration.fastapi_launcher import (
    _output_fields,
    _schema_fields,
    patch_visual_search,
)


def test_schema_and_output_field_projection_are_safe() -> None:
    client = SimpleNamespace(describe_collection=lambda _name: {
        "fields": [
            {"name": "id"}, {"name": "video_id"}, {"name": "frame_id"},
            {"name": "shot_id"}, {"name": "visual_embedding"},
            {"name": "thumbnail"},
        ]
    })
    fields = _schema_fields(client, "BoldSearchV1")
    assert "caption_embedding" not in fields
    config = SimpleNamespace(
        MILVUS_OUTPUT_FIELDS="frame_id,shot_id,video_id,caption_embedding,thumbnail"
    )
    assert _output_fields(config, fields) == [
        "frame_id", "shot_id", "video_id", "thumbnail"
    ]


def test_overlay_sends_visual_request_only(monkeypatch) -> None:
    calls = []

    class AnnSearchRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class WeightedRanker:
        def __init__(self, *weights):
            self.weights = weights

    fake_pymilvus = ModuleType("pymilvus")
    fake_pymilvus.AnnSearchRequest = AnnSearchRequest
    fake_pymilvus.WeightedRanker = WeightedRanker
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    class Client:
        def describe_collection(self, _name):
            return {"fields": [
                {"name": "video_id"}, {"name": "frame_id"},
                {"name": "shot_id"}, {"name": "visual_embedding"},
            ]}

        def hybrid_search(self, **kwargs):
            calls.append(kwargs)
            return [[{"id": 1, "entity": {"video_id": "L21_V001"}}]]

    service = SimpleNamespace(_flatten_milvus_hits=lambda raw: raw)
    patch_visual_search(service)
    config = SimpleNamespace(
        MILVUS_COLLECTION="BoldSearchV1",
        MILVUS_OUTPUT_FIELDS="frame_id,shot_id,video_id,caption_embedding",
        MILVUS_RANKER_WEIGHTS="1.0",
    )
    result = service._hybrid_search(config, Client(), None, [0.1, 0.2], 5)
    assert result["code"] == 0
    request = calls[0]["reqs"][0].kwargs
    assert request["anns_field"] == "visual_embedding"
    assert calls[0]["ranker"].weights == (1.0,)

    monkeypatch.setenv("BOLDSEARCH_SEARCH_MODALITIES", "visual,caption")
    with pytest.raises(ValueError, match="caption_embedding"):
        service._hybrid_search(config, Client(), None, [0.1, 0.2], 5)
