import pytest

from boldsearch_integration.search_modality import (
    request_kwargs,
    select_search_modalities,
)


def test_visual_only_is_the_safe_default_for_pipeline_rows() -> None:
    selected = select_search_modalities(
        None,
        {"id", "video_id", "frame_id", "shot_id", "visual_embedding"},
        has_query_embedding=True,
    )
    assert [(item.name, item.anns_field) for item in selected] == [
        ("visual", "visual_embedding")
    ]


def test_caption_requires_explicit_opt_in_and_schema_field() -> None:
    with pytest.raises(ValueError, match="caption_embedding"):
        select_search_modalities(
            "visual,caption",
            {"visual_embedding"},
            has_query_embedding=True,
        )

    selected = select_search_modalities(
        "visual,caption",
        {"visual_embedding", "caption_embedding"},
        has_query_embedding=True,
        weights={"visual": 0.8, "caption": 0.2},
    )
    assert [item.weight for item in selected] == [0.8, 0.2]


def test_invalid_modality_and_duplicate_are_rejected() -> None:
    fields = {"visual_embedding"}
    with pytest.raises(ValueError, match="unsupported"):
        select_search_modalities("ocr", fields, has_query_embedding=True)
    with pytest.raises(ValueError, match="duplicate"):
        select_search_modalities("visual,visual", fields, has_query_embedding=True)


def test_request_kwargs_has_no_hidden_caption_field() -> None:
    selected = select_search_modalities(
        "visual", {"visual_embedding"}, has_query_embedding=True
    )
    kwargs = request_kwargs(selected[0], query_embedding=[0.1, 0.2], top_k=5)
    assert kwargs["anns_field"] == "visual_embedding"
    assert kwargs["limit"] == 5
