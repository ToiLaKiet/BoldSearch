import json
from pathlib import Path


def test_kaggle_query_only_notebook_does_not_build_or_ingest_a_corpus() -> None:
    notebook = Path("notebooks/kaggle_query_existing_index.ipynb")
    document = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in document["cells"])

    assert document["nbformat"] == 4
    assert "BOLDSEARCH_VIDEO_MANIFEST" in source
    assert "BOLDSEARCH_FRAME_CACHE_ROOT" in source
    assert "ENV_INPUT_FILENAME = 'a.env'" in source
    assert "load_a_env" in source
    assert "kaggle_secrets" not in source
    assert "boldsearch_integration.video_frames" in source
    assert "boldsearch_integration.fastapi_launcher" in source
    assert "boldsearch_integration.gateway" in source
    assert "start_quick_tunnel" in source
    assert "boldsearch_integration.cli', 'run'" not in source
    assert "'bootstrap'" not in source
    assert "'ingest'" not in source
    assert "aic_video_pipeline_v1" not in source
    for index, cell in enumerate(document["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{notebook}:cell-{index}", "exec")
