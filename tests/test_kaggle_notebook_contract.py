import json
from pathlib import Path


def test_kaggle_run_all_notebook_uses_mp4_runtime_branch() -> None:
    notebook = Path("notebooks/kaggle_mp4_run_all.ipynb")
    document = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"]
    )

    assert document["nbformat"] == 4
    assert 'RUNTIME_REPO_REF = "feat/kaggle-mp4-run-all"' in source
    assert "APP_REPO_ROOT" in source and "RUNTIME_REPO_ROOT" in source
    assert "boldsearch_integration.cli" in source and "'run'" in source
    assert "'bootstrap'" in source and "'ingest'" in source
    assert "boldsearch_integration.fastapi_launcher" in source
    assert "boldsearch_integration.gateway" in source
    assert "start_quick_tunnel" in source
    assert "KEYFRAME_ROOT" not in source
    for index, cell in enumerate(document["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{notebook}:cell-{index}", "exec")
