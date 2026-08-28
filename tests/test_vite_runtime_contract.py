from pathlib import Path
import zipfile


def test_runtime_vite_config_is_source_preserving() -> None:
    config = Path("boldsearch_integration/vite.runtime.mjs").read_text(encoding="utf-8")
    assert "http://0.0.0.0:8000/api" in config
    assert "const API_BASE = '/api';" in config
    assert "BOLDSEARCH_FRONTEND_DIST" in config
    assert "node_modules/@vitejs/plugin-react/dist/index.js" in config
    assert "emptyOutDir: true" in config
    assert "loading=\"lazy\"" in config
    assert "decoding=\"async\"" in config


def test_runtime_transform_matches_archived_frontend_contract() -> None:
    config = Path("boldsearch_integration/vite.runtime.mjs").read_text(encoding="utf-8")
    with zipfile.ZipFile("BoldSearch.zip") as archive:
        app = archive.read("BoldSearch/app/frontend/src/App.jsx").decode("utf-8")
    expected = [
        "const API_BASE = 'http://0.0.0.0:8000/api';",
        "<img src={thumbSrc} alt={`Frame ${f.frame_id}`} />",
        "<img src={imageSrc} alt={keyframe.title || `Frame ${frameId}`} />",
    ]
    for contract in expected:
        assert app.count(contract) == 1
        assert contract in config
