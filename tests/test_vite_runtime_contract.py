from pathlib import Path


def test_runtime_vite_config_is_source_preserving() -> None:
    config = Path("boldsearch_integration/vite.runtime.mjs").read_text(encoding="utf-8")
    assert "http://0.0.0.0:8000/api" in config
    assert "const API_BASE = '/api';" in config
    assert "BOLDSEARCH_FRONTEND_DIST" in config
    assert "emptyOutDir: true" in config
    assert "loading=\"lazy\"" in config
    assert "decoding=\"async\"" in config
