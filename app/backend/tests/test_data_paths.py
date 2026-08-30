from app_config import AppConfig
from app_config import app_config
from main import app


def test_default_data_paths_keep_static_media_and_evaluation_artifacts_together() -> None:
    config = AppConfig()

    assert config.KEYFRAMES_DIR == config.DATA_DIR / "keyframes"
    assert config.KEYFRAME_MAP_DIR == config.DATA_DIR / "map-keyframes"
    assert config.EVALUATION_ARTIFACT_DIR == config.DATA_DIR / "evaluation-artifacts"


def test_data_dir_override_re_roots_default_child_paths(tmp_path) -> None:
    config = AppConfig(_env_file=None, DATA_DIR=tmp_path)

    assert config.KEYFRAMES_DIR == tmp_path / "keyframes"
    assert config.KEYFRAME_MAP_DIR == tmp_path / "map-keyframes"
    assert config.EVALUATION_ARTIFACT_DIR == tmp_path / "evaluation-artifacts"


def test_flat_fields_reflect_environment_backed_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("SEARCH_TOP_K", "100")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    config = AppConfig(_env_file=None)

    assert config.PORT == 9000
    assert config.SEARCH_TOP_K == 100
    assert config.KEYFRAMES_DIR == tmp_path / "keyframes"


def test_fastapi_serves_configured_keyframe_and_map_prefixes() -> None:
    routes_by_path = {route.path: route for route in app.routes}

    assert routes_by_path["/keyframes"].app.directory == app_config.KEYFRAMES_DIR
    assert routes_by_path["/map-keyframes"].app.directory == app_config.KEYFRAME_MAP_DIR
