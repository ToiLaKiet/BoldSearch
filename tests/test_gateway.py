from pathlib import Path

import pytest

from boldsearch_integration.gateway import resolve_frame_request, resolve_static_path


def test_frame_request_maps_png_compatibility_url_to_webp(tmp_path: Path) -> None:
    release = tmp_path / "release"
    image = release / "keyframes/L21_V001/20.webp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"RIFFxxxxWEBP")

    path, content_type = resolve_frame_request(
        release, "/keyframes/L21_V001/20.png"
    )

    assert path == image
    assert content_type == "image/webp"


def test_frame_request_rejects_traversal_and_unknown_video_shape(tmp_path: Path) -> None:
    release = tmp_path / "release"
    with pytest.raises(FileNotFoundError):
        resolve_frame_request(release, "/keyframes/../../secret.png")
    with pytest.raises(FileNotFoundError):
        resolve_frame_request(release, "/keyframes/L21_V001/not-a-frame.png")


def test_static_path_stays_under_frontend_dist(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("ok", encoding="utf-8")
    assert resolve_static_path(dist, "/") == dist / "index.html"
    with pytest.raises(PermissionError):
        resolve_static_path(dist, "/../secret")
