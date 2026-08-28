from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen
import json

import pytest

from boldsearch_integration.gateway import (
    make_handler,
    resolve_frame_request,
    resolve_static_path,
)
from boldsearch_integration.video_frames import VideoFrameProvider


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


def test_frame_request_can_generate_a_cached_frame_from_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "L21_V001.mp4"
    video.write_bytes(b"not a real mp4; ffmpeg is mocked")
    cache = tmp_path / "cache"
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return type("Result", (), {"returncode": 0, "stdout": "30/1\n", "stderr": ""})()
        Path(command[-1]).write_bytes(b"RIFFxxxxWEBP")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("boldsearch_integration.video_frames.subprocess.run", fake_run)
    provider = VideoFrameProvider(
        {"L21_V001": video}, cache_root=cache, max_width=960, webp_quality=82,
    )

    image, content_type = resolve_frame_request(
        tmp_path / "release", "/keyframes/L21_V001/30.webp", frame_provider=provider,
    )

    assert image == cache / "keyframes/L21_V001/30.webp"
    assert image.read_bytes() == b"RIFFxxxxWEBP"
    assert content_type == "image/webp"
    assert [command[0] for command in calls] == ["ffprobe", "ffmpeg"]
    assert "1.000000" in calls[-1]

    # The cached image must prevent another decode when the grid re-renders.
    resolved_again, _ = resolve_frame_request(
        tmp_path / "release", "/keyframes/L21_V001/30.png", frame_provider=provider,
    )
    assert resolved_again == image
    assert len(calls) == 2


def test_static_path_stays_under_frontend_dist(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("ok", encoding="utf-8")
    assert resolve_static_path(dist, "/") == dist / "index.html"
    with pytest.raises(PermissionError):
        resolve_static_path(dist, "/../secret")


def test_gateway_serves_active_release_and_proxies_api(tmp_path: Path) -> None:
    class BackendHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
    Thread(target=backend.serve_forever, daemon=True).start()
    public = tmp_path / "public"
    release = public / "releases" / "test-release"
    image = release / "keyframes/L21_V001/0.webp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"RIFFxxxxWEBP")
    (release / "Frames.csv").write_text(
        "video_id,frame_id,shot_id\nL21_V001,0,1\n", encoding="utf-8"
    )
    (public / "active.json").parent.mkdir(parents=True, exist_ok=True)
    (public / "active.json").write_text(
        json.dumps({"schema_version": "1.0", "release_id": "test-release"}),
        encoding="utf-8",
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("frontend", encoding="utf-8")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            public_root=public,
            frontend_dist=dist,
            backend_url=f"http://127.0.0.1:{backend.server_address[1]}",
        ),
    )
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/api/health") as response:
            assert response.read() == b'{"status":"ok"}'
        with urlopen(base + "/keyframes/L21_V001/0.png") as response:
            assert response.headers.get_content_type() == "image/webp"
            assert response.read() == b"RIFFxxxxWEBP"
        with urlopen(base + "/Frames.csv") as response:
            assert b"L21_V001" in response.read()
        with urlopen(base + "/") as response:
            assert response.read() == b"frontend"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/keyframes/../../secret.png")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        backend.shutdown()
        backend.server_close()
