"""
Tests for asr/router.py — DI and HTTP contract.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from asr.schema import AsrRequest, AsrResponse, TranscriptResult, TranscriptSegment


class _FakeTranscriber:
    """Fake for DI testing."""

    def __init__(self, result: TranscriptResult | None = None) -> None:
        self._result = result or TranscriptResult(language="vi", text="xin chào", segments=())

    def transcribe(self, audio_path: str | Path, *, video_id: str) -> TranscriptResult:
        return self._result


class _FakeResolver:
    """Resolve one test video to its fixture media."""

    def __init__(self, media_path: Path) -> None:
        self.media_path = media_path

    def resolve(self, video_id: str) -> Path:
        return self.media_path


def _create_audio(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=d=0.1:f=440",
            "-ac", "1", "-ar", "16000", str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture
def app_with_transcriber(tmp_path):
    """Create a test app with injected fake transcriber."""
    from fastapi import FastAPI
    from asr.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.asr_transcriber = _FakeTranscriber()
    app.state.asr_media_resolver = _FakeResolver(_create_audio(tmp_path / "source.wav"))
    return app


@pytest.fixture
def app_without_transcriber():
    """Create a test app with no transcriber (lifespan not run)."""
    from fastapi import FastAPI
    from asr.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


class TestTranscribeEndpoint:
    def test_request_schema_does_not_expose_audio_url(self):
        request_schema = AsrRequest.model_json_schema()

        assert set(request_schema["properties"]) == {
            "video_id",
            "language",
            "keyframes",
        }
        assert request_schema["required"] == ["video_id"]
        assert request_schema["additionalProperties"] is False

    def test_response_schema_names_complete_keyframe_shape(self):
        response_schema = AsrResponse.model_json_schema()
        keyframe_ref = response_schema["properties"]["keyframes"]["items"]["$ref"]
        keyframe_schema = response_schema["$defs"]["KeyframeWithText"]

        assert keyframe_ref.endswith("/KeyframeWithText")
        assert set(keyframe_schema["properties"]) == {
            "video_id",
            "frame_id",
            "timestamp",
            "img_path",
            "text",
        }
        assert "img_path" in keyframe_schema["required"]

    def test_transcribe_with_video_id_resolver(self, app_with_transcriber):
        app_with_transcriber.state.asr_transcriber = _FakeTranscriber(
            TranscriptResult(
                language="vi",
                text="xin chào",
                segments=(
                    TranscriptSegment(
                        video_id="v1", start=0.0, end=2.0, text="xin chào"
                    ),
                ),
            )
        )
        client = TestClient(app_with_transcriber)
        response = client.post("/api/asr/transcribe", json={
            "video_id": "v1",
            "language": "vi",
            "keyframes": [
                {
                    "video_id": "v1",
                    "frame_id": "001",
                    "timestamp": 1.0,
                    "img_path": "v1/001.png",
                },
                {
                    "video_id": "v1",
                    "frame_id": "002",
                    "timestamp": 3.0,
                    "img_path": "v1/002.png",
                },
            ],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "vi"
        assert data["full_transcript"] == "xin chào"
        assert data["segments"] == [
            {
                "video_id": "v1",
                "start": 0.0,
                "end": 2.0,
                "text": "xin chào",
                "confidence": None,
            }
        ]
        assert data["keyframes"] == [
            {
                "video_id": "v1",
                "frame_id": "001",
                "timestamp": 1.0,
                "img_path": "v1/001.png",
                "text": "xin chào",
            },
            {
                "video_id": "v1",
                "frame_id": "002",
                "timestamp": 3.0,
                "img_path": "v1/002.png",
                "text": None,
            },
        ]
        assert "unmatched_count" not in data

    def test_transcribe_requires_keyframe_img_path(self, app_with_transcriber):
        response = TestClient(app_with_transcriber).post(
            "/api/asr/transcribe",
            json={
                "video_id": "v1",
                "keyframes": [
                    {"video_id": "v1", "frame_id": "001", "timestamp": 1.0}
                ],
            },
        )

        assert response.status_code == 422

    def test_transcribe_rejects_non_vietnamese(self, app_with_transcriber):
        client = TestClient(app_with_transcriber)
        response = client.post("/api/asr/transcribe", json={
            "video_id": "v1",
            "language": "en",
        })
        assert response.status_code == 422
        assert response.json()["detail"] == "ASR_LANGUAGE_UNSUPPORTED"

    def test_transcribe_requires_video_id(self, app_with_transcriber):
        client = TestClient(app_with_transcriber)
        response = client.post("/api/asr/transcribe", json={
            "language": "vi",
        })
        assert response.status_code == 422

    def test_transcribe_rejects_removed_audio_url_field(self, app_with_transcriber):
        client = TestClient(app_with_transcriber)
        response = client.post(
            "/api/asr/transcribe",
            json={"audio_url": "https://attacker.example/audio.wav", "language": "vi"},
        )
        assert response.status_code == 422

    def test_transcribe_requires_configured_media_resolver(self, app_without_transcriber):
        app_without_transcriber.state.asr_transcriber = _FakeTranscriber()
        client = TestClient(app_without_transcriber)
        response = client.post(
            "/api/asr/transcribe",
            json={"video_id": "v1", "language": "vi"},
        )
        assert response.status_code == 503

    def test_missing_media_maps_to_http_status(self, app_with_transcriber):
        class MissingResolver:
            def resolve(self, video_id):
                raise FileNotFoundError("missing")

        app_with_transcriber.state.asr_media_resolver = MissingResolver()
        response = TestClient(app_with_transcriber).post(
            "/api/asr/transcribe",
            json={"video_id": "missing", "language": "vi"},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "MEDIA_NOT_FOUND"

    @pytest.mark.parametrize(
        ("error", "status_code", "detail"),
        [
            (ValueError("bad model output"), 502, "ASR_RESULT_INVALID"),
            (RuntimeError("decode failed"), 503, "ASR_INFERENCE_FAILED"),
        ],
    )
    def test_transcriber_error_is_logged_and_mapped(
        self, app_with_transcriber, caplog, error, status_code, detail
    ):
        class FailingTranscriber:
            def transcribe(self, audio_path, *, video_id):
                raise error

        app_with_transcriber.state.asr_transcriber = FailingTranscriber()
        with caplog.at_level("ERROR", logger="asr.router"):
            response = TestClient(app_with_transcriber).post(
                "/api/asr/transcribe",
                json={"video_id": "v1", "language": "vi"},
            )

        assert response.status_code == status_code
        assert response.json()["detail"] == detail
        assert caplog.messages == [f"{detail} for video_id=v1"]
