from __future__ import annotations

import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from .publisher import resolve_active_release
from .video_frames import FrameExtractionError, VideoFrameProvider


_FRAME_RE = re.compile(r"^/keyframes/(L\d{2}_V\d{2,3})/(\d+)\.(?:png|webp)$")
_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}


def resolve_frame_request(
    release_root: Path,
    request_path: str,
    *,
    frame_provider: VideoFrameProvider | None = None,
) -> tuple[Path, str]:
    """Resolve a frontend URL to a release asset or on-demand MP4 frame cache."""
    path = unquote(urlsplit(request_path).path)
    match = _FRAME_RE.fullmatch(path)
    if match is None:
        raise FileNotFoundError(f"unsupported keyframe path: {request_path}")
    video_id, frame_id = match.groups()
    candidate = (release_root / "keyframes" / video_id / f"{frame_id}.webp").resolve()
    keyframe_root = (release_root / "keyframes").resolve()
    if not candidate.is_relative_to(keyframe_root):
        raise FileNotFoundError(f"keyframe not found: {candidate}")
    if not candidate.is_file():
        if frame_provider is None:
            raise FileNotFoundError(f"keyframe not found: {candidate}")
        candidate = frame_provider.resolve(video_id, int(frame_id))
    return candidate, "image/webp"


def resolve_static_path(frontend_dist: Path, request_path: str) -> Path:
    """Resolve SPA/static path while preventing traversal outside dist."""
    frontend_dist = frontend_dist.expanduser().resolve()
    relative = unquote(urlsplit(request_path).path).lstrip("/") or "index.html"
    candidate = (frontend_dist / relative).resolve()
    if not candidate.is_relative_to(frontend_dist):
        raise PermissionError("static path escapes frontend dist")
    if candidate.is_file():
        return candidate
    fallback = frontend_dist / "index.html"
    if not fallback.is_file():
        raise FileNotFoundError(f"frontend index missing: {fallback}")
    return fallback


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(
    *,
    public_root: Path,
    frontend_dist: Path,
    backend_url: str,
    frame_provider: VideoFrameProvider | None = None,
):
    public_root = public_root.expanduser().resolve()
    frontend_dist = frontend_dist.expanduser().resolve()
    backend_url = backend_url.rstrip("/")

    class GatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "BoldSearchGateway/1.0"

        def do_GET(self):
            self._route(head=False)

        def do_HEAD(self):
            self._route(head=True)

        def do_POST(self):
            self._route_api_only()

        def do_PUT(self):
            self._route_api_only()

        def do_PATCH(self):
            self._route_api_only()

        def do_DELETE(self):
            self._route_api_only()

        def do_OPTIONS(self):
            self._route_api_only()

        def _clean_path(self) -> str:
            return unquote(urlsplit(self.path).path)

        def _route_api_only(self):
            path = self._clean_path()
            if path == "/api" or path.startswith("/api/"):
                self._proxy_api(head=False)
            else:
                self.send_error(405, "method not allowed")

        def _route(self, *, head: bool):
            path = self._clean_path()
            if path == "/api" or path.startswith("/api/"):
                self._proxy_api(head=head)
                return
            release = resolve_active_release(public_root)
            if path == "/Frames.csv":
                self._serve_file(release / "Frames.csv", "text/csv; charset=utf-8", "no-cache", head)
                return
            if path.startswith("/keyframes/"):
                try:
                    image, content_type = resolve_frame_request(release, path)
                except FileNotFoundError:
                    self.send_error(404, "image not found")
                    return
                except FrameExtractionError as exc:
                    self.send_error(502, f"could not extract image: {exc}")
                    return
                self._serve_file(image, content_type, "public, max-age=31536000, immutable", head)
                return
            try:
                candidate = resolve_static_path(frontend_dist, path)
            except PermissionError:
                self.send_error(403, "forbidden")
                return
            except FileNotFoundError:
                self.send_error(404, "frontend not found")
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            asset_root = (frontend_dist / "assets").resolve()
            cache = (
                "public, max-age=31536000, immutable"
                if candidate.is_relative_to(asset_root)
                else "no-cache"
            )
            self._serve_file(candidate, content_type, cache, head)

        def _serve_file(self, path: Path, content_type: str, cache: str, head: bool):
            if not path.is_file():
                self.send_error(404, "file not found")
                return
            stat = path.stat()
            etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Cache-Control", cache)
            self.send_header("ETag", etag)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head:
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        self.wfile.write(chunk)

        def _proxy_api(self, *, head: bool):
            parsed = urlsplit(self.path)
            target = backend_url + parsed.path
            if parsed.query:
                target += "?" + parsed.query
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                self.send_error(400, "invalid content length")
                return
            if length > 32 * 1024 * 1024:
                self.send_error(413, "request too large")
                return
            body = self.rfile.read(length) if length else None
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() not in _HOP_HEADERS
            }
            headers["Accept-Encoding"] = "identity"
            request = Request(target, data=body, headers=headers, method=self.command)
            try:
                response = urlopen(request, timeout=900)
                status = response.status
                response_headers = response.headers
                payload = b"" if head else response.read()
            except HTTPError as error:
                status = error.code
                response_headers = error.headers
                payload = b"" if head else error.read()
            except URLError:
                self.send_error(502, "backend unavailable")
                return
            self.send_response(status)
            for key, value in response_headers.items():
                if key.lower() not in _HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head and payload:
                self.wfile.write(payload)

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}", flush=True)

    return GatewayHandler


def serve(
    *,
    public_root: Path,
    frontend_dist: Path,
    backend_url: str,
    host: str,
    port: int,
    frame_provider: VideoFrameProvider | None = None,
) -> None:
    handler = make_handler(
        public_root=public_root,
        frontend_dist=frontend_dist,
        backend_url=backend_url,
        frame_provider=frame_provider,
    )
    _GatewayServer((host, port), handler).serve_forever()


def main() -> int:
    manifest_value = os.environ.get("BOLDSEARCH_VIDEO_MANIFEST", "").strip()
    frame_provider = None
    if manifest_value:
        frame_provider = VideoFrameProvider.from_json_file(
            Path(manifest_value),
            cache_root=Path(os.environ["BOLDSEARCH_FRAME_CACHE_ROOT"]),
            max_width=int(os.environ.get("BOLDSEARCH_FRAME_MAX_WIDTH", "960")),
            webp_quality=int(os.environ.get("BOLDSEARCH_FRAME_WEBP_QUALITY", "82")),
        )
    serve(
        public_root=Path(os.environ["BOLDSEARCH_PUBLIC_ROOT"]),
        frontend_dist=Path(os.environ["BOLDSEARCH_FRONTEND_DIST"]),
        backend_url=os.environ.get("BOLDSEARCH_BACKEND", "http://127.0.0.1:8000"),
        host=os.environ.get("BOLDSEARCH_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("BOLDSEARCH_GATEWAY_PORT", "7860")),
        frame_provider=frame_provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
