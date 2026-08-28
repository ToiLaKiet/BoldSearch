from __future__ import annotations

import os
import platform
import re
import signal
import subprocess
import time
from pathlib import Path


_ASSET_BY_ARCH = {
    "x86_64": "cloudflared-linux-amd64",
    "amd64": "cloudflared-linux-amd64",
    "aarch64": "cloudflared-linux-arm64",
    "arm64": "cloudflared-linux-arm64",
}
_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def cloudflared_asset(machine: str | None = None) -> str:
    machine = (machine or platform.machine()).lower()
    try:
        return _ASSET_BY_ARCH[machine]
    except KeyError as exc:
        raise ValueError(f"unsupported cloudflared architecture: {machine}") from exc


def tunnel_command(binary: Path, gateway_url: str) -> tuple[str, ...]:
    if not gateway_url.startswith("http://127.0.0.1:"):
        raise ValueError("gateway_url must target a loopback HTTP gateway")
    return (
        str(binary), "tunnel", "--no-autoupdate", "--protocol", "http2",
        "--url", gateway_url,
    )


def extract_trycloudflare_url(log_text: str) -> str | None:
    match = _URL_RE.search(log_text.lower())
    return match.group(0) if match else None


def ensure_cloudflared(binary: Path, *, machine: str | None = None) -> Path:
    """Download the official release only when the requested binary is absent."""
    binary = binary.expanduser().resolve()
    try:
        subprocess.run(
            [str(binary), "--version"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        return binary
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        pass
    binary.parent.mkdir(parents=True, exist_ok=True)
    temporary = binary.with_name(binary.name + ".download")
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
        f"https://github.com/cloudflare/cloudflared/releases/latest/download/{cloudflared_asset(machine)}",
        "--output", str(temporary),
    ], check=True)
    try:
        temporary.chmod(0o755)
        os.replace(temporary, binary)
        binary.chmod(0o755)
    finally:
        temporary.unlink(missing_ok=True)
    return binary


def stop_owned_tunnel(pid_path: Path, binary: Path) -> None:
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
            "utf-8", errors="replace"
        )
        if str(binary) in cmdline and " tunnel " in f" {cmdline.replace(chr(0), ' ')} ":
            os.kill(pid, signal.SIGTERM)
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass


def start_quick_tunnel(
    binary: Path,
    gateway_url: str,
    *,
    log_path: Path,
    pid_path: Path,
    timeout_seconds: float = 90,
) -> tuple[subprocess.Popen, str]:
    binary = binary.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    pid_path = pid_path.expanduser().resolve()
    stop_owned_tunnel(pid_path, binary)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            tunnel_command(binary, gateway_url),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid), encoding="utf-8")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        public_url = extract_trycloudflare_url(text)
        if public_url:
            return process, public_url
        if process.poll() is not None:
            raise RuntimeError(f"cloudflared exited with code {process.returncode}: {text[-4000:]}")
        time.sleep(1)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    raise TimeoutError(f"cloudflared URL not found in {log_path}")
