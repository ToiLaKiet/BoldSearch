from pathlib import Path

import pytest

from boldsearch_integration.tunnel import (
    cloudflared_asset,
    extract_trycloudflare_url,
    tunnel_command,
)


def test_cloudflared_asset_supports_kaggle_architectures() -> None:
    assert cloudflared_asset("x86_64") == "cloudflared-linux-amd64"
    assert cloudflared_asset("aarch64") == "cloudflared-linux-arm64"
    with pytest.raises(ValueError, match="unsupported"):
        cloudflared_asset("mips")


def test_extract_url_requires_trycloudflare_hostname() -> None:
    log = "INF https://safe-name.trycloudflare.com connected"
    assert extract_trycloudflare_url(log) == "https://safe-name.trycloudflare.com"
    assert extract_trycloudflare_url("https://example.com") is None


def test_tunnel_command_is_loopback_only() -> None:
    assert tunnel_command(Path("/tmp/cloudflared"), "http://127.0.0.1:7860") == (
        "/tmp/cloudflared", "tunnel", "--no-autoupdate", "--protocol", "http2",
        "--url", "http://127.0.0.1:7860",
    )
