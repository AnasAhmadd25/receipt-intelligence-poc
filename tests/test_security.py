import io
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from PIL import Image

from app.security import InputError, fetch_image_url, resolve_public_ips, validate_image


def test_valid_image_and_decoded_limit():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8)).save(stream, "PNG")
    assert validate_image(stream.getvalue())[1] == "image/png"
    with pytest.raises(InputError):
        validate_image(b"not an image")
    with pytest.raises(InputError):
        validate_image(stream.getvalue(), max_pixels=30)


@pytest.mark.parametrize("url", ["http://example.com/x", "https://127.0.0.1/x", "https://[::1]/x", "https://user:pass@example.com/x", "https://example.com:444/x"])
async def test_block_unsafe_urls(url):
    with pytest.raises(InputError):
        await fetch_image_url(url)


async def test_dns_rebinding_uses_validated_ip_with_original_tls_host():
    seen = []
    async def fake_send(client, request, **kwargs):
        seen.append(request)
        return httpx.Response(200, content=b"abc", request=request)
    with patch("app.security.resolve_public_ips", AsyncMock(return_value=["93.184.216.34"])), patch.object(httpx.AsyncClient, "send", fake_send):
        assert await fetch_image_url("https://images.example/a.png") == b"abc"
    assert seen[0].url.host == "93.184.216.34"
    assert seen[0].headers["host"] == "images.example"
    assert seen[0].extensions["sni_hostname"] == "images.example"


async def test_redirect_and_oversized_download_rejected():
    for response in [httpx.Response(302, headers={"location": "https://127.0.0.1"}), httpx.Response(200, content=b"x" * 11)]:
        async def fake_send(client, request, **kwargs):
            response.request = request
            return response
        with patch("app.security.resolve_public_ips", AsyncMock(return_value=["93.184.216.34"])), patch.object(httpx.AsyncClient, "send", fake_send):
            with pytest.raises(InputError):
                await fetch_image_url("https://images.example/a", max_bytes=10)


async def test_mixed_public_private_dns_answers_rejected():
    answers = [(2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("10.0.0.8", 443))]
    with patch("asyncio.BaseEventLoop.getaddrinfo", AsyncMock(return_value=answers)), pytest.raises(InputError):
        await resolve_public_ips("mixed.example")


async def test_compressed_response_rejected_before_decoding():
    async def fake_send(client, request, **kwargs):
        return httpx.Response(200, headers={"content-encoding": "gzip"}, request=request, stream=httpx.ByteStream(b"not decoded"))
    with patch("app.security.resolve_public_ips", AsyncMock(return_value=["93.184.216.34"])), patch.object(httpx.AsyncClient, "send", fake_send), pytest.raises(InputError, match="encoding"):
        await fetch_image_url("https://images.example/a.png")


def test_corrupt_png_checksum_is_input_error():
    output = io.BytesIO()
    Image.new("RGB", (4, 4)).save(output, "PNG")
    damaged = bytearray(output.getvalue())
    # IDAT CRC is checked by PNG verify(), after the image header has opened.
    offset = damaged.index(b"IDAT")
    damaged[offset + 4] ^= 1
    with pytest.raises(InputError):
        validate_image(bytes(damaged))
