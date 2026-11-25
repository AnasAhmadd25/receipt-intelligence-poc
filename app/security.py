"""Bounded image decoding and DNS-pinned, HTTPS-only image downloads."""
import asyncio
import io
import ipaddress
import socket
import warnings
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000  # Includes the actual24MP AraReceipt test image.


class InputError(ValueError):
    pass


def validate_image(data: bytes, *, max_pixels: int = MAX_IMAGE_PIXELS) -> tuple[bytes, str]:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InputError("Image must contain 1 byte to 10 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > max_pixels:
                    raise InputError("Decoded image exceeds pixel limit")
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InputError("Use PNG, JPEG, or WebP receipt images")
                mime = Image.MIME[image.format]
                image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputError("Invalid or oversized image") from exc
    return data, mime


def public_ip(value: str) -> bool:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


async def resolve_public_ips(host: str) -> list[str]:
    try:
        records = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM), timeout=5)
    except (OSError, asyncio.TimeoutError) as exc:
        raise InputError("Image host cannot be resolved") from exc
    addresses = list(dict.fromkeys(record[4][0] for record in records))
    if not addresses or not all(public_ip(ip) for ip in addresses):
        raise InputError("Image URL must resolve exclusively to public addresses")
    return addresses


async def fetch_image_url(url: str, *, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.port not in (None, 443) or parsed.fragment:
            raise InputError("Use a public HTTPS image URL on port 443 without credentials or fragment")
        hostname = parsed.hostname.encode("idna").decode("ascii")
        if "%" in hostname:
            raise InputError("Scoped addresses are not permitted")
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
        if literal is not None and not public_ip(str(literal)):
            raise InputError("Private or reserved image addresses are not permitted")
        addresses = await resolve_public_ips(hostname)
        # The connection uses the already validated literal IP. No second hostname
        # lookup can redirect it to a private address after the validation step.
        address = addresses[0]
        host_part = f"[{address}]" if ":" in address else address
        path = parsed.path or "/"
        pinned_url = f"https://{host_part}{path}" + (f"?{parsed.query}" if parsed.query else "")
        original_host = f"[{hostname}]" if ":" in hostname else hostname
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(20, connect=5)) as client:
            async with client.stream("GET", pinned_url, headers={"Host": original_host, "Accept": "image/*", "Accept-Encoding": "identity"}, extensions={"sni_hostname": hostname}) as response:
                if response.status_code != 200:
                    raise InputError("Image server must return HTTP 200 without redirects")
                if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                    raise InputError("Compressed image transfer encoding is not permitted")
                length = response.headers.get("content-length")
                if length is not None and (not length.isdecimal() or int(length) > max_bytes):
                    raise InputError("Image download exceeds size limit")
                parts = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise InputError("Image download exceeds size limit")
                    parts.append(chunk)
                return b"".join(parts)
    except InputError:
        raise
    except (httpx.HTTPError, ValueError, UnicodeError) as exc:
        # Avoid echoing signed URLs, userinfo, or transport errors to logs/clients.
        raise InputError("Unable to download a valid public HTTPS image") from exc
