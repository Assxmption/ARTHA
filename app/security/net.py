"""
Client identity behind reverse proxies.

``X-Forwarded-For`` is attacker-controlled: any client can send
``X-Forwarded-For: 1.2.3.4`` and, if the first entry is trusted, get a fresh
rate-limit bucket (and a fresh demo quota) on every request.

A proxy *appends* the address it saw to the right of the header, so the only
trustworthy entries are the rightmost ``TRUSTED_PROXY_HOPS`` ones — exactly
one per proxy you control.  Set ``TRUSTED_PROXY_HOPS`` to the number of
proxies in front of the app (Render / Railway / Fly / HF Spaces: 1;
Cloudflare in front of one of those: 2).  The default 0 ignores the header
and uses the socket peer address, which is right for a direct connection
and safe (never spoofable) everywhere else.
"""

from __future__ import annotations

import ipaddress
import os

from starlette.requests import Request


def trusted_proxy_hops() -> int:
    try:
        return max(0, min(int(os.getenv("TRUSTED_PROXY_HOPS", "0")), 5))
    except ValueError:
        return 0


def _valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    hops = trusted_proxy_hops()
    if hops == 0:
        return peer
    header = request.headers.get("x-forwarded-for", "")
    chain = [p.strip() for p in header.split(",") if p.strip()]
    # The client is the entry just left of the `hops` addresses our own proxies appended.
    if len(chain) >= hops:
        candidate = chain[-hops]
        if _valid_ip(candidate):
            return candidate
    return peer


# ── Outbound URL guard (SSRF) ─────────────────────────────────────────────────

def is_public_http_url(url: str) -> bool:
    """
    True only for http(s) URLs whose host resolves exclusively to public
    addresses.  Blocks loopback, private ranges, link-local (cloud metadata at
    169.254.169.254), and anything else that is not globally routable.
    """
    import socket
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    addrs = {info[4][0] for info in infos}
    return bool(addrs) and all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addrs)
