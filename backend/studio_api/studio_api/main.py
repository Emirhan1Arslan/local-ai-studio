"""Sidecar giriş noktası.

Tauri bu süreci başlatır: `studio-api --port 0` → boş port seçilir ve stdout'a tek satır
JSON basılır: {"port": ..., "token": "..."}. Kabuk bu satırı okuyup UI'a iletir (ADR-0002/3).
Headless (Docker) modda STUDIO_PORT/STUDIO_API_TOKEN env değişkenleri kullanılır.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import secrets
import socket
import sys

import uvicorn
from studio_core.config import get_settings

_LOOPBACK_NAMES = {"localhost", ""}


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="studio-api")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None, help="0 = boş port seç")
    parser.add_argument("--token", default=None, help="'auto' = üret ve stdout'a yaz")
    args = parser.parse_args()

    settings = get_settings()
    host = args.host or settings.host
    port = args.port if args.port is not None else settings.port
    if port == 0:
        port = _free_port(host)

    if args.token == "auto":
        settings.api_token = secrets.token_urlsafe(32)
    elif args.token:
        settings.api_token = args.token

    # LAN modu (Faz 12 adım 1): loopback dışına bağlanırken token ZORUNLU —
    # yoksa üretilir ve stderr'e yazılır; asla token'sız dışa açılmaz.
    if not is_loopback(host):
        if not settings.api_token:
            settings.api_token = secrets.token_urlsafe(32)
            print(
                f"UYARI: {host} adresine token'sız bağlanma istendi; otomatik token üretildi: "
                f"{settings.api_token}",
                file=sys.stderr,
            )
        print(
            f"UYARI: LAN modu — API ve /mcp {host}:{port} üzerinden ağa açık. Yalnız güvendiğiniz "
            "ağda kullanın; istemciler 'Authorization: Bearer <token>' göndermeli (docs/mcp.md).",
            file=sys.stderr,
        )

    # MCP/istemciler için gerçek dış adres (uygulama içinden output_url üretirken okunur)
    settings.host = host
    settings.port = port

    # Sidecar el sıkışması: kabuk stdout'tan tek satır JSON okur.
    print(json.dumps({"port": port, "token": settings.api_token}), flush=True)

    from studio_api.app import create_app

    uvicorn.run(create_app(), host=host, port=port, log_config=None)


if __name__ == "__main__":
    sys.exit(main())
