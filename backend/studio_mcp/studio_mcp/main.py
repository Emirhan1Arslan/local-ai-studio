"""`studio-mcp` giriş noktası: Studio REST API'sine bağlanan MCP sunucusu.

Varsayılan transport stdio (Claude Desktop / Cursor / VS Code / Windsurf bu
süreci kendisi başlatır). `--transport http` ile bağımsız bir streamable HTTP
sunucusu olarak da çalışır (API sürecinin içindeki `/mcp` mount'undan farklı
olarak Studio'dan ayrı bir makinede köprü kurmak için).

Bağlantı: --url / STUDIO_API_URL (vars. http://127.0.0.1:8710),
          --token / STUDIO_API_TOKEN (LAN modunda zorunlu).

stdio modunda stdout PROTOKOL kanalıdır — tanılama yalnızca stderr'e yazılır.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

from .client import StudioClient
from .server import create_server


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="studio-mcp")
    parser.add_argument("--url", default=os.environ.get("STUDIO_API_URL", "http://127.0.0.1:8710"))
    parser.add_argument("--token", default=os.environ.get("STUDIO_API_TOKEN", ""))
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="yalnız --transport http")
    parser.add_argument("--port", type=int, default=8711, help="yalnız --transport http")
    parser.add_argument(
        "--no-check", action="store_true", help="başlangıçta /v1/health denemesini atla"
    )
    return parser.parse_args(argv)


async def _check(studio: StudioClient) -> None:
    try:
        info = await studio.health()
        print(f"studio-mcp: Studio {info.get('version')} @ {studio.base_url}", file=sys.stderr)
    except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001 - yalnız tanılama
        print(
            f"studio-mcp: UYARI — {studio.base_url} yanıt vermiyor ({exc}). "
            "Studio çalışıyor mu? --url/STUDIO_API_URL doğru mu? Araçlar çağrıldığında "
            "tekrar denenecek.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    studio = StudioClient(args.url, args.token)
    server = create_server(studio)
    if not args.no_check:
        asyncio.run(_check(studio))
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        from mcp.server.transport_security import TransportSecuritySettings

        print(f"studio-mcp: http://{args.host}:{args.port}/mcp", file=sys.stderr)
        asyncio.run(
            server.run_streamable_http_async(
                host=args.host,
                port=args.port,
                transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
            )
        )


if __name__ == "__main__":
    main()
