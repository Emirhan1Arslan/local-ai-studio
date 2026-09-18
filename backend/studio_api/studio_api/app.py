"""FastAPI uygulaması: /v1 sürümleme, standart hata modeli, token auth, WS veriyolu."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send
from studio_core import __version__ as core_version
from studio_core.assets import register_asset_for_job
from studio_core.chat import save_chat_message_for_job
from studio_core.config import get_settings
from studio_core.hardware import SystemInfo, get_system_info
from studio_core.logging import get_logger, setup_logging
from studio_core.migrations import upgrade_to_head
from studio_core.settings_store import load_overrides
from studio_inference.scheduler import scheduler
from studio_inference.telemetry import TelemetryService
from studio_mcp import StudioClient
from studio_mcp import create_server as create_mcp_server
from studio_models.downloader import manager as download_manager
from studio_models.scanner import scan
from studio_models.watcher import ModelWatcher

from studio_api.bus import bus
from studio_api.routes.assets import router as assets_router
from studio_api.routes.chat import router as chat_router
from studio_api.routes.downloads import router as downloads_router
from studio_api.routes.jobs import router as jobs_router
from studio_api.routes.models import router as models_router
from studio_api.routes.projects import router as projects_router
from studio_api.routes.settings import router as settings_router
from studio_api.routes.uploads import router as uploads_router
from studio_api.routes.workflows import router as workflows_router

log = get_logger("studio_api")


class ApiError(Exception):
    """Standart hata modeli: {error: {code, message, detail?}} olarak serileşir."""

    def __init__(self, status: int, code: str, message: str, detail: dict | None = None):
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail or {}


def _check_token(request: Request) -> None:
    token = get_settings().api_token
    if not token:
        return  # dev modu: token pazarlığı sidecar (Tauri) ile gelecek
    provided = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(provided, token):
        raise ApiError(401, "unauthorized", "Geçersiz veya eksik API token'ı")


class _BearerGuard:
    """Mount edilen MCP alt uygulaması için REST ile aynı Bearer token kapısı
    (FastAPI Depends'i Mount'a uygulanamaz, bu yüzden ASGI düzeyinde)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        token = get_settings().api_token
        if scope["type"] == "http" and token:
            provided = Headers(scope=scope).get("authorization", "").removeprefix("Bearer ").strip()
            if not secrets.compare_digest(provided, token):
                response = JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "unauthorized",
                            "message": "Geçersiz veya eksik API token'ı",
                            "detail": {},
                        }
                    },
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _mount_mcp(app: FastAPI):
    """MCP streamable HTTP transport'unu `/mcp` altına takar (Faz 12 adım 3).

    Araçlar aynı süreçteki REST API'yi ASGI transport ile çağırır — ağ yok,
    ikinci bir iş mantığı yok (docs/architecture.md "MCP-first": tek yüzey).
    Uzak istemciler (Cursor/VS Code/özel) doğrudan http://host:port/mcp'ye Bearer
    token ile bağlanır; Claude Desktop gibi header veremeyen istemciler için
    `studio-mcp` stdio köprüsü var (docs/mcp.md).
    """
    settings = get_settings()
    internal = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://studio.internal", timeout=None
    )
    studio = StudioClient(
        "http://studio.internal",
        settings.api_token,
        http=internal,
        public_base_url=f"http://{settings.host}:{settings.port}",
    )
    server = create_mcp_server(studio)
    # DNS-rebinding koruması kapalı: MCP ucu REST ile AYNI token kapısının
    # arkasında ve LAN modunda Host başlığı makine adı/IP olur.
    mcp_app = server.streamable_http_app(
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    app.mount("/mcp", _BearerGuard(mcp_app), name="mcp")
    return server


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    upgrade_to_head()
    load_overrides()  # DB'deki kalıcı ayar override'ları (Faz 1 cila) süreç başında uygulanır

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_running_loop()

        def on_models_changed() -> None:
            # watchdog thread'i: tara, değişiklikleri asyncio veriyoluna aktar
            result = scan()
            for topic, items in (
                ("model.added", result.added),
                ("model.removed", result.removed),
                ("model.updated", result.updated),
            ):
                for item in items:
                    loop.call_soon_threadsafe(bus.emit, topic, **item)

        await asyncio.to_thread(scan)  # açılışta tam senkron
        download_manager.reset()
        download_manager.set_emit(bus.emit)
        await asyncio.to_thread(download_manager.recover)
        watcher = ModelWatcher(on_models_changed)
        watcher.start()

        def on_job_done(job_id: str) -> None:
            register_asset_for_job(job_id)
            save_chat_message_for_job(job_id)

        scheduler._concurrency = settings.worker_concurrency
        scheduler.set_emit(bus.emit)
        scheduler.set_on_done(on_job_done)
        await scheduler.start()
        telemetry = TelemetryService(bus.emit, settings.telemetry_interval)
        telemetry.start()
        import studio_inference.telemetry as telemetry_mod

        telemetry_mod.telemetry = telemetry
        try:
            # Mount edilen alt uygulamanın kendi lifespan'ı çalışmaz; MCP oturum
            # yöneticisini ana lifespan içinde biz çalıştırıyoruz.
            async with mcp_server.session_manager.run():
                yield
        finally:
            await telemetry.stop()
            await scheduler.stop()
            watcher.stop()

    app = FastAPI(
        title="Local AI Studio API",
        version=core_version,
        docs_url="/v1/docs",
        openapi_url="/v1/openapi.json",
        lifespan=lifespan,
    )

    app.include_router(models_router, dependencies=[Depends(_check_token)])
    app.include_router(downloads_router, dependencies=[Depends(_check_token)])
    app.include_router(jobs_router, dependencies=[Depends(_check_token)])
    app.include_router(projects_router, dependencies=[Depends(_check_token)])
    app.include_router(assets_router, dependencies=[Depends(_check_token)])
    app.include_router(uploads_router, dependencies=[Depends(_check_token)])
    app.include_router(settings_router, dependencies=[Depends(_check_token)])
    app.include_router(chat_router, dependencies=[Depends(_check_token)])
    app.include_router(workflows_router, dependencies=[Depends(_check_token)])
    # Üretim çıktıları (galeri/önizleme). LAN modunda da token'sız servis edilir:
    # dosya adları tahmin edilemez uuid'lerdir ve UI/MCP istemcileri <img src> ile okur.
    app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")
    mcp_server = _mount_mcp(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "tauri://localhost"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def _api_error_handler(_req: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(exc.detail), "detail": {}}},
        )

    @app.get("/v1/health")
    async def health() -> dict:
        return {"status": "ok", "version": core_version}

    @app.get("/v1/system/info", dependencies=[Depends(_check_token)])
    async def system_info() -> SystemInfo:
        return await asyncio.to_thread(get_system_info)

    @app.websocket("/v1/ws")
    async def ws_events(ws: WebSocket) -> None:
        token = get_settings().api_token
        if token and ws.query_params.get("token", "") != token:
            await ws.close(code=4401)
            return
        await ws.accept()
        q = bus.subscribe()
        log.info("ws_client_connected", clients=1)
        bus.emit("system.status", message="client connected")
        try:
            while True:
                event = await q.get()
                await ws.send_text(event.model_dump_json())
        except (WebSocketDisconnect, RuntimeError, OSError):
            # istemci koptu / soket kapandı: sessizce ayrıl (uvicorn'a spam olmasın)
            pass
        finally:
            bus.unsubscribe(q)
            log.info("ws_client_disconnected")

    log.info("app_created", data_dir=str(settings.data_dir))
    return app
