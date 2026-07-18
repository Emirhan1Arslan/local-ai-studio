"""FastAPI uygulaması: /v1 sürümleme, standart hata modeli, token auth, WS veriyolu."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from studio_core import __version__ as core_version
from studio_core.config import get_settings
from studio_core.hardware import SystemInfo, get_system_info
from studio_core.logging import get_logger, setup_logging
from studio_core.migrations import upgrade_to_head
from studio_models.scanner import scan
from studio_models.watcher import ModelWatcher

from studio_api.bus import bus
from studio_api.routes.models import router as models_router

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


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    upgrade_to_head()

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
        watcher = ModelWatcher(on_models_changed)
        watcher.start()
        try:
            yield
        finally:
            watcher.stop()

    app = FastAPI(
        title="Local AI Studio API",
        version=core_version,
        docs_url="/v1/docs",
        openapi_url="/v1/openapi.json",
        lifespan=lifespan,
    )

    app.include_router(models_router, dependencies=[Depends(_check_token)])

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
        except WebSocketDisconnect:
            pass
        finally:
            bus.unsubscribe(q)
            log.info("ws_client_disconnected")

    log.info("app_created", data_dir=str(settings.data_dir))
    return app
