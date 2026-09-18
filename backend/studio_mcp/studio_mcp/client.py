"""Studio REST API'sine ince async istemci (Faz 12).

MCP sunucusu Studio'nun iç servislerine DOĞRUDAN dokunmaz; her şeyi `/v1`
REST yüzeyi üzerinden yapar. Böylece aynı araç seti üç yerde çalışır:
- API süreci içinde mount edilmiş streamable HTTP transport (ASGI transport,
  ağ yok),
- Claude Desktop/Cursor'un aynı makinede başlattığı stdio süreci,
- UZAK bir makinede çalışan stdio süreci (`STUDIO_API_URL` LAN/VPN adresi +
  `STUDIO_API_TOKEN`) — Studio'nun GPU'lu makinede, istemcinin dizüstünde
  olduğu senaryo.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import PurePath
from typing import Any

import httpx

TERMINAL_JOB_STATES = {"done", "failed", "canceled"}
TERMINAL_DOWNLOAD_STATES = {"done", "failed", "canceled"}


class StudioApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{code} ({status}): {message}")
        self.status = status
        self.code = code
        self.message = message


class StudioTimeout(Exception):
    pass


class StudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8710",
        token: str = "",
        *,
        http: httpx.AsyncClient | None = None,
        public_base_url: str | None = None,
        poll_interval: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        # Dosya URL'leri istemciye bu adresle döner (mount edilmiş modda iç
        # ASGI adresi değil, gerçek host:port).
        self.public_base_url = (public_base_url or self.base_url).rstrip("/")
        self.poll_interval = poll_interval
        self._http = http or httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        self._outputs_dir: str | None = None

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- temel -----------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        resp = await self._http.request(
            method, url, json=json, params=params, headers=self._headers()
        )
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", {})
            except ValueError:
                err = {}
            raise StudioApiError(
                resp.status_code,
                str(err.get("code", "http_error")),
                str(err.get("message", resp.text[:500])),
            )
        if not resp.content:
            return None
        return resp.json()

    async def get_bytes(self, path: str) -> bytes:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        resp = await self._http.get(url, headers=self._headers())
        if resp.status_code >= 400:
            raise StudioApiError(resp.status_code, "http_error", resp.text[:200])
        return resp.content

    # -- sistem ----------------------------------------------------------------
    async def health(self) -> dict:
        return await self.request("GET", "/v1/health")

    async def system_info(self) -> dict:
        return await self.request("GET", "/v1/system/info")

    async def settings(self) -> dict:
        return await self.request("GET", "/v1/settings")

    async def outputs_dir(self) -> str:
        if self._outputs_dir is None:
            self._outputs_dir = str((await self.settings())["paths"]["outputs_dir"])
        return self._outputs_dir

    async def output_url(self, path: str | None) -> str | None:
        """outputs/ altındaki bir dosya için `/outputs/...` URL'i — uzak istemciler
        dosya yoluna erişemez, bu URL'den indirir."""
        if not path:
            return None
        try:
            rel = PurePath(path).relative_to(PurePath(await self.outputs_dir()))
        except ValueError:
            return None
        return f"{self.public_base_url}/outputs/{rel.as_posix()}"

    async def fetch_output(self, path: str) -> bytes:
        url = await self.output_url(path)
        if url is None:
            raise StudioApiError(400, "bad_request", f"dosya outputs dizini altında değil: {path}")
        return await self.get_bytes(url.replace(self.public_base_url, self.base_url, 1))

    # -- modeller / indirmeler ---------------------------------------------------
    async def list_models(self, kind: str | None = None) -> list[dict]:
        return await self.request("GET", "/v1/models", params={"kind": kind} if kind else None)

    async def rescan_models(self) -> dict:
        return await self.request("POST", "/v1/models/rescan")

    async def search_models(self, q: str, pipeline_tag: str = "", limit: int = 30) -> list[dict]:
        return await self.request(
            "GET",
            "/v1/downloads/search",
            params={"q": q, "pipeline_tag": pipeline_tag, "limit": limit},
        )

    async def create_download(self, body: dict) -> dict:
        return await self.request("POST", "/v1/downloads", json=body)

    async def create_snapshot_download(self, body: dict) -> list[dict]:
        return await self.request("POST", "/v1/downloads/snapshot", json=body)

    async def list_downloads(self) -> list[dict]:
        return await self.request("GET", "/v1/downloads")

    async def wait_downloads(
        self,
        ids: list[str],
        *,
        timeout: float,
        on_progress: Callable[[list[dict]], Awaitable[None]] | None = None,
    ) -> list[dict]:
        deadline = time.monotonic() + timeout
        while True:
            rows = [d for d in await self.list_downloads() if d["id"] in set(ids)]
            if on_progress is not None:
                await on_progress(rows)
            if rows and all(d["status"] in TERMINAL_DOWNLOAD_STATES for d in rows):
                return rows
            if time.monotonic() > deadline:
                raise StudioTimeout(f"indirme(ler) {timeout:.0f} sn içinde bitmedi")
            await asyncio.sleep(max(self.poll_interval, 1.0))

    # -- işler -------------------------------------------------------------------
    async def create_job(
        self, kind: str, params: dict, *, project_id: str | None = None, priority: int = 0
    ) -> dict:
        return await self.request(
            "POST",
            "/v1/jobs",
            json={"kind": kind, "params": params, "priority": priority, "project_id": project_id},
        )

    async def get_job(self, job_id: str) -> dict:
        return await self.request("GET", f"/v1/jobs/{job_id}")

    async def list_jobs(self, status: str | None = None, limit: int = 50) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self.request("GET", "/v1/jobs", params=params)

    async def cancel_job(self, job_id: str) -> bool:
        out = await self.request("POST", f"/v1/jobs/{job_id}/cancel")
        return bool(out and out.get("ok"))

    async def wait_job(
        self,
        job_id: str,
        *,
        timeout: float,
        on_status: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        deadline = time.monotonic() + timeout
        last_status = None
        while True:
            job = await self.get_job(job_id)
            if job["status"] != last_status:
                last_status = job["status"]
                if on_status is not None:
                    await on_status(job)
            if job["status"] in TERMINAL_JOB_STATES:
                return job
            if time.monotonic() > deadline:
                raise StudioTimeout(
                    f"iş {job_id} {timeout:.0f} sn içinde bitmedi (durum: {job['status']}); "
                    "get_job/wait_job ile daha sonra sorgulayın"
                )
            await asyncio.sleep(self.poll_interval)

    # -- yükleme -----------------------------------------------------------------
    async def upload(self, filename: str, data_base64: str) -> dict:
        return await self.request(
            "POST", "/v1/uploads", json={"filename": filename, "data_base64": data_base64}
        )

    # -- sohbet ------------------------------------------------------------------
    async def create_chat_session(
        self, model_path: str, title: str = "MCP sohbeti", system_prompt: str = ""
    ) -> dict:
        return await self.request(
            "POST",
            "/v1/chat",
            json={"model_path": model_path, "title": title, "system_prompt": system_prompt},
        )

    async def send_chat_message(
        self, session_id: str, content: str, *, temperature: float, max_tokens: int
    ) -> dict:
        return await self.request(
            "POST",
            f"/v1/chat/{session_id}/messages",
            json={"content": content, "temperature": temperature, "max_tokens": max_tokens},
        )

    # -- workflow'lar ------------------------------------------------------------
    async def node_types(self) -> list[dict]:
        return await self.request("GET", "/v1/workflows/node-types")

    async def list_workflows(self, project_id: str | None = None) -> list[dict]:
        return await self.request(
            "GET", "/v1/workflows", params={"project_id": project_id} if project_id else None
        )

    async def get_workflow(self, workflow_id: str) -> dict:
        return await self.request("GET", f"/v1/workflows/{workflow_id}")

    async def create_workflow(self, name: str, graph: dict, project_id: str | None = None) -> dict:
        return await self.request(
            "POST", "/v1/workflows", json={"name": name, "graph": graph, "project_id": project_id}
        )

    async def update_workflow(
        self, workflow_id: str, name: str, graph: dict, project_id: str | None = None
    ) -> dict:
        return await self.request(
            "PUT",
            f"/v1/workflows/{workflow_id}",
            json={"name": name, "graph": graph, "project_id": project_id},
        )

    async def run_workflow(self, workflow_id: str, seed: int = 0) -> dict:
        return await self.request("POST", f"/v1/workflows/{workflow_id}/run", json={"seed": seed})

    async def get_workflow_run(self, run_id: str) -> dict:
        return await self.request("GET", f"/v1/workflows/runs/{run_id}")

    async def wait_workflow_run(
        self,
        run_id: str,
        *,
        timeout: float,
        on_status: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            run = await self.get_workflow_run(run_id)
            if on_status is not None:
                await on_status(run)
            if run["status"] != "running":
                return run
            if time.monotonic() > deadline:
                raise StudioTimeout(f"workflow run {run_id} {timeout:.0f} sn içinde bitmedi")
            await asyncio.sleep(self.poll_interval)
