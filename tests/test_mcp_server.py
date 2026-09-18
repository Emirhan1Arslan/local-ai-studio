"""Faz 12 MCP sunucusu testleri.

MCP araçları REST API'yi çağırır; burada API `client` fixture'ıyla (TestClient,
gerçek scheduler + dummy adapter) ayakta, MCP sunucusu ise resmi SDK'nın süreç-içi
`Client`'ıyla sürülür — StudioClient, API'ye httpx ASGI transport'uyla bağlanır
(ağ yok). Ayrıca `/mcp` mount'unun streamable HTTP el sıkışması ve Bearer token
kapısı doğrudan HTTP ile doğrulanır.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fakes import write_fake_safetensors
from mcp.client.client import Client
from studio_mcp import StudioClient, create_server
from studio_mcp.main import _parse

PLAN_TOOLS = {
    "generate_image",
    "generate_video",
    "generate_3d",
    "chat",
    "transcribe_audio",
    "synthesize_voice",
    "remove_background",
    "upscale",
    "caption",
    "ocr",
    "segment",
    "download_model",
    "install_model",
    "list_models",
    "workflow_run",
    "workflow_create",
    "workflow_save",
    "workflow_export",
}


def _run(client, coro_fn):
    """API TestClient'ı kendi thread/loop'unda çalışır; MCP tarafını ayrı bir
    asyncio loop'ta sürüyoruz, ASGI transport aynı FastAPI app nesnesini çağırır."""

    async def main():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://t")
        studio = StudioClient(
            "http://t", "", http=http, public_base_url="http://127.0.0.1:8710", poll_interval=0.05
        )
        server = create_server(studio)
        async with Client(server) as mcp:
            return await coro_fn(mcp, studio)

    return asyncio.run(main())


def _text(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def test_tool_list_covers_plan(client):
    async def go(mcp, _):
        return {t.name: t for t in (await mcp.list_tools()).tools}

    tools = _run(client, go)
    assert PLAN_TOOLS <= set(tools)
    assert tools["list_models"].annotations.read_only_hint is True
    assert tools["download_model"].annotations.open_world_hint is True
    assert tools["cancel_job"].annotations.destructive_hint is True
    # şema: model_path opsiyonel, prompt zorunlu
    schema = tools["generate_image"].input_schema
    assert "prompt" in schema["required"]
    assert "model_path" not in schema.get("required", [])


def test_system_info_tool(client):
    async def go(mcp, _):
        return await mcp.call_tool("system_info", {})

    r = _run(client, go)
    assert r.is_error is False, _text(r)
    assert r.structured_content["studio_version"]
    assert "gpus" in r.structured_content


def test_run_job_waits_and_returns_result(client):
    async def go(mcp, _):
        return await mcp.call_tool(
            "run_job", {"kind": "dummy", "params": {"steps": 2, "step_ms": 10}}
        )

    r = _run(client, go)
    assert r.is_error is False, _text(r)
    assert r.structured_content["status"] == "done"
    assert r.structured_content["result"] == {"steps": 2}


def test_run_job_nowait_then_wait_job(client):
    async def go(mcp, _):
        first = await mcp.call_tool(
            "run_job", {"kind": "dummy", "params": {"steps": 2, "step_ms": 10}, "wait": False}
        )
        job_id = first.structured_content["job_id"]
        assert first.structured_content["status"] in ("queued", "running")
        progress: list[float] = []

        async def on_progress(p, total, message):
            progress.append(p)

        waited = await mcp.call_tool("wait_job", {"job_id": job_id}, progress_callback=on_progress)
        listed = await mcp.call_tool("list_jobs", {"status": "done"})
        return waited, listed, progress

    waited, listed, progress = _run(client, go)
    assert waited.structured_content["status"] == "done"
    assert any(
        j["job_id"] == waited.structured_content["job_id"]
        for j in listed.structured_content["result"]
    )
    assert progress and progress[-1] == 1.0


def test_failed_job_surfaces_readable_error(client):
    async def go(mcp, _):
        return await mcp.call_tool("run_job", {"kind": "dummy", "params": {"fail": True}})

    r = _run(client, go)
    assert r.is_error is True
    assert "kasıtlı test hatası" in _text(r)


def test_generate_image_without_model_gives_actionable_error(client):
    async def go(mcp, _):
        return await mcp.call_tool("generate_image", {"prompt": "bir kedi"})

    r = _run(client, go)
    assert r.is_error is True
    msg = _text(r)
    assert "kurulu" in msg and "download_model" in msg  # ham "Error executing tool" değil


def test_list_models_after_install(client, tmp_path):
    models_dir = tmp_path / "LocalAIStudio" / "models"
    write_fake_safetensors(models_dir / "checkpoints" / "sd_xl_base_1.0.safetensors")
    assert client.post("/v1/models/rescan").status_code == 200

    async def go(mcp, _):
        return await mcp.call_tool("list_models", {"kind": "checkpoint"})

    r = _run(client, go)
    assert r.is_error is False, _text(r)
    rows = r.structured_content["result"]
    assert len(rows) == 1
    assert rows[0]["path"].endswith("sd_xl_base_1.0.safetensors")
    assert rows[0]["kind"] == "checkpoint"


def test_workflow_create_run_export_roundtrip(client, tmp_path):
    dest = str(tmp_path / "mcp_note.txt")
    graph = {
        "nodes": [
            {"id": "p", "type": "core.prompt", "params": {"text": "mcp merhaba"}},
            {"id": "s", "type": "core.save", "params": {"path": dest}},
        ],
        "edges": [
            {"id": "e1", "from_node": "p", "from_port": "text", "to_node": "s", "to_port": "value"}
        ],
    }

    async def go(mcp, _):
        created = await mcp.call_tool("workflow_create", {"name": "mcp-wf", "graph": graph})
        wf_id = created.structured_content["id"]
        run = await mcp.call_tool("workflow_run", {"workflow_id": wf_id})
        exported = await mcp.call_tool("workflow_export", {"workflow_id": wf_id})
        graph2 = dict(graph)
        graph2["nodes"] = [dict(graph["nodes"][0], params={"text": "değişti"}), graph["nodes"][1]]
        saved = await mcp.call_tool("workflow_save", {"workflow_id": wf_id, "graph": graph2})
        listed = await mcp.call_tool("list_workflows", {})
        return created, run, exported, saved, listed

    created, run, exported, saved, listed = _run(client, go)
    assert created.is_error is False, _text(created)
    assert run.structured_content["status"] == "done", _text(run)
    assert run.structured_content["node_results"]["s"]["outputs"]["path"] == dest
    assert exported.structured_content["format"] == "localaistudio.workflow"
    assert exported.structured_content["graph"]["nodes"][0]["params"]["text"] == "mcp merhaba"
    assert saved.structured_content["graph"]["nodes"][0]["params"]["text"] == "değişti"
    assert any(
        w["id"] == created.structured_content["id"] for w in listed.structured_content["result"]
    )


def test_workflow_create_invalid_graph_is_readable(client):
    async def go(mcp, _):
        return await mcp.call_tool(
            "workflow_create",
            {"name": "bozuk", "graph": {"nodes": [{"id": "a", "type": "yok.tur"}], "edges": []}},
        )

    r = _run(client, go)
    assert r.is_error is True
    assert "bilinmeyen düğüm türü" in _text(r)


def test_output_url_maps_outputs_dir(client):
    async def go(_, studio):
        outputs = await studio.outputs_dir()
        inside = await studio.output_url(f"{outputs}/abc.png")
        outside = await studio.output_url("C:/baska/yer.png")
        return inside, outside

    inside, outside = _run(client, go)
    assert inside == "http://127.0.0.1:8710/outputs/abc.png"
    assert outside is None


def _initialize_body() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }


_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_http_mount_initialize_handshake(client):
    r = client.post("/mcp", json=_initialize_body(), headers=_MCP_HEADERS)
    assert r.status_code == 200, r.text
    assert r.headers.get("mcp-session-id")
    payload = json.loads(r.text.split("data: ", 1)[1].split("\n", 1)[0])
    assert payload["result"]["serverInfo"]["name"] == "local-ai-studio"
    assert "tools" in payload["result"]["capabilities"]


def test_http_mount_requires_token_when_set(client, monkeypatch):
    from studio_core.config import get_settings

    monkeypatch.setattr(get_settings(), "api_token", "gizli")
    denied = client.post("/mcp", json=_initialize_body(), headers=_MCP_HEADERS)
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "unauthorized"

    ok = client.post(
        "/mcp",
        json=_initialize_body(),
        headers={**_MCP_HEADERS, "Authorization": "Bearer gizli"},
    )
    assert ok.status_code == 200, ok.text


def test_cli_defaults_and_env(monkeypatch):
    monkeypatch.delenv("STUDIO_API_URL", raising=False)
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    args = _parse([])
    assert args.url == "http://127.0.0.1:8710" and args.transport == "stdio" and args.token == ""

    monkeypatch.setenv("STUDIO_API_URL", "http://192.168.1.20:8710")
    monkeypatch.setenv("STUDIO_API_TOKEN", "t0k")
    args = _parse(["--transport", "http", "--port", "9000"])
    assert args.url == "http://192.168.1.20:8710" and args.token == "t0k" and args.port == 9000


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("192.168.1.5", False),
    ],
)
def test_lan_mode_loopback_detection(host, expected):
    from studio_api.main import is_loopback

    assert is_loopback(host) is expected
