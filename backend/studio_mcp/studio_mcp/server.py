"""Local AI Studio MCP sunucusu (Faz 12 adım 3).

Araç seti plan.md'nin listesi: generate_image, generate_video, generate_3d, chat,
transcribe_audio, synthesize_voice, remove_background, upscale, caption, ocr,
segment, download_model, install_model, list_models, workflow_run,
workflow_create, workflow_save, workflow_export — artı job-id + poll deseni
için get_job/wait_job/cancel_job/list_jobs ve uzak istemcilerin girdi
dosyası gönderebilmesi için upload_file.

Uzun işler: her üretim aracı `wait=True` (vars.) ile işi bitene kadar bekler
ve durum geçişlerini MCP progress bildirimi olarak akıtır; `wait=False` ile
hemen job_id döner, istemci get_job/wait_job ile sorgular.

Bütün araçlar Studio'nun REST API'sini (`StudioClient`) çağırır; iç servislere
doğrudan erişim yok — sunucu Studio ile aynı makinede de, uzak bir makinede de
aynı kodla çalışır.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any

import httpx
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .client import StudioApiError, StudioClient, StudioTimeout

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
GENERATE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
NETWORK = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

DEFAULT_TIMEOUT = 900.0

_STATUS_PROGRESS = {"queued": 0.05, "running": 0.5, "done": 1.0, "failed": 1.0, "canceled": 1.0}

INSTRUCTIONS = """Local AI Studio: yerel GPU üzerinde görsel/video/3D/ses/LLM üretimi.
Model yolu (model_path) bilmiyorsan boş bırak — kurulu ilk uygun model seçilir;
kurulu modelleri list_models ile gör, yenilerini search_models → download_model /
install_model ile indir. Uzun işler wait=true ile bitene kadar bekler; wait=false
ile job_id döner, get_job/wait_job ile sorgula. Çıktı dosyaları Studio makinesinde
(path) durur; uzak istemciler output_url'den indirir."""


def _job_summary(job: dict, output_url: str | None = None) -> dict:
    result = job.get("result") or {}
    return {
        "job_id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "result": result,
        "output_path": result.get("path"),
        "output_url": output_url,
        "error": job.get("error") or None,
    }


def create_server(studio: StudioClient) -> MCPServer:
    server = MCPServer(
        name="local-ai-studio",
        title="Local AI Studio",
        version="0.1.0",
        instructions=INSTRUCTIONS,
    )

    def tool(annotations: ToolAnnotations) -> Callable:
        """`server.tool` + hata çevirisi: MCP yalnız `ToolError`'ın mesajını istemciye
        iletir, diğer istisnalar "Error executing tool" olarak gizlenir. Studio'nun
        anlaşılır hataları (400 mesajları, zaman aşımı, model yok) istemciye ulaşmalı."""

        def decorate(fn: Callable) -> Callable:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except ToolError:
                    raise
                except httpx.HTTPError as exc:
                    raise ToolError(
                        f"Studio API'sine ulaşılamadı ({studio.base_url}): {exc}"
                    ) from exc
                except (StudioApiError, StudioTimeout, RuntimeError, ValueError) as exc:
                    raise ToolError(str(exc)) from exc

            return server.tool(annotations=annotations)(wrapper)

        return decorate

    # -- ortak yardımcılar --------------------------------------------------------
    async def _report(ctx: Context, job: dict) -> None:
        await ctx.report_progress(
            _STATUS_PROGRESS.get(job["status"], 0.5),
            1.0,
            f"{job['kind']} işi {job['id']}: {job['status']}",
        )

    async def _submit(
        kind: str,
        params: dict,
        *,
        wait: bool,
        timeout_s: float,
        ctx: Context,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        job = await studio.create_job(kind, params, project_id=project_id)
        await _report(ctx, job)
        if not wait:
            return _job_summary(job)
        job = await studio.wait_job(
            job["id"], timeout=timeout_s, on_status=lambda j: _report(ctx, j)
        )
        summary = _job_summary(job, await studio.output_url((job.get("result") or {}).get("path")))
        if job["status"] != "done":
            raise RuntimeError(
                f"iş {job['id']} {job['status']}: {job.get('error') or 'bilinmeyen hata'}"
            )
        return summary

    async def _pick_model(kind: str, what: str) -> str:
        models = await studio.list_models(kind=kind)
        if not models:
            raise RuntimeError(
                f"kurulu {what} modeli yok (kind={kind!r}). search_models ile bul, "
                "download_model / install_model ile indir ya da model_path ver."
            )
        return str(models[0]["path"])

    # -- sistem / modeller --------------------------------------------------------
    @tool(READ_ONLY)
    async def system_info() -> dict[str, Any]:
        """Studio'nun çalıştığı makinenin donanım/yazılım bilgisi: GPU'lar, VRAM,
        RAM, disk, backend (cuda/rocm/mps/cpu), sürüm, veri dizini."""
        info = await studio.system_info()
        info["studio_version"] = (await studio.health()).get("version")
        info["api_url"] = studio.public_base_url
        return info

    @tool(READ_ONLY)
    async def list_models(kind: str = "") -> list[dict[str, Any]]:
        """Kurulu modelleri listeler. kind ile filtrele: checkpoint (görsel üretim
        SDXL/SD3/FLUX), lora, vae, controlnet, llm (GGUF), tts, stt, video, 3d,
        upscale, background_removal, captioning, segmentation, face_restoration, ocr.
        Dönen `path` alanı araçların model_path parametresine verilir."""
        rows = await studio.list_models(kind=kind or None)
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "kind": r["kind"],
                "family": r["family"],
                "format": r["format"],
                "path": r["path"],
                "size_mb": round(r["size_bytes"] / 1_048_576, 1),
                "license": r.get("license", ""),
            }
            for r in rows
        ]

    @tool(NETWORK)
    async def search_models(
        query: str, pipeline_tag: str = "", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Hugging Face Hub'da model arar (ör. "sdxl", "flux", "qwen gguf").
        pipeline_tag: text-to-image, text-generation, automatic-speech-recognition,
        text-to-speech, image-to-3d vb. Sonuçtaki repo_id download_model/install_model'e verilir."""
        return await studio.search_models(query, pipeline_tag=pipeline_tag, limit=limit)

    @tool(NETWORK)
    async def download_model(
        repo_id: str,
        filename: str,
        subdir: str,
        revision: str = "main",
        wait: bool = False,
        timeout_s: float = 3600.0,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Tek dosyalık bir model indirir (HF repo_id + filename → models/<subdir>/).
        subdir: checkpoints | lora | vae | controlnet | llm | tts | stt | upscale | ...
        Büyük dosyalar dakikalar sürer; wait=false ile kuyruğa alıp list_models ile
        takip etmek genelde daha iyidir. İndirme bitince model otomatik listelenir."""
        rows = [
            await studio.create_download(
                {"repo_id": repo_id, "filename": filename, "subdir": subdir, "revision": revision}
            )
        ]
        if wait:
            rows = await studio.wait_downloads(
                [r["id"] for r in rows], timeout=timeout_s, on_progress=_dl_progress(ctx)
            )
        return {"downloads": rows}

    @tool(NETWORK)
    async def install_model(
        repo_id: str,
        subdir: str,
        only_patterns: list[str] | None = None,
        revision: str = "main",
        wait: bool = False,
        timeout_s: float = 3600.0,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Çok dosyalı bir HF repo'sunu (diffusers pipeline, Kokoro, BiRefNet, SAM...)
        bütün olarak models/<subdir>/<repo-adı>/ altına kurar. only_patterns ile
        yalnızca eşleşen dosyalar (ör. ["*.bf16*"]) indirilir."""
        rows = await studio.create_snapshot_download(
            {
                "repo_id": repo_id,
                "subdir": subdir,
                "revision": revision,
                "only_patterns": only_patterns or [],
            }
        )
        if wait:
            rows = await studio.wait_downloads(
                [r["id"] for r in rows], timeout=timeout_s, on_progress=_dl_progress(ctx)
            )
        return {"file_count": len(rows), "downloads": rows}

    def _dl_progress(ctx: Context):
        async def _on(rows: list[dict]) -> None:
            total = sum(r["total_bytes"] for r in rows) or 1
            done = sum(r["downloaded_bytes"] for r in rows)
            await ctx.report_progress(done, total, f"indiriliyor {done / 1_048_576:.0f} MB")

        return _on

    # -- görsel ------------------------------------------------------------------
    @tool(GENERATE)
    async def generate_image(
        prompt: str,
        model_path: str = "",
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg: float = 7.0,
        seed: int = -1,
        model_family: str = "",
        sampler: str = "",
        vram_mode: str = "max",
        loras: list[dict] | None = None,
        include_image: bool = False,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> Any:
        """Metinden görsel üretir (SDXL / SD3 / FLUX — aile dosyadan otomatik algılanır).
        model_path boşsa kurulu ilk checkpoint kullanılır. seed=-1 rastgele; aynı seed +
        parametre aynı görseli verir. include_image=true ise PNG'yi de içerik olarak döner
        (büyük görsellerde yavaş). Çıktı: output_path (Studio makinesinde) + output_url."""
        params: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "mode": "txt2img",
            "vram_mode": vram_mode,
            "model_path": model_path or await _pick_model("checkpoint", "görsel üretim"),
        }
        if model_family:
            params["model_family"] = model_family
        if sampler:
            params["sampler"] = sampler
        if loras:
            params["loras"] = loras
        summary = await _submit("image", params, wait=wait, timeout_s=timeout_s, ctx=ctx)
        if include_image and summary.get("output_path"):
            png = await studio.fetch_output(summary["output_path"])
            return [json.dumps(summary, ensure_ascii=False), Image(data=png, format="png")]
        return summary

    @tool(GENERATE)
    async def edit_image(
        prompt: str,
        init_image: str,
        mode: str = "img2img",
        mask_image: str = "",
        strength: float = 0.75,
        outpaint: dict | None = None,
        model_path: str = "",
        negative_prompt: str = "",
        steps: int = 30,
        cfg: float = 7.0,
        seed: int = -1,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Var olan görseli düzenler: mode=img2img (strength 0-1), inpaint (mask_image: beyaz =
        yeniden çizilecek), outpaint (outpaint={"left":128,"right":128,"top":0,"bottom":0}).
        init_image Studio makinesindeki bir yol olmalı — uzaktan upload_file ile gönder."""
        params: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "mode": mode,
            "init_image": init_image,
            "strength": strength,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "model_path": model_path or await _pick_model("checkpoint", "görsel üretim"),
        }
        if mask_image:
            params["mask_image"] = mask_image
        if outpaint:
            params["outpaint"] = outpaint
        return await _submit("image", params, wait=wait, timeout_s=timeout_s, ctx=ctx)

    # -- görsel araçları ---------------------------------------------------------
    async def _imgtool(
        op: str, image_path: str, extra: dict, *, wait: bool, timeout_s: float, ctx: Context
    ) -> dict[str, Any]:
        params = {"op": op, "image_path": image_path, **extra}
        return await _submit("imgtool", params, wait=wait, timeout_s=timeout_s, ctx=ctx)

    @tool(GENERATE)
    async def remove_background(
        image_path: str,
        model_path: str = "",
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Görselin arka planını kaldırır, şeffaf PNG üretir (vars. BiRefNet)."""
        return await _imgtool(
            "bg_remove",
            image_path,
            {"model_path": model_path},
            wait=wait,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    @tool(GENERATE)
    async def upscale(
        image_path: str,
        model_path: str = "",
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Görseli ESRGAN ailesi bir modelle büyütür (ör. RealESRGAN_x4plus → 4x).
        model_path boşsa kurulu ilk upscale modeli kullanılır."""
        return await _imgtool(
            "upscale",
            image_path,
            {"model_path": model_path or await _pick_model("upscale", "upscale")},
            wait=wait,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    @tool(GENERATE)
    async def caption(
        image_path: str,
        task: str = "short",
        model_path: str = "",
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Görseli metinle açıklar (Florence-2). task: short | detailed | objects."""
        return await _imgtool(
            "caption",
            image_path,
            {"task": task, "model_path": model_path},
            wait=wait,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    @tool(GENERATE)
    async def ocr(
        image_path: str,
        langs: list[str] | None = None,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Görseldeki metni okur (EasyOCR). langs: ["en"], ["tr","en"] vb.
        Sonuç: json + txt + hOCR dosyaları."""
        return await _imgtool(
            "ocr", image_path, {"langs": langs or ["en"]}, wait=wait, timeout_s=timeout_s, ctx=ctx
        )

    @tool(GENERATE)
    async def segment(
        image_path: str,
        mode: str = "text",
        text_prompt: str = "",
        points: list[list[float]] | None = None,
        box: list[float] | None = None,
        cutout: bool = True,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Görselde bölge segmentler (SAM + Grounding DINO). mode=text: text_prompt
        ("kırmızı araba"); point: points=[[x,y],...]; box: box=[x1,y1,x2,y2].
        cutout=true ise maskeyle kesilmiş RGBA da üretilir."""
        extra: dict[str, Any] = {"mode": mode, "cutout": cutout}
        if text_prompt:
            extra["text_prompt"] = text_prompt
        if points:
            extra["points"] = points
        if box:
            extra["box"] = box
        return await _imgtool("segment", image_path, extra, wait=wait, timeout_s=timeout_s, ctx=ctx)

    # -- video / 3D --------------------------------------------------------------
    @tool(GENERATE)
    async def generate_video(
        prompt: str,
        family: str = "ltx",
        model_path: str = "",
        mode: str = "t2v",
        init_image: str = "",
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        num_frames: int = 0,
        fps: int = 0,
        steps: int = 0,
        cfg: float = 0.0,
        seed: int = -1,
        vram_mode: str = "max",
        wait: bool = True,
        timeout_s: float = 3600.0,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Metinden (t2v) ya da görselden (i2v, init_image) video üretir.
        family: ltx (hızlı, vars.) | wan | cogvideox | hunyuan | mochi. 0 verilen
        boyut/kare/adım alanları ailenin varsayılanına düşer. Dakikalar sürebilir."""
        params: dict[str, Any] = {
            "family": family,
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "vram_mode": vram_mode,
        }
        if model_path:
            params["model_path"] = model_path
        if init_image:
            params["init_image"] = init_image
        for key, val in (
            ("width", width),
            ("height", height),
            ("num_frames", num_frames),
            ("fps", fps),
            ("steps", steps),
            ("cfg", cfg),
        ):
            if val:
                params[key] = val
        return await _submit("video", params, wait=wait, timeout_s=timeout_s, ctx=ctx)

    @tool(GENERATE)
    async def generate_3d(
        image_path: str,
        family: str = "triposr",
        model_path: str = "",
        format: str = "glb",
        remove_background: bool = True,
        options: dict | None = None,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Tek görselden 3D mesh üretir. family: triposr (hızlı, vars.) | sf3d (dokulu) |
        openlrm | hunyuan3d | instantmesh | trellis (dokulu, en kaliteli) | crm.
        format: glb | obj | stl | ply. options: aileye özgü ek parametreler
        (ör. trellis {"num_inference_steps": 25, "texture_size": 1024};
        hunyuan3d {"texture": true} → Paint ile 2048² dokulu GLB, daha uzun sürer)."""
        params: dict[str, Any] = {
            "family": family,
            "image_path": image_path,
            "format": format,
            "remove_background": remove_background,
            **(options or {}),
        }
        if model_path:
            params["model_path"] = model_path
        return await _submit("mesh", params, wait=wait, timeout_s=timeout_s, ctx=ctx)

    # -- ses ---------------------------------------------------------------------
    @tool(GENERATE)
    async def transcribe_audio(
        audio_path: str,
        model_path: str = "",
        model_size: str = "base",
        language: str = "",
        task: str = "transcribe",
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Ses dosyasını yazıya çevirir (faster-whisper). language boş = otomatik
        algıla ("tr" desteklenir). task: transcribe | translate. Çıktı: json+txt+srt+vtt."""
        params: dict[str, Any] = {
            "op": "stt",
            "audio_path": audio_path,
            "model_size": model_size,
            "task": task,
        }
        if model_path:
            params["model_path"] = model_path
        if language:
            params["language"] = language
        return await _submit("audio", params, wait=wait, timeout_s=timeout_s, ctx=ctx)

    @tool(GENERATE)
    async def synthesize_voice(
        text: str,
        engine: str = "kokoro",
        model_path: str = "",
        voice: str = "af_heart",
        lang_code: str = "a",
        language: str = "en",
        speaker: str = "",
        speaker_wav: str = "",
        consent: bool = False,
        speed: float = 1.0,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Metni seslendirir (WAV). engine=kokoro (hızlı; voice/lang_code) veya xtts
        (çok dilli, language="tr" dahil; speaker: hazır ses adı ya da speaker_wav:
        klonlanacak referans — bu durumda consent=true zorunlu, ses sahibinin izni).
        XTTS-v2 yalnızca ticari olmayan kullanım (CPML)."""
        params: dict[str, Any] = {"op": "tts", "text": text, "engine": engine, "speed": speed}
        if model_path:
            params["model_path"] = model_path
        if engine == "kokoro":
            params.update({"voice": voice, "lang_code": lang_code})
        else:
            params["language"] = language
            if speaker:
                params["speaker"] = speaker
            if speaker_wav:
                params.update({"speaker_wav": speaker_wav, "consent": consent})
        return await _submit("audio", params, wait=wait, timeout_s=timeout_s, ctx=ctx)

    # -- LLM ---------------------------------------------------------------------
    @tool(GENERATE)
    async def chat(
        message: str,
        model_path: str = "",
        session_id: str = "",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Yerel LLM'e (GGUF, llama.cpp) mesaj gönderir. session_id boşsa yeni oturum
        açılır ve döner — devam etmek için aynı session_id'yi ver. model_path boşsa
        kurulu ilk GGUF kullanılır."""
        if not session_id:
            session = await studio.create_chat_session(
                model_path or await _pick_model("llm", "LLM"), system_prompt=system_prompt
            )
            session_id = session["id"]
        sent = await studio.send_chat_message(
            session_id, message, temperature=temperature, max_tokens=max_tokens
        )
        job = await studio.get_job(sent["job_id"])
        await _report(ctx, job)
        if not wait:
            return {"session_id": session_id, **_job_summary(job)}
        job = await studio.wait_job(
            job["id"], timeout=timeout_s, on_status=lambda j: _report(ctx, j)
        )
        if job["status"] != "done":
            raise RuntimeError(f"LLM işi {job['status']}: {job.get('error')}")
        return {
            "session_id": session_id,
            "job_id": job["id"],
            "reply": (job.get("result") or {}).get("text", ""),
            "tokens": (job.get("result") or {}).get("tokens"),
        }

    # -- işler -------------------------------------------------------------------
    @tool(GENERATE)
    async def run_job(
        kind: str,
        params: dict,
        wait: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT,
        project_id: str = "",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Genel iş kuyruklama: Studio'nun herhangi bir iş türünü (image, imgtool, llm,
        audio, video, videotool, mesh, meshtool ve plugin türleri) ham params ile çalıştırır.
        Özel araçların kapsamadığı op'lar için (videotool trim/concat, meshtool decimate...)."""
        return await _submit(
            kind, params, wait=wait, timeout_s=timeout_s, ctx=ctx, project_id=project_id or None
        )

    @tool(READ_ONLY)
    async def get_job(job_id: str) -> dict[str, Any]:
        """Bir işin anlık durumunu/sonucunu döner (queued|running|done|failed|canceled)."""
        job = await studio.get_job(job_id)
        return _job_summary(job, await studio.output_url((job.get("result") or {}).get("path")))

    @tool(READ_ONLY)
    async def wait_job(
        job_id: str,
        timeout_s: float = DEFAULT_TIMEOUT,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """wait=false ile kuyruklanmış bir işi bitene kadar bekler (progress akıtır)."""
        job = await studio.wait_job(job_id, timeout=timeout_s, on_status=lambda j: _report(ctx, j))
        return _job_summary(job, await studio.output_url((job.get("result") or {}).get("path")))

    @tool(DESTRUCTIVE)
    async def cancel_job(job_id: str) -> dict[str, Any]:
        """Kuyruktaki/çalışan bir işi iptal eder; VRAM serbest kalır."""
        return {"job_id": job_id, "ok": await studio.cancel_job(job_id)}

    @tool(READ_ONLY)
    async def list_jobs(status: str = "", limit: int = 20) -> list[dict[str, Any]]:
        """Son işleri listeler; status ile filtrele (queued|running|done|failed|canceled)."""
        return [_job_summary(j) for j in await studio.list_jobs(status or None, limit=limit)]

    # -- dosya -------------------------------------------------------------------
    @tool(GENERATE)
    async def upload_file(filename: str, data_base64: str) -> dict[str, Any]:
        """Uzak istemciden Studio makinesine girdi dosyası gönderir (png/jpg/webp/wav/
        mp3/mp4...). Dönen path, image_path/audio_path/init_image parametrelerine verilir."""
        return await studio.upload(filename, data_base64)

    # -- workflow'lar ------------------------------------------------------------
    @tool(READ_ONLY)
    async def list_node_types() -> list[dict[str, Any]]:
        """Workflow editörünün düğüm türleri (port ve parametre şemalarıyla) —
        workflow_create için grafik kurarken kullan."""
        return await studio.node_types()

    @tool(READ_ONLY)
    async def list_workflows(project_id: str = "") -> list[dict[str, Any]]:
        """Kayıtlı workflow'ları listeler."""
        rows = await studio.list_workflows(project_id or None)
        return [
            {
                "id": w["id"],
                "name": w["name"],
                "project_id": w.get("project_id"),
                "node_count": len((w.get("graph") or {}).get("nodes", [])),
                "updated_at": w["updated_at"],
            }
            for w in rows
        ]

    @tool(GENERATE)
    async def workflow_create(
        name: str, graph: dict[str, Any], project_id: str = ""
    ) -> dict[str, Any]:
        """Yeni workflow kaydeder. graph = {"nodes":[{"id","type","params"}],
        "edges":[{"id","from_node","from_port","to_node","to_port"}]}; şema list_node_types'tan.
        Grafik kaydedilirken doğrulanır (bilinmeyen tür, tip uyumsuzluğu, döngü → hata)."""
        return await studio.create_workflow(name, graph, project_id or None)

    @tool(GENERATE)
    async def workflow_save(
        workflow_id: str, graph: dict[str, Any], name: str = ""
    ) -> dict[str, Any]:
        """Var olan workflow'un grafiğini (ve isteğe bağlı adını) günceller."""
        current = await studio.get_workflow(workflow_id)
        return await studio.update_workflow(
            workflow_id, name or current["name"], graph, current.get("project_id")
        )

    @tool(GENERATE)
    async def workflow_run(
        workflow_id: str,
        seed: int = 0,
        wait: bool = True,
        timeout_s: float = 3600.0,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Kayıtlı workflow'u çalıştırır; düğüm başına durum/çıktı döner.
        Değişmeyen dallar ikinci çalıştırmada cache'ten gelir (from_cache)."""
        run = await studio.run_workflow(workflow_id, seed=seed)
        if wait:

            async def _on(r: dict) -> None:
                done = sum(1 for v in r["node_results"].values() if v["status"] != "running")
                await ctx.report_progress(done, max(1, len(r["node_results"])), r["status"])

            run = await studio.wait_workflow_run(run["id"], timeout=timeout_s, on_status=_on)
        for res in run["node_results"].values():
            out = res.get("outputs") or {}
            for val in out.values():
                if isinstance(val, dict) and val.get("path"):
                    val["url"] = await studio.output_url(val["path"])
        return run

    @tool(READ_ONLY)
    async def workflow_export(workflow_id: str) -> dict[str, Any]:
        """Workflow'u taşınabilir, sürümlü JSON belgesi olarak döner
        (format=localaistudio.workflow v1) — dosyaya yazıp paylaşılabilir,
        workflow_create'e graph olarak geri verilebilir."""
        wf = await studio.get_workflow(workflow_id)
        return {
            "format": "localaistudio.workflow",
            "version": 1,
            "name": wf["name"],
            "graph": wf["graph"],
        }

    return server
