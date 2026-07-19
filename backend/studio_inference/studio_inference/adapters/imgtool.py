"""Görüntü araç seti adapter'ı (Faz 5): arka plan kaldırma, upscale, caption, OCR.

DİKKAT: Bu modül worker env'inde (envs/image — SdxlAdapter ile PAYLAŞILIR, bkz.
scheduler._ENV_ALIAS) çalışır. 3.11+ sözdizimi KULLANMA. Ağır importlar (torch/
transformers/spandrel/easyocr) metod içindedir; API süreci bu modülü hiç yüklemez.

params:
- op: bg_remove | upscale | caption | ocr (zorunlu)
- image_path: kaynak görsel (zorunlu, tüm op'lar için)
- output_path: route doldurur (bg_remove/upscale: PNG; caption/ocr: JSON)
- model_path: op'a özgü model (opsiyonel; boşsa varsayılan HF repo kullanılır)
  - bg_remove varsayılan: ZhengPeng7/BiRefNet (MIT lisans — RMBG-2.0'ın ticari
    kısıtlı lisansından kaçınmak için tercih edildi)
  - caption varsayılan: microsoft/Florence-2-base (MIT lisans)
  - upscale: Model Browser'dan bir .pth/.safetensors ZORUNLU (spandrel ile yüklenir —
    RealESRGAN/ESRGAN ailesini basicsr'a ihtiyaç duymadan çalıştırır)
- caption task: short | detailed | objects (vars. short)
- ocr langs: dil kodları listesi (vars. ["en"])
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

_FLORENCE_TASKS = {
    "short": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "objects": "<OD>",
}


def _import_error(hint: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(
        f"imgtool worker ortamı eksik ({hint}). "
        "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
    )


class ImgToolAdapter:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}  # "op:anahtar" -> yüklü model(ler)

    def load(self, model_path: str, **kwargs: Any) -> None:
        pass  # her op kendi modelini infer() içinde tembel yükler + cache'ler

    def infer(self, params: dict) -> Iterator[dict]:
        op = params.get("op", "")
        handlers = {
            "bg_remove": self._bg_remove,
            "upscale": self._upscale,
            "caption": self._caption,
            "ocr": self._ocr,
        }
        handler = handlers.get(op)
        if handler is None:
            raise RuntimeError(f"bilinmeyen op: {op!r} (bg_remove|upscale|caption|ocr)")
        image_path = params.get("image_path", "")
        if not image_path:
            raise RuntimeError("image_path gerekli")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        yield from handler(params, image_path, output_path)

    # -- arka plan kaldırma (BiRefNet) --------------------------------------

    def _get_birefnet(self, model_path: str):
        key = f"bg_remove:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as exc:
            raise _import_error("torch/transformers bulunamadı", exc) from exc

        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        self._cache[key] = model
        return model

    def _bg_remove(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        model_path = params.get("model_path") or "ZhengPeng7/BiRefNet"
        yield {"event": "progress", "value": 0.1, "message": "model yükleniyor…"}
        # ImportError burada (henüz cache'lenmemişse) yakalanıp anlaşılır bir
        # RuntimeError'a çevrilir — sonraki importlar bu yüzden guard'sız olabilir.
        model = self._get_birefnet(model_path)

        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resized = image.resize((1024, 1024))
        inp = TF.normalize(TF.to_tensor(resized), [0.5] * 3, [0.5] * 3).unsqueeze(0).to(device)

        yield {"event": "progress", "value": 0.5, "message": "maske çıkarılıyor…"}
        with torch.no_grad():
            preds = model(inp)[-1].sigmoid().cpu()
        mask = TF.to_pil_image(preds[0].squeeze(0)).resize(image.size)

        out = image.convert("RGBA")
        out.putalpha(mask)
        out.save(output_path)
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "bg_remove",
                "width": out.width,
                "height": out.height,
            },
        }

    # -- upscale (spandrel — basicsr gerektirmez) ---------------------------

    def _get_upscaler(self, model_path: str):
        key = f"upscale:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from spandrel import ModelLoader
        except ImportError as exc:
            raise _import_error("spandrel bulunamadı", exc) from exc

        model = ModelLoader().load_from_file(model_path)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        self._cache[key] = model
        return model

    def _upscale(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        model_path = params.get("model_path", "")
        if not model_path:
            raise RuntimeError(
                "upscale için model_path gerekli (Model Browser'dan bir ESRGAN modeli seçin)"
            )
        yield {"event": "progress", "value": 0.1, "message": "model yükleniyor…"}
        model = self._get_upscaler(model_path)

        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inp = TF.to_tensor(image).unsqueeze(0).to(device)

        yield {"event": "progress", "value": 0.5, "message": "büyütülüyor…"}
        with torch.no_grad():
            out_t = model(inp)
        out = TF.to_pil_image(out_t.squeeze(0).clamp(0, 1).cpu())
        out.save(output_path)
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "upscale",
                "width": out.width,
                "height": out.height,
            },
        }

    # -- caption / nesne tespiti (Florence-2) --------------------------------

    def _get_florence(self, model_path: str):
        key = f"caption:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise _import_error("torch/transformers bulunamadı", exc) from exc

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        self._cache[key] = (model, processor)
        return model, processor

    def _caption(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        model_path = params.get("model_path") or "microsoft/Florence-2-base"
        task_key = str(params.get("task", "short"))
        prompt = _FLORENCE_TASKS.get(task_key)
        if prompt is None:
            raise RuntimeError(f"bilinmeyen caption task: {task_key!r} (short|detailed|objects)")

        yield {"event": "progress", "value": 0.1, "message": "model yükleniyor…"}
        model, processor = self._get_florence(model_path)

        import torch
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = next(model.parameters()).dtype

        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        yield {"event": "progress", "value": 0.6, "message": "üretiliyor…"}
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            text, task=prompt, image_size=(image.width, image.height)
        )
        raw = parsed.get(prompt, "")
        caption = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"task": task_key, "result": raw}, f, ensure_ascii=False, indent=2)
        yield {
            "event": "done",
            "result": {"path": output_path, "op": "caption", "task": task_key, "caption": caption},
        }

    # -- OCR (EasyOCR) -------------------------------------------------------

    def _get_reader(self, langs: tuple[str, ...]):
        key = f"ocr:{','.join(langs)}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import easyocr
            import torch
        except ImportError as exc:
            raise _import_error("easyocr bulunamadı", exc) from exc

        # verbose=False: EasyOCR'ın ilk-kullanım model indirme ilerlemesi stdout'a
        # yazılırsa worker'ın JSON-lines IPC'sini bozar (\r'li tek satır asyncio'nun
        # readline tampon sınırını aşıp "Separator is not found" hatasına yol açar).
        reader = easyocr.Reader(list(langs), gpu=torch.cuda.is_available(), verbose=False)
        self._cache[key] = reader
        return reader

    def _ocr(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        langs = tuple(params.get("langs") or ["en"])
        yield {"event": "progress", "value": 0.1, "message": "model yükleniyor…"}
        reader = self._get_reader(langs)

        yield {"event": "progress", "value": 0.5, "message": "metin okunuyor…"}
        results = reader.readtext(image_path)
        items = [
            {
                "text": text,
                "confidence": float(conf),
                "bbox": [[float(x), float(y)] for x, y in box],
            }
            for box, text, conf in results
        ]
        full_text = "\n".join(item["text"] for item in items)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"items": items, "text": full_text}, f, ensure_ascii=False, indent=2)
        txt_path = output_path.rsplit(".", 1)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "ocr",
                "text": full_text,
                "item_count": len(items),
            },
        }

    def unload(self) -> None:
        self._cache = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def vram_estimate(self, params: dict) -> int:
        op = params.get("op", "")
        return {"bg_remove": 2000, "upscale": 2000, "caption": 1500, "ocr": 1500}.get(op, 1000)
