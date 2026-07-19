"""Görüntü araç seti adapter'ı (Faz 5): arka plan kaldırma, upscale, caption, OCR.

DİKKAT: Bu modül worker env'inde (envs/image — SdxlAdapter ile PAYLAŞILIR, bkz.
scheduler._ENV_ALIAS) çalışır. 3.11+ sözdizimi KULLANMA. Ağır importlar (torch/
transformers/spandrel/easyocr) metod içindedir; API süreci bu modülü hiç yüklemez.

params:
- op: bg_remove | upscale | caption | ocr | segment | crop | adjust (zorunlu)
- image_path: kaynak görsel (zorunlu, tüm op'lar için)
- output_path: route doldurur (görsel çıktılı op'lar: PNG; caption/ocr: JSON)
- model_path: op'a özgü model (opsiyonel; boşsa varsayılan HF repo kullanılır)
  - bg_remove varsayılan: ZhengPeng7/BiRefNet (MIT lisans — RMBG-2.0'ın ticari
    kısıtlı lisansından kaçınmak için tercih edildi)
  - caption varsayılan: microsoft/Florence-2-base (MIT lisans)
  - upscale: Model Browser'dan bir .pth/.safetensors ZORUNLU (spandrel ile yüklenir —
    RealESRGAN/ESRGAN ailesini basicsr'a ihtiyaç duymadan çalıştırır)
- caption task: short | detailed | objects (vars. short)
- ocr langs: dil kodları listesi (vars. ["en"]); çıktı: json + txt + hOCR
- segment mode: point | box | text
  - point: points=[[x,y],...] · point_labels=[1,0,...] (vars. hepsi ön plan)
  - box: box=[x1,y1,x2,y2]
  - text: text_prompt="kırmızı araba" — Grounding DINO ile açık kelime dağarcığı
    tespiti, en yüksek skorlu kutu SAM'e verilir (box_threshold/text_threshold opsiyonel)
  - cutout: true ise maskeyle birlikte RGBA kesim de üretir (result.cutout_path)
  - varsayılan modeller: facebook/sam-vit-base, IDEA-Research/grounding-dino-tiny
- crop box: [x1,y1,x2,y2] (zorunlu)
- adjust: brightness/contrast/saturation/sharpness (0'dan büyük çarpanlar, 1.0=değişiklik yok)
- face_restore: model_path ZORUNLU (Model Browser'dan bir GFPGAN .pth),
  weight (vars. 0.5, 0=orijinale yakın·1=tam restorasyon), only_center_face (vars. false)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

_FLORENCE_TASKS = {
    "short": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "objects": "<OD>",
}

_SAM_DEFAULT = "facebook/sam-vit-base"
_DINO_DEFAULT = "IDEA-Research/grounding-dino-tiny"


def _hocr_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _make_hocr(items: list[dict], width: int, height: int) -> str:
    """OCR sonuçlarını minimal ama geçerli bir hOCR belgesine dönüştürür (plan.md Faz 5)."""
    words = []
    for i, item in enumerate(items):
        xs = [p[0] for p in item["bbox"]]
        ys = [p[1] for p in item["bbox"]]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        conf = int(item["confidence"] * 100)
        words.append(
            f'<span class="ocrx_word" id="word_{i + 1}" '
            f'title="bbox {x0} {y0} {x1} {y1}; x_wconf {conf}">{_hocr_escape(item["text"])}</span>'
        )
    body = " ".join(words)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n<head>\n'
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n'
        '<meta name="ocr-system" content="local-ai-studio easyocr" />\n'
        '<meta name="ocr-capabilities" content="ocr_page ocrx_word" />\n'
        "</head>\n<body>\n"
        f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">\n'
        f'<div class="ocr_carea" id="block_1"><p class="ocr_par" id="par_1">'
        f'<span class="ocr_line" id="line_1">{body}</span></p></div>\n'
        "</div>\n</body>\n</html>\n"
    )


def _resolve_repo(path_or_repo: str) -> str:
    """Model Browser'da kayıtlı bir snapshot klasöründeki ağırlık dosyası verilmişse
    (yanında config.json varsa) from_pretrained için üst klasöre döner; tek dosya yol
    ya da HF repo id'si olduğu gibi kullanılır (bkz. /v1/downloads/snapshot)."""
    p = str(path_or_repo)
    parent = os.path.dirname(p)
    if parent and os.path.exists(os.path.join(parent, "config.json")):
        return parent
    return p


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
            "segment": self._segment,
            "crop": self._crop,
            "adjust": self._adjust,
            "face_restore": self._face_restore,
        }
        handler = handlers.get(op)
        if handler is None:
            raise RuntimeError(
                f"bilinmeyen op: {op!r} "
                "(bg_remove|upscale|caption|ocr|segment|crop|adjust|face_restore)"
            )
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

        model = AutoModelForImageSegmentation.from_pretrained(
            _resolve_repo(model_path), trust_remote_code=True
        )
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

        repo = _resolve_repo(model_path)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
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

        from PIL import Image

        width, height = Image.open(image_path).size  # torch gerekmez, hafif

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"items": items, "text": full_text}, f, ensure_ascii=False, indent=2)
        base = output_path.rsplit(".", 1)[0]
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        with open(base + ".hocr", "w", encoding="utf-8") as f:
            f.write(_make_hocr(items, width, height))

        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "ocr",
                "text": full_text,
                "item_count": len(items),
                "hocr_path": base + ".hocr",
            },
        }

    # -- segmentasyon (SAM + Grounding DINO ile metinden segment) -----------

    def _get_sam(self, model_path: str):
        key = f"sam:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise _import_error("torch/transformers bulunamadı", exc) from exc

        repo = _resolve_repo(model_path)
        model = SamModel.from_pretrained(repo)
        processor = SamProcessor.from_pretrained(repo)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        self._cache[key] = (model, processor)
        return model, processor

    def _get_grounding_dino(self, model_path: str):
        key = f"dino:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise _import_error("torch/transformers bulunamadı", exc) from exc

        repo = _resolve_repo(model_path)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(repo)
        processor = AutoProcessor.from_pretrained(repo)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        self._cache[key] = (model, processor)
        return model, processor

    def _segment(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        mode = str(params.get("mode", "point"))
        if mode not in ("point", "box", "text"):
            raise RuntimeError(f"bilinmeyen segment mode: {mode!r} (point|box|text)")

        box: list[float] | None = None
        dino_score: float | None = None
        if mode == "box":
            box = params.get("box")
            if not box or len(box) != 4:
                raise RuntimeError("mode=box için box=[x1,y1,x2,y2] gerekli")
        elif mode == "text":
            text_prompt = str(params.get("text_prompt", "")).strip()
            if not text_prompt:
                raise RuntimeError("mode=text için text_prompt gerekli")
            dino_path = params.get("dino_model") or _DINO_DEFAULT
            yield {"event": "progress", "value": 0.1, "message": "Grounding DINO yükleniyor…"}
            dino_model, dino_processor = self._get_grounding_dino(dino_path)

            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # HF örnekleri: sorgu küçük harf + nokta ile bitmeli
            query = text_prompt.lower().rstrip(".") + "."
            dino_inputs = dino_processor(images=image, text=query, return_tensors="pt").to(device)
            yield {"event": "progress", "value": 0.3, "message": "nesne aranıyor…"}
            with torch.no_grad():
                dino_out = dino_model(**dino_inputs)
            detected = dino_processor.post_process_grounded_object_detection(
                dino_out,
                input_ids=dino_inputs.input_ids,
                # Varsayılan eşik 0.35→0.4: canlı testte "spaceship" gibi sahnede
                # olmayan nesneler için bile neredeyse tüm kareyi kaplayan düşük
                # güvenli kutular sızıyordu — eşik yükseltilip DINO skoru sonuca
                # eklendi ki kullanıcı zayıf eşleşmeyi görebilsin.
                box_threshold=float(params.get("box_threshold", 0.4)),
                text_threshold=float(params.get("text_threshold", 0.3)),
                target_sizes=[(image.height, image.width)],
            )[0]
            if len(detected["boxes"]) == 0:
                raise RuntimeError(
                    f"'{text_prompt}' için bölge bulunamadı (eşikleri düşürmeyi deneyin)"
                )
            best = int(detected["scores"].argmax())
            box = [float(v) for v in detected["boxes"][best].tolist()]
            dino_score = float(detected["scores"][best])

        sam_path = params.get("sam_model") or _SAM_DEFAULT
        yield {"event": "progress", "value": 0.5, "message": "SAM yükleniyor…"}
        sam_model, sam_processor = self._get_sam(sam_path)

        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if mode == "point":
            points = params.get("points")
            if not points:
                raise RuntimeError("mode=point için points=[[x,y],...] gerekli")
            labels = params.get("point_labels") or [1] * len(points)
            sam_inputs = sam_processor(
                image,
                input_points=[[list(p) for p in points]],
                input_labels=[list(labels)],
                return_tensors="pt",
            ).to(device)
        else:
            assert box is not None
            sam_inputs = sam_processor(image, input_boxes=[[list(box)]], return_tensors="pt").to(
                device
            )

        yield {"event": "progress", "value": 0.8, "message": "maske çıkarılıyor…"}
        with torch.no_grad():
            sam_out = sam_model(**sam_inputs)
        masks = sam_processor.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )
        scores = sam_out.iou_scores[0, 0]
        best_idx = int(scores.argmax())
        mask_tensor = masks[0][0, best_idx]

        mask_img = Image.fromarray((mask_tensor.numpy() * 255).astype("uint8"), mode="L")
        mask_img.save(output_path)

        result: dict[str, Any] = {
            "path": output_path,
            "op": "segment",
            "mode": mode,
            "width": mask_img.width,
            "height": mask_img.height,
            # SAM'in maskeye güveni (kutu/nokta ne olursa olsun genelde yüksektir)
            "score": float(scores[best_idx]),
        }
        if box is not None:
            result["box"] = box
        if dino_score is not None:
            result["dino_score"] = dino_score  # Grounding DINO'nun metin eşleşme güveni

        if bool(params.get("cutout", False)):
            cutout = image.convert("RGBA")
            cutout.putalpha(mask_img)
            cutout_path = output_path.rsplit(".", 1)[0] + "_cutout.png"
            cutout.save(cutout_path)
            result["cutout_path"] = cutout_path

        yield {"event": "done", "result": result}

    # -- kırpma (saf PIL, torch gerekmez) ------------------------------------

    def _crop(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        from PIL import Image

        box = params.get("box")
        if not box or len(box) != 4:
            raise RuntimeError("crop için box=[x1,y1,x2,y2] gerekli")
        yield {"event": "progress", "value": 0.3, "message": "kırpılıyor…"}
        image = Image.open(image_path).convert("RGB")
        x1, x2 = sorted((max(0, int(box[0])), min(image.width, int(box[2]))))
        y1, y2 = sorted((max(0, int(box[1])), min(image.height, int(box[3]))))
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("geçersiz kırpma alanı")
        out = image.crop((x1, y1, x2, y2))
        out.save(output_path)
        yield {
            "event": "done",
            "result": {"path": output_path, "op": "crop", "width": out.width, "height": out.height},
        }

    # -- temel ayarlar (saf PIL, torch gerekmez) -----------------------------

    def _adjust(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        from PIL import Image, ImageEnhance

        yield {"event": "progress", "value": 0.3, "message": "uygulanıyor…"}
        image = Image.open(image_path).convert("RGB")
        enhancers = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
            "sharpness": ImageEnhance.Sharpness,
        }
        for name, cls in enhancers.items():
            factor = params.get(name)
            if factor is not None:
                image = cls(image).enhance(float(factor))
        image.save(output_path)
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "adjust",
                "width": image.width,
                "height": image.height,
            },
        }

    # -- yüz restorasyonu (GFPGAN) -------------------------------------------

    def _get_gfpganer(self, model_path: str):
        key = f"face_restore:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import sys
            import types

            import torch
            import torchvision.transforms.functional as _tvf

            # basicsr, yeni torchvision'larda kaldırılan functional_tensor modülünü
            # import etmeye çalışıyor (üst akışta bilinen, yaygın bir sorun). Gerçek
            # torchvision paketine DOKUNMADAN, yalnızca bu worker sürecinde geçerli
            # bir uyumluluk shim'i sağlanıyor — sistem kurulumunu etkilemez.
            if "torchvision.transforms.functional_tensor" not in sys.modules:
                shim = types.ModuleType("torchvision.transforms.functional_tensor")
                shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
                sys.modules["torchvision.transforms.functional_tensor"] = shim
            from gfpgan import GFPGANer
        except ImportError as exc:
            raise _import_error("gfpgan bulunamadı", exc) from exc

        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,  # yalnız yüz restorasyonu; büyütme ayrı 'upscale' op'unda
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self._cache[key] = restorer
        return restorer

    def _face_restore(self, params: dict, image_path: str, output_path: str) -> Iterator[dict]:
        model_path = params.get("model_path", "")
        if not model_path:
            raise RuntimeError(
                "face_restore için model_path gerekli (Model Browser'dan bir GFPGAN modeli seçin)"
            )
        yield {"event": "progress", "value": 0.1, "message": "model yükleniyor…"}
        restorer = self._get_gfpganer(model_path)

        import cv2
        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        yield {"event": "progress", "value": 0.5, "message": "yüzler onarılıyor…"}
        _cropped, restored_faces, restored_img = restorer.enhance(
            bgr,
            has_aligned=False,
            only_center_face=bool(params.get("only_center_face", False)),
            paste_back=True,
            weight=float(params.get("weight", 0.5)),
        )
        if restored_img is None:
            raise RuntimeError("görüntüde yüz bulunamadı")

        out_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(out_rgb).save(output_path)

        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "face_restore",
                "face_count": len(restored_faces),
                "width": int(restored_img.shape[1]),
                "height": int(restored_img.shape[0]),
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
        return {
            "bg_remove": 2000,
            "upscale": 2000,
            "caption": 1500,
            "ocr": 1500,
            "segment": 2500,  # SAM + Grounding DINO ikisi de belleğe alınabilir
            "crop": 0,
            "adjust": 0,
            "face_restore": 2000,
        }.get(op, 1000)
