"""Sohbet adapter'ı (Faz 6): llama.cpp ile GGUF modelleri, token streaming.

DİKKAT: Bu modül worker env'inde (envs/llm) çalışır. 3.11+ sözdizimi KULLANMA.
torch/llama_cpp importları metod içindedir; API süreci bu modülü hiç yüklemez.

Windows notu: pip'in llama-cpp-python için PyPI'daki kaynak dağıtımı, gömülü web
arayüzünün derin dizin yapısı yüzünden Windows'un 260 karakter yol sınırına takılıyor.
Çözüm: hazır CUDA wheel'i (docs/worker-envs.md). O wheel'in CUDA çalışma zamanı
DLL'leri (cudart/cublas) sistemde ayrıca kurulu değil — sistem torch'unun ZATEN
indirdiği aynı sürüm DLL'leri ödünç alınır (torch'a dokunulmaz, yalnızca konumu
okunup PATH'e eklenir). RTX 5090 (compute capability 12.0) canlı doğrulandı.

params:
- model_path: GGUF dosya yolu (zorunlu, Model Browser'dan)
- messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] (zorunlu)
- temperature (vars. 0.7) · max_tokens (vars. 1024) · n_ctx (vars. 8192)
- n_gpu_layers: -1 = tüm katmanlar GPU'ya (vars.)

Sohbet şablonu: GGUF'un gömülü tokenizer.chat_template'i llama.cpp tarafından
otomatik algılanır (chat_format parametresi verilmez) — plan.md'nin "otomatik
chat-template tespiti" maddesi bu sayede ek koda gerek kalmadan sağlanır.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def _import_error(hint: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(
        f"llm worker ortamı eksik ({hint}). "
        "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
    )


class LlamaAdapter:
    def __init__(self) -> None:
        self._llm: Any = None
        self._model_path = ""
        self._n_ctx = 0

    def load(self, model_path: str, **kwargs: Any) -> None:
        pass  # n_ctx/n_gpu_layers parametreye bağlı olduğundan infer() içinde yüklenir

    def _ensure_loaded(self, model_path: str, n_ctx: int, n_gpu_layers: int) -> Any:
        if self._llm is not None and self._model_path == model_path and self._n_ctx == n_ctx:
            return self._llm
        if self._llm is not None:
            self.unload()

        try:
            import os

            import torch  # yalnızca CUDA runtime DLL'lerinin konumunu okumak için

            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            path = os.environ.get("PATH", "")
            if torch_lib not in path:
                os.environ["PATH"] = torch_lib + os.pathsep + path
            from llama_cpp import Llama
        except ImportError as exc:
            raise _import_error("torch/llama_cpp bulunamadı", exc) from exc

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._model_path = model_path
        self._n_ctx = n_ctx
        return self._llm

    def infer(self, params: dict) -> Iterator[dict]:
        model_path = params.get("model_path", "")
        if not model_path:
            raise RuntimeError("model_path gerekli (Model Browser'dan bir GGUF seçin)")
        messages = params.get("messages")
        if not messages:
            raise RuntimeError("messages gerekli")

        n_ctx = int(params.get("n_ctx", 8192))
        n_gpu_layers = int(params.get("n_gpu_layers", -1))
        temperature = float(params.get("temperature", 0.7))
        max_tokens = int(params.get("max_tokens", 1024))

        yield {"event": "progress", "value": 0.0, "message": "model yükleniyor…", "delta": ""}
        llm = self._ensure_loaded(model_path, n_ctx, n_gpu_layers)

        chunks: list[str] = []
        count = 0
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if not delta:
                continue
            count += 1
            chunks.append(delta)
            yield {
                "event": "progress",
                "value": min(1.0, count / max_tokens),
                "message": f"{count} token",
                "delta": delta,
            }

        text = "".join(chunks)
        yield {"event": "done", "result": {"text": text, "tokens": count}}

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._model_path = ""
            self._n_ctx = 0

    def vram_estimate(self, params: dict) -> int:
        return 6000  # kaba varsayım — gerçek kullanım GGUF boyutu + n_ctx'e göre değişir
