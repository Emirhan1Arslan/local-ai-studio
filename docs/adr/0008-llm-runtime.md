# ADR-0008: LLM Runtime — llama.cpp Birincil

**Durum:** accepted · 2026-07-18

**Karar:** llama-cpp-python birincil; Transformers (safetensors LLM); opsiyonel vLLM
(Linux/CUDA); Ollama köprüsü (kurulu modelleri yeniden indirmeden kullan).

**Gerekçe:** GGUF her platformda (CUDA/ROCm/MPS/CPU) çalışır; Ollama köprüsü mevcut kullanıcı
kurulumlarını değerlendirir.

**Sonuçlar:** Tüm LLM aileleri (Qwen, Llama, Gemma, Mistral…) tek GGUF adapter'ıyla gelir
(Faz 6).
