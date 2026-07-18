# ADR-0007: Model Formatları — safetensors Varsayılan

**Durum:** accepted · 2026-07-18

**Karar:** safetensors varsayılan; GGUF, ONNX, TensorRT engine cache desteklenir. Pickle
(.ckpt/.pt) karantina + açık kullanıcı onayı ile yüklenir.

**Gerekçe:** Pickle deserializasyonu RCE riskidir; safetensors + GGUF hem güvenli hem hızlı.

**Sonuçlar:** Model Manager (Faz 2) format tespiti ve karantina akışını uygular; Faz 14'te
sıkılaştırılır.
