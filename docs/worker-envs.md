# Worker Ortamları (ADR-0005)

Ağır ML bağımlılıkları ana venv'e **girmez**; iş türü başına izole env kullanılır.
Scheduler, worker'ı `~/LocalAIStudio/envs/<tür>/Scripts/python.exe` ile başlatır
(override: `STUDIO_WORKER_PYTHONS` JSON'ı), kodu PYTHONPATH ile kaynaktan import eder —
worker env'ine studio paketi kurulmaz.

## İlke: mevcut kullanıcı ortamlarına dokunma

Kullanıcının sistem Python'ında torch kuruluysa (bu makinede: **Python 3.10 +
torch 2.11.0+cu128 + transformers 4.46.3**), o ortam **asla değiştirilmez**.
Venv `--system-site-packages` ile oluşturulur: sistem paketlerini görür, kurulumlar
yalnızca venv'e yazılır (shadowing).

## image env reçetesi (doğrulanmış)

```bash
uv venv ~/LocalAIStudio/envs/image \
  --python "C:\Users\Tars\AppData\Local\Programs\Python\Python310\python.exe" \
  --system-site-packages
uv pip install --python ~/LocalAIStudio/envs/image/Scripts/python.exe \
  "diffusers==0.32.2" "huggingface-hub<1.0"
```

**Sürüm pinlerinin nedeni** (sürüm cehennemi — plan.md §8 risk #1):
- `diffusers>=0.39` → `transformers>=4.48` sınıfları ister (`Dinov2WithRegistersConfig`);
  sistemde 4.46.3 var → ImportError.
- `diffusers 0.39` kurulumu `huggingface-hub 1.24`'ü çeker; `transformers 4.46` ise
  `hub<1.0` ister → çalışma zamanı hatası.
- `diffusers==0.32.2` + `hub 0.36.2` + `transformers 4.46.3` + `torch 2.11+cu128`
  kombinasyonu SDXL için uçtan uca doğrulandı (RTX 5090, seed determinizmi dahil).

Sistem transformers'ı ileride yükselirse pin gevşetilebilir; Faz 15 ilk-çalıştırma
sihirbazı bu reçeteyi otomatikleştirecek.
