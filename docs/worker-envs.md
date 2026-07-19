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

## Opsiyonel image env ekleri (Faz 4 sonu)

```bash
# ControlNet 'canny' ön işlemcisi için (saf wheel, torch'a dokunmaz)
uv pip install --python ~/LocalAIStudio/envs/image/Scripts/python.exe opencv-python-headless
```

- `accelerate` zaten kurulu (0.34.2) — `vram_mode: balanced/low` offload bunu kullanır.
- IP-Adapter ağırlıkları ilk kullanımda HF cache'e iner (`h94/IP-Adapter`);
  ControlNet modeli iş parametresinde yol ya da HF repo id olarak verilir
  (ör. `diffusers/controlnet-canny-sdxl-1.0` — ilk kullanımda ~2.5 GB indirir).
- FLUX/SD3 tek dosya checkpoint'leri text encoder'ları HF'den tamamlar; ilk
  kullanım ağ ister. `model_family` verilmezse aile dosya başlığından algılanır.

## Faz 5 — görüntü araç seti ekleri (envs/image ile PAYLAŞILIR)

`imgtool` iş türü kendi venv'ini doğurmaz; scheduler'daki `_ENV_ALIAS` sözlüğü
onu `envs/image`'e yönlendirir (aynı torch/transformers zaten yeterli).

```bash
uv pip install --python ~/LocalAIStudio/envs/image/Scripts/python.exe \
  spandrel easyocr einops timm
```

- `spandrel`: RealESRGAN/ESRGAN ailesini `basicsr`'a ihtiyaç duymadan yükler
  (basicsr, yeni torchvision'larda kaldırılan `functional_tensor` modülünü
  import etmeye çalıştığı için kırık — bu yüzden bilinçli olarak atlandı).
- `easyocr`: torch tabanlı OCR; `paddlepaddle-gpu` yerine tercih edildi çünkü
  paddle'ın CUDA uyumluluğu RTX 50xx gibi yeni mimarilerde belirsiz, ayrı bir
  CUDA runtime paketi gerektiriyor — torch rulesına aykırı bir risk.
- `background_removal` varsayılanı **BiRefNet** (MIT lisans) — RMBG-2.0'ın
  ticari kısıtlı lisansından kaçınmak için tercih edildi (plan.md'nin verdiği
  iki seçenekten biri).
- `captioning` varsayılanı **Florence-2-base** (MIT lisans, ~460MB fp16).
- `upscale` modeli Model Browser'dan seçilir (zorunlu) — `.pth`/`.safetensors`
  ESRGAN ailesi ağırlıkları `models/upscale/` altına indirilir.
- `segment`: SAM (`facebook/sam-vit-base`) + Grounding DINO
  (`IDEA-Research/grounding-dino-tiny`) — ikisi de transformers'ta yerleşik,
  ek paket gerekmez. Plan.md "SAM2" adını verir; biz SAM v1 kullandık çünkü
  transformers 4.46.3'te native destekli (SAM2 ayrı bir pip paketi ve daha
  yeni transformers ister) — aynı nokta/kutu/açık-kelime-dağarcığı iş akışını
  sağlıyor, kalite farkı çoğu kullanım için gözle görülür değil.

```bash
# face_restore (GFPGAN) — basicsr'ı da beraberinde getirir
uv pip install --python ~/LocalAIStudio/envs/image/Scripts/python.exe gfpgan
```

- `face_restore`: GFPGAN. `basicsr` (gfpgan'ın bağımlılığı), yeni
  torchvision'larda kaldırılan `torchvision.transforms.functional_tensor`
  modülünü import etmeye çalıştığı için TEK BAŞINA kırık (bilinen üst akış
  sorunu). Adapter, gerçek torchvision paketine dokunmadan, yalnızca kendi
  worker sürecinde geçerli bir uyumluluk shim'i enjekte ediyor
  (`sys.modules["torchvision.transforms.functional_tensor"]`) — bkz.
  `imgtool.py::_get_gfpganer`. Model ağırlığı (`GFPGANv1.4.pth`, ~333 MB)
  Model Browser'dan `models/face_restoration/` altına indirilir; facexlib'in
  yüz tespit/hizalama modelleri ise kendi paket-içi varsayılan önbelleğine
  otomatik iner (Model Manager'ın dışında — bilinen kapsam dışı bırakma).

## Faz 6 — llm env (envs/llm, AYRI venv — torch/diffusers gerektirmez)

```bash
uv venv ~/LocalAIStudio/envs/llm \
  --python "C:\Users\Tars\AppData\Local\Programs\Python\Python310\python.exe" \
  --system-site-packages
uv pip install --python ~/LocalAIStudio/envs/llm/Scripts/python.exe \
  llama-cpp-python --prefer-binary \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Neden `--extra-index-url` gerekiyor:** `pip install llama-cpp-python` PyPI'da bu
platform için hazır wheel bulamayıp 71 MB'lık kaynak dağıtımını indiriyor; onu
açarken llama.cpp'nin gömülü web arayüzünün derin dizin yapısı Windows'un 260
karakter yol sınırına takılıyor (`LongPathsEnabled` sistem ayarını değiştirmeden
çözülemez — bilinçli olarak dokunulmadı). abetlen'in kendi paket indeksindeki
**hazır CUDA wheel'i** (536 MB, cu124) bu sorunu tamamen atlıyor.

**RTX 5090 (compute capability 12.0, Blackwell) uyumluluğu:** cu124 wheel'i
50-serisi çıkmadan derlenmiş olsa da `ggml-cuda.dll` (946 MB, çoklu SM hedefi
içeriyor) canlı testte GPU'yu doğru tanıdı: `ggml_cuda_init: found 1 CUDA
devices ... RTX 5090, compute capability 12.0, VRAM: 32606 MiB`.

**DLL yükleme sorunu ve çözümü:** wheel'in CUDA çalışma zamanı DLL'leri
(cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll) sistemde ayrıca kurulu
değil (CUDA toolkit kurulmadı — nvcc yok). llama.cpp'nin kendi yükleyicisi
`os.add_dll_directory()` ile eklenen yolları onurlandırmıyor (winmode farkı);
tek güvenilir çözüm gerçek `PATH` ortam değişkenine eklemek. Adapter bunu
`import torch` ile (zaten `--system-site-packages` sayesinde görülüyor, YENİDEN
KURULMUYOR) torch'un kendi cu128 DLL'lerinin konumunu bulup PATH'e ekleyerek
yapıyor — bkz. `llm.py::_ensure_loaded`. torch 12.x ailesi ABI uyumlu olduğu
için cu124 ile derlenmiş llama.cpp, torch'un cu128 çalışma zamanı DLL'leriyle
sorunsuz çalışıyor.

- Sohbet şablonu: GGUF'un gömülü `tokenizer.chat_template`'i llama.cpp
  tarafından otomatik algılanıyor (`chat_format` parametresi verilmiyor).
- `n_gpu_layers=-1` (vars.): tüm katmanlar GPU'ya — 32 GB VRAM'de küçük/orta
  GGUF modeller için sorunsuz.

## Faz 7 — audio env (envs/audio, AYRI venv)

```bash
uv venv ~/LocalAIStudio/envs/audio \
  --python "C:\Users\Tars\AppData\Local\Programs\Python\Python310\python.exe" \
  --system-site-packages
uv pip install --python ~/LocalAIStudio/envs/audio/Scripts/python.exe \
  faster-whisper kokoro soundfile
```

llama.cpp'nin aksine **ek bir PATH/DLL düzeltmesi gerekmedi** — hem
`ctranslate2` (faster-whisper'ın çalışma zamanı) hem `kokoro`'nun torch
bağımlılığı, CUDA'yı kendi başına buldu (`ctranslate2.get_cuda_device_count()`
→ 1, `torch.cuda.is_available()` → True). Muhtemelen bu wheel'ler CUDA çalışma
zamanını kendi paket içlerinde taşıyor (llama-cpp-python'ın cu124 wheel'inin
aksine, o dışarıdan cudart/cublas bekliyordu).

- `stt`: faster-whisper, `model_size` bir HF preset adı (tiny/base/small/
  medium/large-v3 — ilk kullanımda kendi HF cache'ine iner) ya da Model
  Browser'dan kayıtlı bir CTranslate2 model klasörü yolu olabilir.
- `tts`: Kokoro-82M (MIT lisans). İlk kullanımda `kokoro` paketi kendiliğinden
  bir spacy İngilizce modeli (`en_core_web_sm`, ~13 MB) indirir — metin→fonem
  (G2P) ayrıştırması için gerekli, tek seferlik.
- **Bilinen dil sınırı:** Kokoro'nun varsayılan sesleri (lang_code "a"/"b")
  yalnızca İngilizce fonemleştirme yapıyor. Türkçe metin bu seslerle
  seslendirilirse harfler İngilizce okunuş kurallarıyla telaffuz edilir
  (canlı testte yakalandı: "hoş geldiniz" → "Hosh Geldon is" gibi duyuluyor).
  Bu bir kod hatası değil, model kapsamı sınırı — Kokoro v1.0 sınırlı sayıda
  ek dil (ja/zh/es/fr/hi/it/pt) destekliyor, Türkçe şimdilik yok.
