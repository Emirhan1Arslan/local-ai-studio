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
  ek dil (ja/zh/es/fr/hi/it/pt) destekliyor, Türkçe şimdilik yok. Çok dilli/
  Türkçe ihtiyaç için aşağıdaki XTTS-v2 motoru kullanılır.

### XTTS-v2 (ses klonlama, çok dilli — Türkçe dahil)

```bash
uv pip install --python ~/LocalAIStudio/envs/audio/Scripts/python.exe \
  --no-deps coqui-tts
uv pip install --python ~/LocalAIStudio/envs/audio/Scripts/python.exe \
  --no-deps torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python ~/LocalAIStudio/envs/audio/Scripts/python.exe \
  --no-deps "transformers==4.57.6" "huggingface-hub==0.36.2"
uv pip install --python ~/LocalAIStudio/envs/audio/Scripts/python.exe \
  --no-deps torchcodec
```

**Neden bu kadar çok `--no-deps` + pin var** (sürüm cehennemi, bu kez `envs/audio`'da):
- Sistemde torch var (2.11.0+cu128) ama **torchaudio hiç yok** — coqui-tts
  onu import ediyor. PyPI'daki torchaudio, pip'in serbest çözümüyle uyuşmayan
  bir sürüm çekebileceği için doğrudan torch ile birebir eşleşen sürüm
  (`2.11.0+cu128`) PyTorch'un kendi cu128 wheel index'inden `--no-deps` ile
  kuruldu.
- `coqui-tts`'in taban bağımlılığı `transformers>=4.57` — pin'siz kurulum en
  yeni sürümü (`5.14.1`, büyük major sürüm) çekiyor; ama XTTS'in tortoise
  katmanı (`TTS/tts/layers/tortoise/autoregressive.py`) transformers 5.x'te
  kaldırılmış bir fonksiyonu (`isin_mps_friendly`) import ediyor →
  `ImportError`. Çözüm: son 4.x sürümüne (`4.57.6`) sabitlemek — coqui-tts'in
  kendi `>=4.57` şartını hâlâ karşılıyor.
- `transformers==4.57.6` ise `huggingface-hub>=0.34.0,<1.0` istiyor; coqui-tts
  kurulumunun çektiği `hub 1.24.0` bunu bozuyor → son 0.x sürümüne
  (`0.36.2`) indirildi.
- torch 2.9+ için coqui-tts, ses I/O'sunu `torchcodec` üzerinden yapıyor;
  paket PyPI'da normal `pip install` ile geliyor ama **FFmpeg'in paylaşımlı
  DLL'lerini** (avcodec/avformat/avutil…) çalışma zamanında arıyor — bunlar
  Python paketi değil, ayrı bir sistem aracı.

**FFmpeg (sistem aracı, Python/torch'a dokunmaz):** Bu makinede winget
kabuktan görünmediği için (bkz. üstteki ortam notu) BtbN'nin FFmpeg 7.1
"shared" build'i (DLL'ler ayrı, GPL) doğrudan indirilip
`~/LocalAIStudio/tools/ffmpeg/` altına açıldı — torchcodec 0.15.0'ın aradığı
`libtorchcodec_core7.dll` FFmpeg 7.x ABI'siyle eşleşiyor. `adapters/audio.py`
içindeki `_ensure_ffmpeg_on_path()`, `tools/ffmpeg/bin`'i (varsa) gerçek
`PATH`'e ekliyor — llm.py'nin torch-DLL çözümüyle birebir aynı desen
(`os.add_dll_directory` değil, gerçek `PATH`; winmode farkı nedeniyle).

Tüm zincir doğrulandı: `torch.__file__` hâlâ sistem yolunu gösteriyor (pip
her defasında sistem kopyasını "outside environment" diyerek reddetti, yalnız
venv'e shadow kurdu); Kokoro ve faster-whisper regresyon testleri bu
değişikliklerden sonra da temiz geçti.

- `engine=xtts` params: `language` (vars. "en" — tr dahil ~17 dil) + ikisinden
  biri zorunlu: `speaker` (checkpoint'e gömülü 58 hazır sesten biri, ör. "Ana
  Florence" — klonlama değildir, `consent` gerekmez) ya da `speaker_wav`
  (klonlanacak referans ses; verilirse `consent=true` ZORUNLU — UI'da bir
  onay kutusuyla toplanır, plan.md Faz 7 adım 4). Hazır ses listesi
  `tts.speakers` özelliğinden okunur, bu modele özgü sabit bir listedir.
- **LİSANS:** XTTS-v2 model ağırlıkları **Coqui Public Model License
  (CPML)** ile geliyor — yalnızca ticari OLMAYAN kullanım. `COQUI_TOS_AGREED=1`
  adapter içinde otomatik ayarlanır (worker sürecinde etkileşimli prompt
  mümkün değil); lisans metni kullanıcıya UI'da (Ses paneli, XTTS motoru
  seçilince) ayrıca gösterilir.
- Canlı doğrulama (RTX 5090): Türkçe metin + İngilizce referans ses →
  klonlanmış Türkçe konuşma (model yükleme 9.4sn, sentez 3.1sn) →
  faster-whisper `language=tr` ile geri transkribe edildi, dil algılama
  %100 güvenle "tr", metin kaynağa yakın eşleşti (round-trip doğrulandı).

### Ses modelleri Model Manager'a entegrasyonu

**Sorun:** faster-whisper/Kokoro/coqui-tts başlangıçta kendi kendine indiriyordu
(HF Hub'ın global `~/.cache/huggingface/hub` önbelleğine ya da coqui'nin kendi
`get_user_data_dir()` konumuna) — bu, görüntü araçlarının (`imgtool.py`)
kullandığı `model_path` + Model Browser deseninden tamamen kopuktu, dolayısıyla
ses modelleri Modeller panelinde hiç görünmüyordu.

**Çözüm — üç kütüphanenin de yerel-klasörden yükleme desteği doğrulandı:**
- faster-whisper: `WhisperModel(model_size_or_path, ...)` zaten bir preset adı
  ya da yerel bir CTranslate2 klasör yolu kabul ediyor — ek kod gerekmedi.
- Kokoro: `KModel(repo_id=, config=<yerel config.json>, model=<yerel .pth>)` +
  `KPipeline(..., model=kmodel)` tamamen yerelden yükler; ses paketleri de
  `voice="<tam .pt yolu>"` verilirse (`.pt` ile bitiyorsa) `hf_hub_download`'a
  hiç gitmez (`kokoro/pipeline.py::load_single_voice`).
- coqui-tts: `TTS(model_path=<yerel klasör>, config_path=<yerel config.json>)`
  kendi `ModelManager` önbelleğini tamamen atlar.

`adapters/audio.py`'deki `_stt`/`_tts_kokoro`/`_tts_xtts` artık opsiyonel bir
`model_path` parametresi kabul ediyor (Model Browser'dan seçilen ağırlık
dosyası) — verilirse `os.path.dirname()` ile üst klasöre çözülüp yukarıdaki
yerel-yükleme yollarına yönlendiriliyor; verilmezse eski self-download
davranışı (geriye dönük uyumlu) devam ediyor.

**`downloads.py` genellemesi:** `/v1/downloads/snapshot`, önceden yalnızca düz
yapılı (alt klasörsüz) HF repo'larını destekliyordu — Kokoro-82M'nin
`voices/*.pt` alt klasörü bu kısıtlamaya takılıyordu. `_safe_dest()`'e
`preserve_subpath` parametresi eklendi: snapshot indirmede (dosya adları
`hf_list_repo_files`'tan gelen güvenilir bir listedir) repo-içi göreli yol
korunuyor (`models/<subdir>/<repo>/voices/af_heart.pt` gibi), path-traversal
koruması (`..`/mutlak yol reddi) aynen duruyor. Tekil dosya indirme
endpoint'i (`POST /v1/downloads`, dış istemcilere de açık — ADR-0003)
değişmedi, hâlâ yalnızca düz dosya adı kabul ediyor.

**Bu makinede canlı olarak yerleştirildi** (indirme motorunu yeniden
çalıştırmak yerine, zaten kurulum sırasında doğrulanmış kütüphane
önbelleklerinden doğrudan kopyalandı):
- `models/tts/Kokoro-82M/{config.json,kokoro-v1_0.pth,voices/*.pt}` (7 ses:
  af_heart, af_bella, af_nicole, am_adam, am_michael, bf_emma, bm_george)
- `models/tts/XTTS-v2/{config.json,model.pth,speakers_xtts.pth,vocab.json}`
- `models/stt/faster-whisper-base/{config.json,model.bin,tokenizer.json,vocabulary.txt}`

Çalışan uygulamanın canlı `ModelWatcher`'ı (watchdog) bu dosyaları anlık
algılayıp DB'ye kaydetti — ayrı bir rescan adımı gerekmedi. Doğrulama:
`HF_HUB_OFFLINE=1` + boş `HF_HOME` ile üçü de (Kokoro TTS, XTTS-v2 TTS,
faster-whisper STT) `model_path` üzerinden hiçbir ağ isteği yapmadan başarıyla
çalıştı — gerçek "tamamen yerel" yükleme kanıtlandı.

## Faz 8 — video env (envs/image ile PAYLAŞILIR)

`video`/`videotool` iş türleri kendi venv'ini doğurmaz; `scheduler._ENV_ALIAS`
onları `envs/image`'e yönlendirir. Neden: LTX-Video (plan.md'nin kademeli video
model listesindeki ilk/varsayılan, en hafif model) diffusers 0.32.2 içinde
zaten `LTXPipeline`/`LTXImageToVideoPipeline` olarak hazır — ayrı bir izole env
kurmaya gerek kalmadı.

```bash
# Tek eksik: T5 tokenizer'ın ihtiyaç duyduğu sentencepiece (saf wheel, torch'a dokunmaz)
uv pip install --python ~/LocalAIStudio/envs/image/Scripts/python.exe sentencepiece
```

**Neden gerekiyor:** `LTXPipeline.from_pretrained(...)` T5Tokenizer'ı yüklemeye
çalışırken `ImportError: T5Tokenizer requires the SentencePiece library`
fırlattı — `transformers`'ın `dummy_sentencepiece_objects` şablonu devreye
girdi. `sentencepiece` kendi başına duran (bağımlılıksız) bir C++ wheel'i,
`--no-deps` gerek kalmadan güvenle kuruldu (canlıda doğrulandı: kurulumdan
sonra `torch.__file__` hâlâ sistem yolunu gösteriyor).

**Video encode — kendi FFmpeg çözümü, ek bağımlılık yok:** diffusers'ın
`export_to_video()` yardımcısı `imageio-ffmpeg` paketini istiyor (denendi,
eksik olduğu canlıda doğrulandı: `AttributeError: ... imageio backend`).
Bunun yerine `adapters/video.py::_encode_mp4()` kareleri geçici bir klasöre
PNG olarak yazıp Faz 7'de kurulan FFmpeg build'iyle (`adapters/_ffmpeg.py`,
artık audio/video/videotool arasında paylaşılan ortak modül) doğrudan H.264
MP4'e kodluyor — yeni bir Python paketi gerekmedi.

**Canlı doğrulama (RTX 5090, `Lightricks/LTX-Video`, ~20 GB/18 dosya, ilk
indirmede tek seferlik):**
- Model yükleme: 43.7 sn (soğuk, diskten okuma) · VRAM sonrası: 14.2 GB
- Üretim: 41 kare, 30 adım, 512×320 → **3.3 sn** (LTX gerçekten hafif — RTX
  5090'da bellek stratejisi/offload gerekmeden rahat çalışıyor) · VRAM tepe:
  14.95 GB (32 GB'nin yarısından az — plan.md'nin "en VRAM-yoğun faz" uyarısı
  LTX için değil, kademeli listenin sonundaki ağır modeller için geçerli).
- FFmpeg encode ile üretilen MP4 doğrulandı: h264, 512×320, 25 fps, 1.64 sn
  (41/25 ile birebir eşleşti).
- i2v (image-to-video) da aynı şekilde canlı doğrulandı (55.8 sn, model zaten
  bellekteyken ~ilk yüklemeyle aynı — kalıcı worker deseni sayesinde sonraki
  işler çok daha hızlı olacak).

**videotool bulgusu (canlı testte yakalandı, düzeltildi):** LTX-Video çıktısı
ses izi taşımıyor. `speed`/`add_audio` op'larının FFmpeg filtre grafikleri
başlangıçta koşulsuz `[0:a]` (ses akışı) referansı içeriyordu — ses izsiz bir
video verildiğinde `Stream specifier ':a' ... matches no streams` ile
patlıyordu. Çözüm: `adapters/_ffmpeg.py::has_audio_stream()` (ffprobe ile
yoklama) eklendi, her iki op da ses izi yoksa yalnız-video bir filtre
grafiğine düşüyor. `trim`/`concat`/`convert` (gif dahil) op'ları da canlıda
doğrulandı.

### Video modelleri Model Manager entegrasyonu

Ses modelleri entegrasyonundan (yukarı) farklı bir desen gerekti: LTX-Video
tek dosyalık bir checkpoint değil, **çok-bileşenli bir diffusers pipeline'ı**
— `model_index.json` + `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`,
`scheduler/` alt klasörleri (bazı bileşenler parçalı: text_encoder 4, transformer
2 `.safetensors` parçası). Tarayıcı (Faz 2) her ağırlık dosyasını ayrı bir
`ModelRecord` olarak listeler — bu repo için 7 satır (canlıda doğrulandı: 4
text_encoder + 2 transformer + 1 vae).

**Çözüm:** `adapters/video.py::_resolve_pipeline_root()` — Model Browser'da
HANGİ bileşen seçilirse seçilsin (transformer'ın bir parçası, vae, text_encoder
parçalarından biri...) en fazla 3 üst klasöre kadar `model_index.json` arar ve
bulduğu kökü `LTXPipeline.from_pretrained()`'a verir; bulamazsa tek-bileşenli
desene (sdxl.py'nin `_resolve_repo`'suyla aynı: `config.json` kardeşi varsa üst
klasör) düşer, o da yoksa yol/HF repo id'sini olduğu gibi kullanır. Üç farklı
bileşen dosyası üzerinde test edildi, üçü de aynı köke çözüldü.

**Not (sonradan basitleşti):** İlk sürümde UI'da 7 satırlık ham listeyi
gizlemek için `VideoPanel.tsx`'te yalnızca `vae` bileşenini filtreleyen bir
client-side geçici çözüm vardı. `studio_models/metadata.py::group_key()`
eklenince (bkz. plan.md Faz 2 düzeltmesi) tarayıcı zaten TEK bir "LTX-Video"
kaydı döndürmeye başladı — bu client-side filtre artık gereksizdi ve
kaldırıldı, `VideoPanel.tsx` şimdi `fetchModels("video")` sonucunu doğrudan
kullanıyor.

**Bu makinede canlı olarak yerleştirildi:** `models/video/LTX-Video/` (indirilen
~27 GB'lık tam snapshot, HF cache'inden doğrudan kopyalandı — indirme motoru
yeniden çalıştırılmadı, Faz 7'deki ses modelleri deseniyle aynı). Doğrulama:
`HF_HUB_OFFLINE=1` + boş `HF_HOME` ile, Model Browser'dan seçilmiş gibi bir
bileşen dosyası `model_path` olarak verilip gerçek bir üretim hiçbir ağ
isteği yapmadan başarıyla tamamlandı.

### Faz 8 — kademeli model listesi genişletmesi (Wan/CogVideoX/Hunyuan/Mochi)

**Diffusers pin güncellemesi (0.32.2 → 0.33.1):** Wan 2.x diffusers'ta ancak
0.33+ ile geldi. `--no-deps` ile kuruldu (metadata kontrolü: `transformers`
yalnızca `extra == "dev"/"test"` altında isteniyor, taban kuruluma etkisi yok).
Canlı regresyon: `torch.__file__`/`transformers.__file__` hâlâ sistem yolunu
gösteriyor; SDXL (txt2img, 4 adım) VE LTX-Video (t2v, 8 adım) ikisi de
0.33.1 altında sorunsuz üretim yaptı — mevcut hiçbir pipeline bozulmadı.

**Mimari:** `adapters/video.py`'deki `_FAMILIES` tablosu artık 5 aileyi
(ltx/wan/cogvideox/hunyuan/mochi) tek bir `VideoAdapter` üzerinden yönetiyor —
her aile kendi pipeline sınıflarını (`t2v_cls`/`i2v_cls`), varsayılan HF
repo'sunu, `negative_prompt` desteğini (Hunyuan'da yok) ve varsayılan
çözünürlük/kare/adım/CFG değerlerini tanımlıyor. `_resolve_pipeline_root()`
üst klasör arama derinliği 3'ten 4'e çıkarıldı (Mochi/Hunyuan'ın daha derin
transformer/text_encoder yapıları için).

**Adım 2 (uzun iş yönetimi) — canlıda LTX ile doğrulandı:**
- Video/videotool işleri, istemci açıkça belirtmediyse (vars. 0) öncelik -1'e
  düşer (`routes/jobs.py::_create_job_row`) — hızlı iş türleri kuyrukta önce
  koşar.
- **Ara kare önizlemesi:** `_latent_preview_jpeg()` — birkaç adımda bir,
  mevcut latent'in ilk karesi pipeline'ın KENDİ VAE'siyle gerçek decode
  edilip küçük bir JPEG önizlemesi üretiliyor (SDXL'in ucuz doğrusal
  yaklaşımının aksine — video VAE'leri bunu desteklemiyor). LTX'e özgü iki
  gerçek hata canlı testte yakalanıp düzeltildi: (1) LTX latentleri
  düzleştirilmiş/patchified `[B,S,D]` tutuyor, pipeline'ın kendi
  `_unpack_latents`/`_denormalize_latents` yardımcıları gerekiyor; (2)
  latentler fp32, VAE bf16 — decode'dan önce `lat.to(vae.dtype)` gerekti.
  Canlı doğrulama: üretilen önizleme JPEG'i gerçekten "kırmızı spor araba
  dağ yolunda" istemine uyan tutarlı bir görüntüydü.
- Tahmini süre göstergesi: `estimate_seconds()` — LTX için ölçülen (13
  adım/sn), diğer 4 aile için model boyutuna göre kaba tahmin (canlı
  ölçüldükçe güncellenecek).

**Adım 3 (bellek stratejileri) — otomatik çözünürlük önerisi:**
`recommend_settings()` toplam VRAM'i algılar, ailenin kaba tahminine göre
yetersizse çözünürlük/kare sayısını orantılı küçültüp `vram_mode`'u en az
`balanced`'a yükseltir. 32 GB'lık bu makinede LTX için devreye girmiyor
(zaten yarısından azını kullanıyor); torch yokken (test ortamı gibi) güvenli
no-op'a düşer. fp8 nicemleme bu dilimde YOK — `torchao` + RTX 5090
(Blackwell) özel doğrulama gerektiriyor, kapsam dışı bırakıldı.

**Adım 4 (video upscale) — canlıda doğrulandı:** `videotool.py`'ye `upscale`
op'u eklendi — imgtool.py'nin spandrel/ESRGAN motorunu birebir yeniden
kullanıyor (kareleri FFmpeg ile PNG'ye çıkarır, her birini ayrı ayrı büyütür,
orijinal fps + ses izini koruyarak yeniden kodlar). Canlı test: 256×256 →
1024×1024 (RealESRGAN_x4plus, 4x), fps ve süre birebir korundu. **RIFE kare
interpolasyonu YOK:** `rife-ncnn-vulkan-python` PyPI'da yalnızca kaynak
dağıtım (.tar.gz) olarak mevcut — Vulkan SDK + C++ derleyici zinciri
gerektiriyor, bu ortamda güvenle derlenemedi.

**Adım 5 (timeline UI):** `VideoPanel.tsx`'in düzenleme sekmesine gerçek bir
video scrubber eklendi — kırpma (trim) için yüklenen kaynak videonun yerel
blob URL'i (`URL.createObjectURL`, sunucu yüklemesi bitmeden anında
oynatılabilir) bir `<video>` öğesinde gösterilip iki `range` slider'ı
`currentTime`'ı canlı güncelleyerek kare-kare önizleme sağlıyor.

**İndirme hızı sınırlaması (kullanıcı isteğiyle, gerçek Downloads Manager'a
entegre edildi):** HF Hub'ın yeni Xet depolama istemcisi (`hf_xet` paketi)
Python'ın `requests` oturumunu tamamen atlayıp kendi Rust HTTP istemcisiyle
indiriyor — `configure_http_backend` ile kurulan bir hız sınırlayıcı
session'ı görmüyor. Çözüm: `HF_HUB_DISABLE_XET=1` ortam değişkeniyle Xet
devre dışı bırakılıp klasik `requests` tabanlı indirmeye düşülüyor.

İlk turda ad-hoc bir `HTTPAdapter` sarmalayıcı script'i (scratchpad) ile
denendi; kullanıcı bunun yerine **kalıcı, uygulama-içi** bir çözüm istedi
("hız limitini ben ayarlayabileyim, indirmeleri kendi Downloads sayfamdan
görüp duraklatıp devam ettirebilmeliyim"). Sonuç:

- `download_speed_limit_mbps` — yeni Settings alanı (`config.py`,
  `settings_store.py` `EDITABLE_KEYS`, `routes/settings.py`), 0 = sınırsız.
- `_RateLimiter` (`studio_models/downloader.py`) — sanal-zaman damgalı token
  bucket, TÜM eşzamanlı indirmeler için TEK toplam (aggregate) limit
  uyguluyor (indirme başına değil). Her chunk'ta `get_settings()`'i taze
  okuyor — limit değişikliği restart gerektirmeden anında etkili oluyor.
  `DownloadManager.reset()` limiter'ı yeniden kuruyor çünkü `asyncio.Lock`
  event-loop'a bağlı.
- `DownloadsPanel.tsx` — hız limiti girişi (500ms debounce), gerçek
  indirmeler artık BURADAN görülüp duraklatılıp/devam ettirilip iptal
  edilebiliyor; ayrıca dosyalar `models_dir` altındaki repo-slug'a göre
  gruplanıp (`groupKeyFor`) tek bir model başlığı altında toplanıyor, her
  dosya genişletildiğinde ayrı ayrı hâlâ izlenebiliyor/duraklatılabiliyor
  (bu gruplama salt UI katmanı — DB kayıtları ayrı ayrı kalıyor; Model
  Manager'daki fiziksel birleştirme `group_key()`'den farklı bir kavram).
- `SnapshotRequest.only_patterns` (fnmatch, `routes/downloads.py`) — çok
  dosyalı repo indirmelerinde alt küme seçimi için genel amaçlı filtre;
  Mochi'nin aynı repo içindeki hem fp32 hem bf16 varyantından yalnızca
  bf16'yı (~65GB, 19 dosya) çekmek için kullanıldı (repo toplamı ~133GB).
- Canlı ölçüm: gerçek API üzerinden 5 MB/sn hedefine karşı ~4.8 MB/sn
  toplam indirme hızı ölçüldü.
- **CogVideoX-2b istisnası:** bu model ad-hoc yöntemle zaten ~%87 inmiş
  durumdaydı; ilerlemeyi (~12GB) çöpe atmamak için o tek model HF cache
  üzerinden eski yöntemle tamamlanıp `models_dir/video/`'a yerleştirildi,
  Wan/Hunyuan/Mochi ise baştan gerçek Downloads Manager kuyruğundan indirildi
  — bu karar kullanıcıya açıkça anlatıldı.

### Downloads Manager'da canlı testte bulunan 3 hata (`studio_models/downloader.py`)

Wan/Hunyuan/Mochi'nin çok-dosyalı indirmeleri sırasında dosyalar tekrar tekrar
"paused"a düşüyor, hatta bazen **zaten `done` olan dosyalar sıfırdan tekrar
iniyordu** gibi görünüyordu. Kök neden üç ayrı, gerçek hataydı:

1. **416-at-EOF resume hatası:** `.part` dosyası tam boyutuna ulaşmış ama
   `dest`'e taşınmadan (crash/cancel tam o arada) kesilirse, sonraki resume
   sunucudan `Range: bytes=<toplam>-` ister — "eklenecek bayt yok" demek olan
   `416 Range Not Satisfiable` döner, bu FAILED'a düşürülüyordu. Düzeltme:
   `Content-Range: bytes */<boyut>` ile gerçek boyut doğrulanıp eşleşiyorsa
   dosya tamamlanmış sayılıyor.
2. **Asıl suçlu — `resume()` çifte-başlatma yarışı:** `resume()`, aynı
   indirme için zaten çalışan bir `asyncio.Task` varken tekrar çağrılırsa
   engellemiyordu — iki coroutine aynı `.part` dosyasına birden yazıp onu
   bozuyordu. Canlı örnek: Wan'ın `transformer/...00001-of-00002.safetensors`
   parçası HF'de 4.998.781.576 bayt iken diskte 5.261.145.360 bayt (262MB
   fazla) olarak bulundu — DB'de `sha256_actual` alanı da (muhtemelen
   önceki, bozulmamış denemeden kalma) doğru hash'i gösteriyordu, bu da
   teşhisi bir süre karıştırdı. Düzeltme: `resume()` artık aynı ID için canlı
   task varsa reddediyor.
3. **"dest var, .part yok" durumunda boyut doğrulamadan DONE'a çevirme
   riski:** rename başarılı olup DB güncellemesi kesintiye uğradığında
   dosyayı ağdan tekrar çekmeden tamamlanmış saymak için eklenen hızlı yol,
   bilinen `total_bytes` ile eşleşmiyorsa dosyayı bozuk kabul edip siliyor ve
   normal indirmeye düşüyor (yukarıdaki #2'nin bıraktığı bozuk dosyaları
   sessizce "tamam" işaretlemesin diye).

Ayrıca tek seferlik bir onarım script'i (`reconcile_stale_downloads.py`)
yazılıp `paused`/`failed` durumundaki, `.part`'ı olmayan ama `dest`'i doğru
boyutta duran kayıtlar toplu olarak DONE'a çevrildi.

### Canlı GPU smoke testleri (RTX 5090)

- **CogVideoX-2b** — 480×320, 17 kare, 10 adım, cfg 6.0: **56.2 sn** toplam
  (model yükleme ~22sn + checkpoint ~16sn + 10 adım ~2.2sn + encode).
  ffprobe ile doğrulandı: h264, 480×320, 8fps, 17 kare. Ölçülen hız
  (`_STEPS_PER_SEC["cogvideox"]`): 4.5 adım/sn.
- **Wan2.1-T2V-1.3B** — 480×256, 17 kare, 10 adım, cfg 5.0: **66.8 sn**
  toplam. ffprobe ile doğrulandı: h264, 480×256, 8fps, 17 kare. Ölçülen hız
  (`_STEPS_PER_SEC["wan"]`): 5.0 adım/sn. **Canlı testte bulunan ayrı bir
  hata:** diffusers'ın `WanPipeline`'ı (`pipeline_wan.py::basic_clean()`)
  `ftfy.fix_text()`'i, `ftfy` kurulu olmasa bile modül seviyesinde
  `if is_ftfy_available(): import ftfy` korumasından SONRA koşulsuz çağırıyor
  — `ftfy` kurulu değilse `NameError: name 'ftfy' is not defined` ile
  patlıyor (bizim kodumuzda değil, diffusers'ın kendi eksik bağımlılık
  koruması). Düzeltme: `uv pip install --python envs/image/Scripts/python.exe
  ftfy` (hafif, saf-Python, torch/diffusers'a dokunmuyor).

### HunyuanVideo/Mochi — bu makinede RAM sınırı (bilinen, açık sınırlama)

**Sonuç: bu makinede canlı test EDİLEMEDİ.** Sebep model boyutu değil (RTX
5090'ın 32GB VRAM'i rahat yeter) — sistem RAM'i (**15.7 GB toplam**, tipik
olarak ~9-10GB boş) yetersiz:

| Aile | Diskte toplam boyut |
|---|---|
| CogVideoX-2b | 12.8 GB (çalışıyor) |
| Wan2.1-T2V-1.3B | ~10 GB (çalışıyor) |
| **HunyuanVideo** | **39.0 GB** |
| **Mochi-1-preview** | **37.3 GB** |

İlk deneme (`vram_mode="low"` → `enable_sequential_cpu_offload()`) modeli
ÖNCE tamamen CPU RAM'e yüklüyor, SONRA GPU'ya parça parça veriyor — 39GB'lık
bir model 15.7GB RAM'e sığmayınca canlı ölçümle doğrulanan bir OOM ile
segfault (exit 139) veriyordu: boş RAM 10GB'tan 108MB'a düşüp süreç anında
öldü, bellek hemen geri döndü.

**Denenen düzeltme (`_load_pipeline_split()`, `adapters/video.py`):**
`vram_mode="low"` artık pipeline'ı bütün olarak yüklemek yerine her ağır
alt-modeli (`transformer`/`text_encoder`/`vae`) `model_index.json`'dan okuyup
accelerate'in `device_map`+`max_memory` mekanizmasıyla TEK TEK, doğrudan
hedef cihaza dağıtıyor — önce GPU'yu doldur, taşanı CPU RAM'e ver, model asla
CPU'da tam materialize olmuyor. Yolda iki alt-hata daha bulundu:

- `device_map="auto"` hem `transformers` hem `diffusers`'ın
  `get_balanced_memory()` adımını tetikliyor, bu fonksiyon `max_memory`'deki
  `"disk"` anahtarını `torch.device(...)` ile ayrıştırmaya çalışıp
  `RuntimeError` atıyor. `device_map="sequential"` bu adımı tamamen atlıyor
  (ayrıca GPU'yu önce doldurup taşanı sırayla CPU/disk'e vermesi zaten
  istediğimiz davranış).
- Bu düzeltmeyle text_encoder (Llama, ~15GB) doğru şekilde GPU'ya dağıtılıp
  yükleniyor (nvidia-smi'de 21GB kullanım doğrulandı) — önceki gibi anında
  rastgele bir noktada çökmüyor.

**Ama tam çözülmedi:** `max_memory`'ye `"disk": "200GiB"` eklenmesine ve
`offload_folder` verilmesine rağmen transformer'ın checkpoint parçaları
yüklenirken (6 parçanın ~3'ünde) yine aynı RAM-taşması segfault'u oluyor —
disk katmanı max_memory'de doğru şekilde tanımlı olsa da, ilk shard okuma
adımı (accelerate'in `from_pretrained` iç akışı) parçayı diske fiilen
dispatch etmeden önce CPU RAM'de tutuyor gibi görünüyor. GPU(~32GB)+boş
RAM(~10GB) ≈ 35GB bütçe, Hunyuan'ın ~39GB'ı için ~4GB kısa kalıyor.

**Kullanıcı kararı (2026-07-20):** RAM yükseltmesi ya da text_encoder
nicemleme (8-bit, ~15GB→~7.5GB) gibi ciddi bir yatırım yapılana kadar
HunyuanVideo/Mochi bu makinede canlı kullanılamıyor olarak bırakıldı — Mochi
(37.3GB) hiç denenmedi çünkü aynı RAM açığını vereceği neredeyse kesin. Kod
(`_load_pipeline_split`) kalıcı olarak eklendi — daha küçük modellerde veya
gelecekte RAM artınca faydalı, geri alınmadı.

## Faz 9 — 3d env (envs/3d, AYRI venv — TripoSR canlı doğrulandı)

```bash
uv venv ~/LocalAIStudio/envs/3d \
  --python "C:\Users\Tars\AppData\Local\Programs\Python\Python310\python.exe" \
  --system-site-packages
uv pip install --python ~/LocalAIStudio/envs/3d/Scripts/python.exe \
  "omegaconf==2.3.0" "einops==0.7.0" rembg "imageio[ffmpeg]" "xatlas==0.0.9" \
  "transformers==4.35.0" "huggingface-hub>=0.25,<1.0" "trimesh>=4.12" \
  scikit-image
```

**TripoSR'ın orijinal `requirements.txt`'indeki iki pin canlı testte
sorunlu çıktı, üstteki komutta düzeltilmiş hâliyle var:**
- `huggingface-hub` (pinsiz bırakılınca transformers==4.35.0'ın izin
  verdiği EN ESKİ sürüme, 0.17.3'e düştü) — ama `--system-site-packages`
  ile miras alınan `accelerate` (0.34.2, çok daha yeni) `huggingface_hub.
  split_torch_state_dict_into_shards`i istiyor, 0.17.3'te yok →
  `ImportError`. Çözüm: `huggingface-hub>=0.25` — transformers 4.35.0 ile de
  sorunsuz çalışıyor (o, hub sürümüne karşı gevşek bağlı).
- `trimesh==4.0.5` (TripoSR'ın pini) GLB dışa aktarımında `ndarray.ptp()`
  kullanıyor — NumPy 2.0'da kaldırıldı, `numpy==2.2.6` ile `AttributeError`.
  Çözüm: NumPy'ı düşürmek yerine `trimesh>=4.12`'ye yükseltmek (aktif
  bakımlı, NumPy 2.x uyumlu).

**Neden ayrı venv (image ile paylaşılmadı):** TripoSR (image-to-3D kademeli
listenin ilk/en hafif modeli) `transformers==4.35.0`'a sabitlenmiş —
envs/image'ın 4.46.3'üyle çakışır. `--system-site-packages` sayesinde her
venv kendi shadow'unu bağımsız tutuyor (bkz. dosyanın başındaki ilke), yani
envs/3d'ye 4.35.0 kurmak envs/image'ı ETKİLEMEDİ — canlı doğrulandı
(`torch.__version__` her iki ortamda da ayrı ayrı doğru geldi).

**TripoSR pip ile kurulamıyor** (resmi README: repo klonlanıp `run.py`
çalıştırılıyor) — `tsr` paketi bir sonraki adımda vendor edilecek
(muhtemelen `PYTHONPATH` ile, studio paketlerinin kendi kaynak-import
deseniyle aynı).

**Bulunan ve çözülen gerçek engel — `torchmcubes` CUDA derlemesi:**
TripoSR'ın `requirements.txt`'i `git+https://github.com/tatsy/torchmcubes.git`
istiyor (mesh çıkarımı için marching-cubes, GPU hızlandırmalı). Bu
makinede derleme **kesin olarak imkansız** — sebep GPU değil, sürücü değil,
`torch` bile değil: `torch 2.11.0+cu128` CUDA DESTEĞİYLE derlenmiş olduğu
için, ona bağlanan HERHANGİ bir yeni C++/CUDA eklentisinin CMake
yapılandırması (`Caffe2Config.cmake`) sistemde **NVIDIA CUDA Toolkit'in**
(nvcc + geliştirme kütüphaneleri) kurulu olmasını zorunlu kılıyor —
bu makinede yalnızca GPU sürücüsü var, Toolkit yok (`nvcc` PATH'te
bulunamadı; llama-cpp-python notundaki "CUDA toolkit kurulmadı" tespitiyle
tutarlı). Adım adım doğrulandı: build-isolation kapatıldı, `scikit-build-
core`/`cmake`/`ninja`/`pybind11` elle kuruldu, `CMAKE_PREFIX_PATH` torch'un
cmake dizinine ayarlandı — hepsi sırayla ilerledi, son adımda kesin olarak
`Caffe2Config.cmake`'in "CUDA_TOOLKIT_ROOT_DIR not found" hatasında durdu.

CUDA Toolkit'in tam kurulumu büyük, sistem geneli bir değişiklik (sadece bir
venv'e değil) — kullanıcı onayı olmadan yapılmadı. **Uygulanabilir alternatif
canlı doğrulandı ve üretime alındı:** `scikit-image`'in saf-Python
`measure.marching_cubes()`'i — mesh çıkarımı GPU'da değil CPU'da, TripoSR'ın
asıl sinir ağı çıkarımı (VRAM'in çoğunu kullanan kısım) yine RTX 5090'da
koşuyor; canlı ölçümde marching cubes darboğaz OLMADI (256³ çözünürlükte
~2sn, toplam 9.8sn'lik sürenin küçük bir kısmı).

**Kasıtlı olarak kurulmadı:** `gradio`, `moderngl` — TripoSR'ın kendi demo
UI'sine/önizleme render'ına ait, bu projede zaten React tabanlı bir 3D
viewer planlandığı için (plan.md Faz 9 adım 4) gerekmiyor. `bake_texture.py`
(moderngl gerektiren doku-pişirme özelliği) bu yüzden kullanılmadı — ilk
sürüm vertex-color tabanlı mesh üretiyor (GLB bunu destekliyor).

**Vendoring:** TripoSR'ın `tsr` paketi (MIT lisans) `backend/studio_inference/
studio_inference/vendor/triposr/tsr/`'a kopyalandı (yalnızca çıkarım kodu —
`run.py`/`gradio_app.py` demo scriptleri değil); `tsr/models/isosurface.py`
yukarıdaki torchmcubes→scikit-image yamasını içeriyor. Provenance/yama notu:
`vendor/triposr/NOTICE.md`. `adapters/mesh.py` (`MeshAdapter`,
`EngineAdapter` sözleşmesi) bu paketi `sys.path`'e ekleyip kullanıyor —
studio paketlerinin kendi "PYTHONPATH ile kaynaktan import" deseniyle aynı.
Scheduler'da `mesh` iş türü `_ENV_ALIAS` ile `envs/3d`'ye yönlendiriliyor
(fiziksel klasör adı iş türü adından farklı, video/imgtool'un `image`'ı
paylaşmasıyla aynı desen).

**Canlı doğrulama (RTX 5090, `stabilityai/TripoSR`, `chair.png` örnek
görseli):** arkaplan temizliği (rembg) → sinir ağı çıkarımı → CPU marching
cubes (mc_resolution=256) → GLB dışa aktarım, uçtan uca **9.8 sn**.
Üretilen mesh: 42.076 vertex, 84.156 yüz, `trimesh` ile geri yüklenip
doğrulandı: **watertight=True**, pozitif hacim/alan, makul sınır kutusu —
scikit-image yamasının doğru geometri ürettiğinin kanıtı (bozuk bir
marching-cubes eşlemesi neredeyse kesin non-watertight/self-intersecting
bir mesh üretirdi).

**Model Manager/Downloads Manager entegrasyonu:** İlk canlı testte TripoSR
ağırlıkları `TSR.from_pretrained()`'in kendi `hf_hub_download` çağrısıyla
`~/.cache/huggingface/hub`'a inmişti — projenin gerçek indirme kuyruğundan
geçmemişti. Düzeltildi: `POST /v1/downloads/snapshot` ile (`repo_id=
"stabilityai/TripoSR"`, `subdir="3d"`, `only_patterns=["config.yaml",
"model.ckpt"]`) `models/3d/TripoSR/` altına indirildi — Wan/Hunyuan/Mochi ile
aynı, önceki `HTTPAdapter` denemesi yerine kalıcı uygulama-içi indirme yolu.

Bunun ortaya çıkardığı ayrı bir eksik: `metadata.py::group_key()`, çok
dosyalı/tek-dosyalı bir modeli tek `ModelRecord`'a toplarken yalnızca
`config.json`'lı klasörleri tanıyordu (LTX/Wan gibi diffusers pipeline'ları
için yeterliydi) — TripoSR `config.yaml` kullandığı için `model.ckpt` kendi
başına, "TripoSR" değil "model" adıyla bir kayıt oluyordu. `config.yaml`
desteği eklendi (muhtemelen kademeli listenin diğer 3D modelleri de YAML
config kullanacak); canlı doğrulandı — `GET /v1/models?kind=3d` artık doğru
`{"name": "TripoSR", "kind": "3d", ...}` döndürüyor. (Not: scanner.py yalnızca
dosya BOYUTU değiştiğinde metadata'yı yeniden çıkarıyor — bu kod değişikliği
sonrası var olan kaydın adının güncellenmesi için tek seferlik bir DB
silme+rescan gerekti, gelecekteki yeni indirmelerde otomatik doğru gelecek.)

`routes/jobs.py::_create_job_row()`'a `mesh` iş türü için otomatik
`output_path` üretimi eklendi (`outputs/<uuid>.glb`) — image/audio/video ile
aynı desen. Model-Manager-kayıtlı yoldan (`models/3d/TripoSR/model.ckpt`,
`MeshAdapter.load()`'daki `os.path.isfile→dirname` çözümlemesiyle) ikinci bir
görselle (`hamburger.png`) uçtan uca tekrar canlı doğrulandı: 9.6sn, 55.599
vertex/111.172 yüz.

## Faz 9 adım 2-3 — meshtool (mesh işleme + export, envs/3d paylaşılıyor)

`adapters/meshtool.py` (`MeshToolAdapter`) — imgtool.py/videotool.py ile aynı
op-dispatch deseni, model gerektirmez (saf trimesh/CPU). Op'lar: `convert`
(GLB/GLTF/OBJ/STL/PLY arası, uzantıya göre), `decimate` (`fast-simplification`
paketiyle quadric decimation — `envs/3d`'ye eklendi), `repair` (fix_normals/
fix_winding/fill_holes), `info` (dosya yazmadan vertices/faces/watertight/
bounds/volume/area — bir STL'i slicer'a göndermeden sağlık kontrolü). `mesh`
gibi `_ENV_ALIAS` ile `envs/3d`'ye yönlendiriliyor; `routes/jobs.py`'de
`info` hariç tüm op'lar için otomatik `output_path` (.glb) üretimi var.

**Bulgu — decimate sonrası watertight garantisi YOK:** Quadric decimation
(84.156→16.830 yüz, chair.png meshi) sonrası mesh watertight olmaktan çıktı;
otomatik onarım denemesi (trimesh `fix_winding`+`fix_normals`+`fill_holes`)
BAŞARISIZ kaldı (euler_number=1, 6 "broken face" — muhtemelen decimation'ın
ürettiği gerçek bir non-manifold kusur). Daha güçlü bir alternatif olarak
`pymeshlab` canlı denendi (`meshing_remove_duplicate_faces/vertices`,
`meshing_repair_non_manifold_edges`, `meshing_close_holes` filtreleri) — O DA
bu spesifik kusuru çözemedi. Sonuç: `decimate` op'u best-effort onarım
uyguluyor (çoğu durumda yeterli) ama GARANTİ etmiyor; `pymeshlab` işe
yaramadığı ve kullanılmadığı için `envs/3d`'den kaldırıldı (kurulu
bağımlılık şişmesin diye) — sonuçtaki `watertight` alanı her zaman gerçek
durumu yansıtıyor, UI/çağıran taraf buna göre karar vermeli (ör. gerekirse
`repair` op'unu ayrıca deneyebilir ya da kullanıcıyı uyarabilir). Bu,
mesh basitleştirmenin genel olarak zor bir problemi — profesyonel araçlar
(Netfabb/Meshmixer) bu yüzden var; kapsam dışı bırakıldı.

**Canlı doğrulama:** `mesh_smoke.glb` (chair.png'den) üzerinde tüm 4 op
test edildi — `convert`→STL (4.2MB)/OBJ (4.5MB)/PLY (1.8MB, hepsi doğru
boyutlarda), `decimate` (yukarıdaki not), `repair` (zaten watertight bir
mesh'te no-op gibi davrandı, beklenen), `info` (doğru vertices/faces/
watertight/bounds/volume/area döndü).

## Faz 9 — FBX export (headless Blender köprüsü, plan.md DoD maddesi)

trimesh Autodesk'in kapalı FBX formatını yazamıyor; pyassimp'in FBX
export'u güvenilmez (canlı test edilmedi — dokümantasyon/topluluk
raporları tutarlı şekilde eksik/bozuk export'lardan bahsediyor). Bu
makinede **Blender 5.0.1 zaten kurulu** (`C:\Program Files\Blender
Foundation\Blender 5.0\blender.exe`) — kurulum gerekmeden doğrudan
kullanıldı.

**Mimari:** `adapters/_blender_scripts/mesh_to_fbx.py` — Blender'ın KENDİ
Python'unda (`bpy`, worker env'inin torch/trimesh'i YOK) `blender
--background --python mesh_to_fbx.py -- --input X --output Y.fbx` olarak
çalışır; glb/gltf/obj/ply/stl içe aktarıp FBX dışa aktarıyor.
`adapters/_blender_bridge.py::export_fbx()` bu subprocess'i worker env'inden
çağırıyor — `find_blender_exe()` sırayla `STUDIO_BLENDER_PATH` ortam
değişkenini, PATH'i, sonra `Program Files\Blender Foundation\Blender
*\blender.exe` glob'unu dener (Ollama köprüsüyle aynı "opsiyonel dış
kurulum" deseni — Blender bulunamazsa RIFE gibi sessizce atlanmıyor,
anlaşılır bir hata mesajı veriyor). `meshtool.py::_export()`'a entegre
edildi: `output_path` `.fbx` ile bitiyorsa mesh önce geçici bir GLB'ye
export edilip köprüden geçiyor — `convert`/`decimate`/`repair` op'larının
HEPSİ doğrudan `.fbx` çıktısı üretebiliyor.

**Canlı doğrulama (round-trip):** `mesh_smoke.glb` (42.076 vertex/84.156
yüz) → FBX (2.58MB) → Blender'a GERİ import edildi: 42.076 vertex/84.156
poly, birebir eşleşti. Bu aynı zamanda plan.md'nin DoD'sini ("FBX Blender'da
import edilir") otomatik sağlıyor — Blender kendi ürettiği FBX'i sorunsuz
okuyabiliyor.

## Faz 9 — Stable Fast 3D (SF3D) ikinci aile olarak eklendi (dokulu çıktı)

`adapters/mesh.py` TripoSR-özel tek-aile yapısından çok-aileli bir yapıya
(video.py'nin `_FAMILIES` deseniyle aynı ruhta) refactor edildi —
`family: "triposr" | "sf3d"`.

**Bağımlılıklar (envs/3d'ye eklendi):** `jaxtyping==0.2.31`,
`open_clip_torch==2.24.0`, `pynanoinstantmeshes==0.0.3`, `gpytoolbox==0.2.0`,
`transformers==4.42.3` (TripoSR'ın 4.35.0'ından yükseltildi — canlı testte
TripoSR'ın hâlâ sorunsuz çalıştığı doğrulandı, aynı `envs/3d` paylaşılıyor).

**KRİTİK KAZA (ve düzeltmesi) — torch'u kazayla ezmek:** `open_clip_torch`/
`timm` bağımlılık çözümü, pinlenmemiş bir `torch` isteyince uv PyPI'dan
**CPU-only torch 2.13.0** indirip `envs/3d`'nin kendi site-packages'ına
kurdu — bu, `--system-site-packages` ile miras alınması gereken korumalı
sistem torch'unu (2.11.0+cu128, CUDA'lı) GÖLGELEDİ. `torch.cuda.is_available()
→ False` ile fark edildi, `uv pip uninstall torch torchvision` ile düzeltildi
(sistem torch'u zaten hiç değişmemişti, yalnızca envs/3d'nin ÜSTÜNE binen
kopya kaldırıldı) — TripoSR canlı smoke testiyle CUDA'nın geri geldiği
doğrulandı. **Ders:** paylaşılan `--system-site-packages` env'lere yeni
paket eklerken `torch`/`torchvision` gibi büyük paketlerin transitive
dependency olarak sessizce yeniden kurulmadığından HER SEFERİNDE emin
olunmalı (`torch.cuda.is_available()` kontrolü kurulumdan hemen sonra).

**İki C++ eklentisi (`texture_baker`, `uv_unwrapper`) CPU olarak derlendi**
(`--no-build-isolation --no-deps`, `CMAKE_PREFIX_PATH` torch'un cmake
dizinine ayarlanarak) — torchmcubes'un aksine bunlar CUDA_HOME yoksa
`CUDAExtension` yerine `CppExtension`'a **zarifçe düşüyor** (setup.py'da
`use_cuda = use_cuda and CUDA_HOME is not None`), yani derlemesi hiç
sorun çıkarmadı.

**SF3D marching cubes/`torchmcubes` KULLANMIYOR** — kendi marching
TETRAHEDRA uygulaması var (`sf3d/models/isosurface.py`, saf PyTorch,
önceden hesaplanmış `load/tets/160_tets.npz` ızgarası) — TripoSR'daki gibi
bir yama gerekmedi.

**Canlı testte bulunan iki gerçek hata:**
1. SF3D `prepare_image()` RGBA bekliyor (alfa kanalı = maske) — TripoSR'ın
   gri-arkaplana compose edilmiş düz RGB deseni SF3D'ye uygulanınca
   `ValueError: Image must be in RGBA mode` verdi. Düzeltme: family'ye göre
   dallandı — sf3d RGBA'da kalıyor, triposr gri-compose ediliyor.
2. `texture_baker` CPU çekirdeği + SF3D'nin GPU'daki tensörleri ->
   `NotImplementedError: Could not run 'texture_baker_cpp::rasterize' ...
   CUDA backend`. Düzeltme: `TextureBaker.rasterize`/`interpolate`'i
   çalışma zamanında monkeypatch'leyip (vendored `sf3d/system.py`'yi
   DEĞİŞTİRMEDEN) girdileri CPU'ya taşıyıp sonucu orijinal cihaza geri
   döndürüyoruz — sinir ağı çıkarımı yine GPU'da, yalnızca rasterizasyon
   CPU'da.

**LİSANS UYARISI:** Kod + ağırlıklar Stability AI Community License —
XTTS-v2 ile aynı desende UI'da gösteriliyor (`MeshPanel.tsx`, family=sf3d
seçiliyken).

**Canlı doğrulama (gerçek Downloads/Model Manager yoluyla indirilen ağırlık,
gerçek `/v1/jobs` API'si üzerinden — yalnızca standalone script değil):**
portre fotoğrafından **15.6 sn'de** 17.152 vertex/30.700 yüz, gerçek
`PBRMaterial` + 1024×1024 baked `baseColorTexture` ile GLB üretildi
(`trimesh` ile doğrulandı: `visual.kind == "texture"`).

**MeshPanel.tsx güncellendi:** aile seçici (TripoSR/SF3D/OpenLRM), aileye
özgü alanlar (triposr: mc_resolution; sf3d: texture_resolution/remesh/
target_vertex_count; openlrm: mesh_size/mesh_thres), model listesi seçili
aileye göre filtreleniyor, SF3D + OpenLRM lisans notları.

## Faz 9 — nvdiffrast engeli: InstantMesh/TRELLIS/CRM ORTAK sorunu

Kademeli listenin üç modeli (InstantMesh, TRELLIS, CRM) araştırıldı — üçü
de mesh çıkarım/render adımında `nvdiffrast`'ın `dr.RasterizeCudaContext()`'ini
kod içinde SABİT kullanıyor (yapılandırılabilir değil, birden fazla dosyada
hardcoded). `nvdiffrast` texture_baker/torchmcubes'un aksine ÇALIŞMA
ZAMANINDA JIT CUDA derlemesi yapıyor (`torch.utils.cpp_extension.load`) —
bu, `CUDA_HOME`'un o an sistemde bulunmasını gerektiriyor, yalnızca pip
install anında değil HER model yüklemesinde. `RasterizeGLContext` bile
(OpenGL tabanlı rasterizasyon) `RasterizeCudaContext`'in ALT SINIFI —
interpolasyon/antialiasing/doku örnekleme operatörleri hâlâ CUDA çekirdekleri,
yalnızca üçgen rasterizasyon adımı OpenGL'e düşüyor — yani GL modu da CUDA
Toolkit'ten kaçamıyor. Ayrıca TRELLIS `spconv`/`kaolin`/`flash-attn` gibi
başka CUDA-Toolkit-gerektiren ağır bağımlılıklara da sahip (nvdiffrast'tan
bağımsız olarak).

**Sonuç:** Bu üç model, NVIDIA CUDA Toolkit kurulmadan bu makinede ÇALIŞMAZ
— torchmcubes/texture_baker'daki gibi bir CPU-fallback yamasıyla
atlatılamıyor (kod düzeyinde değil, çalışma zamanı JIT derleme düzeyinde bir
kısıt). CUDA Toolkit'in tam kurulumu büyük, sistem geneli bir değişiklik —
kullanıcı onayı olmadan yapılmadı, `adapters/mesh.py`'nin modül
docstring'inde bu üçünün neden eklenmediği açıkça not edildi.

## Faz 9 — OpenLRM üçüncü aile olarak eklendi

`envs/3d`'ye eklenen bağımlılıklar: `lpips`, `megfile`, `PyMCubes`
(TripoSR/SF3D'nin aksine OpenLRM `mcubes.marching_cubes` — saf CPU marching
cubes — kullanıyor, `nvdiffrast` YOK, render tamamen kendi EG3D-tarzı saf
PyTorch `MipRayMarcher2` hacim render'ıyla yapılıyor).

**AYNI torch-gölgeleme kazası TEKRAR oldu:** `lpips`/`megfile` kurulumu yine
pinlenmemiş bir `torch` isteyip `envs/3d`'nin korumalı sistem torch'unu
(2.11.0+cu128) CPU-only bir kopyayla gölgeledi — aynı `uv pip uninstall
torch torchvision` ile düzeltildi. **Ders pekişti:** bu paylaşılan env'e
YENİ bir paket eklenen HER SEFERİNDE `torch.cuda.is_available()` hemen
kontrol edilmeli, varsayım yapılmamalı.

**Mimari zorluk:** `LRMInferrer` (openlrm/runners/infer/lrm.py) kendi
kurucusunda `argparse`/ortam değişkenlerinden (`APP_MODEL_NAME`, `APP_INFER`)
okuyan bir CLI/Hydra-tarzı yapılandırma kullanıyor — basit bir
`Class.from_pretrained()` deseni değil. Kendi `app.py`'sinin (gradio demo)
yaptığı gibi taklit edildi: `os.environ` ile `APP_ENABLED`/`APP_MODEL_NAME`/
`APP_INFER` (repo adında "small"/"large" geçmiyorsa `infer-b.yaml`, ağırlığı
indirilen tek model `openlrm-mix-base-1.1` olduğu için) ayarlanıp
`LRMInferrer()` örnekleniyor. `infer_single()` bir dosya yolundan okuyor
(PIL Image kabul etmiyor) ve mesh'i doğrudan döndürMÜYOR (yalnızca dosyaya
export ediyor) — geçici bir PNG'ye yazılıp sonuç `trimesh` ile geri
okunuyor.

**Kod düzenlemesi:** `remove_background`/`resize_foreground` artık
`mesh.py`'de aileden bağımsız, PAYLAŞILAN bir kopya (önceden TripoSR/SF3D
kendi vendor modüllerinden import ediyordu — openlrm yüklüyken sf3d'nin
vendor dizini sys.path'te olmayabileceği için çapraz-aile import'u kaldırıldı).

**LİSANS UYARISI:** kod Apache-2.0, ağırlıklar **CC BY-NC 4.0** (yalnızca
araştırma/kişisel, ticari kullanım YOK) — UI'da SF3D ile aynı desende
gösteriliyor.

**Canlı doğrulama (gerçek `/v1/jobs` API'si üzerinden):** bir portre
fotoğrafından (mesh_size=192) **7.3 sn'de** 115.857 vertex/231.688 yüz;
standalone smoke testte (chair.png, mesh_size=384, ilk kullanım — 330MB'lık
DINOv2 kodlayıcı indirmesi dahil) 40.6 sn'de 289.393 vertex/578.542 yüz —
TripoSR/SF3D'den belirgin biçimde daha yoğun mesh üretiyor (varsayılan UI
değeri bu yüzden 256'ya düşürüldü, kaydırıcıyla 384'e çıkarılabilir).

## Faz 9 — Hunyuan3D-2 dördüncü aile olarak eklendi (yalnızca shape, dokusuz)

Araştırma: Hunyuan3D-2'nin `requirements.txt`'inde `nvdiffrast` YOK (grep
boş döndü) — InstantMesh/TRELLIS/CRM'e göre umut vericiydi. Ama repo İKİ ayrı
alt paket içeriyor: `hy3dgen/shapegen` (image→dokusuz mesh, diffusion+VAE) ve
`hy3dgen/texgen` (doku/boyama — multi-view diffusion + UV'ye pişirme).
`hy3dgen/texgen/custom_rasterizer/setup.py` `custom_rasterizer_kernel`'i
`torch.utils.cpp_extension.CUDAExtension` ile derliyor (`rasterizer_gpu.cu`,
nvcc gerektiriyor) ve `render.py` içinde KOŞULSUZ `import
custom_rasterizer_kernel` var — texture_baker/uv_unwrapper'ın aksine CPU'ya
düşen bir yol yok. **Bu, nvdiffrast ile AYNI sınıf engel** (CUDA Toolkit
gerektiren, çalışma zamanı derleme). `differentiable_renderer/setup.py`
(`mesh_processor`, saf pybind11/C++, CUDA'sız) tek başına bir şey
çözmüyor — texgen pipeline'ı `custom_rasterizer` olmadan zaten çalışmıyor.

**Kritik bulgu — shapegen texgen'den TAMAMEN bağımsız:**
`grep -rl texgen hy3dgen/shapegen` boş döndü, `hy3dgen/__init__.py`'de de
texgen'i eager import eden bir şey yok. Yani shape üretimi bu makinede
CUDA Toolkit olmadan ÇALIŞIYOR — yalnızca texgen/paint aşaması engelli.
Bu, "Hunyuan3D-2 diffusers-uyumlu (`model_index.json` var)" şeklindeki
önceki notumuzdan farklı çıktı — gerçekte kendi omegaconf/yaml
tabanlı config + tek-checkpoint (`from_single_file`) yükleme deseni
kullanıyor, diffusers `model_index.json` deseni değil.

`envs/3d`'ye eklenen bağımlılıklar: `diffusers==0.39.0`, `pymeshlab`
(`hy3dgen/shapegen/__init__.py`, `postprocessors.py`'yi eager import ettiği
için pymeshlab HER ZAMAN gerekli — FaceReducer/FloaterRemover kullanılmasa
bile). Kurulumdan sonra **yine `torch.cuda.is_available()` kontrol edildi —
bu sefer gölgeleme OLMADI** (`diffusers`/`pymeshlab` pinsiz `torch` çekmedi).

**Model yükleme:** upstream'in `smart_load_model()`'ı kendi önbellek-yolu
kuralını (`~/.cache/hy3dgen/<repo>/<subfolder>`) kullanıyor — Model
Manager'ın gerçek indirme klasörüyle uyuşmuyor. `adapters/mesh.py` bunu
atlayıp `Hunyuan3DDiTFlowMatchingPipeline.from_single_file(ckpt_path,
config_path, ...)`'i doğrudan `hunyuan3d-dit-v2-0-fast/` klasörüne
işaret ederek çağırıyor (model_path boşsa upstream'in `from_pretrained`+
`smart_load_model` akışına, yani HF indirmesine düşülüyor). Ayrıca
**ayrıca indirilen `hunyuan3d-vae-v2-0` klasörü gereksiz** olduğu
keşfedildi: `hunyuan3d-dit-v2-0-fast/model.fp16.safetensors`'ın
safetensors anahtar önekleri incelendiğinde (`model.*`, `vae.*`,
`conditioner.*`) TEK checkpoint'in üçünü de birlikte barındırdığı görüldü
— `from_single_file` zaten hepsini aynı dosyadan okuyor.

**LİSANS UYARISI:** Tencent Hunyuan 3D 2.0 Community License — AB/UK/Güney
Kore hariç her yerde serbest; aylık 1M aktif kullanıcıyı aşan ürünler için
Tencent'ten ayrı lisans (`hunyuan3d@tencent.com`) gerekir. SF3D/OpenLRM'le
aynı desende UI'da gösteriliyor.

**Canlı doğrulama (gerçek `/v1/jobs` API'si üzerinden):** `num_inference_
steps=20, octree_resolution=256` ile **41 sn'de** (worker soğuk başlangıcı +
model yükleme dahil — model yükleme tek başına ~25sn) 232.496 vertex/464.984
yüz. VRAM tepe noktası standalone ölçümde (steps=30, octree_resolution=384 —
upstream varsayılanı) **5983MB**, 584.401 vertex/1.168.808 yüz — TripoSR/
SF3D/OpenLRM'den belirgin biçimde daha yoğun (UI varsayılanı bu yüzden
octree_resolution=256'ya düşürüldü).

Vendor: yalnızca `hy3dgen/shapegen/` (+ paket kökü `hy3dgen/__init__.py`)
`vendor/hunyuan3d/`'ye kopyalandı — `hy3dgen/texgen/` bilerek dışarıda
bırakıldı (yukarıdaki engel nedeniyle). Ayrıntı ve provenance:
`vendor/hunyuan3d/NOTICE.md`.

**Kademeli listenin geri kalanı (InstantMesh/TRELLIS/CRM), bu yazıldığı an
itibarıyla hâlâ engelli — AMA CUDA Toolkit artık kurulu (bkz. aşağıdaki
bölüm), engel şu an "CUDA Toolkit yok" değil, sadece "adapter henüz
yazılmadı".** Hunyuan3D-2'nin shapegen/texgen ayrımı bu üçünde YOK — üçü de
rasterizasyonu mesh çıkarımının kendisinde (yalnızca ayrı bir doku
aşamasında değil) kullanıyor, dolayısıyla "yalnızca shape kısmını al" gibi
bir bölünme bunlar için mümkün değildi zaten — ama artık gerekmiyor da.

## Faz 9 — CUDA Toolkit kuruldu: InstantMesh/TRELLIS/CRM engeli KALKTI

Kullanıcı "evet kur" onayı verdi (2026-07-20). NVIDIA CUDA Toolkit 12.8.0
Windows local installer'ı (`developer.download.nvidia.com/compute/cuda/
12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe`, HEAD isteğiyle
doğrulanan resmi NVIDIA CDN, 3.15GB) indirildi. Sürücü 576.88 zaten CUDA
12.9'u destekliyordu (`nvidia-smi`), torch da cu128 ile derlenmişti — bu
yüzden 12.8 toolkit sürümü seçildi.

**Kurulum, elle (kullanıcı tarafından) yapıldı — UAC gerektiriyor:**
installer'ın HERHANGİ bir çağrısı (hatta `-s -h` help metni bile) admin
yükseltmesi istiyor, ben doğrudan tetikleyemedim. İlk `Start-Process -Verb
RunAs` denemesi kullanıcı tarafından iptal edildi ("İşlem kullanıcı
tarafından iptal edildi") — bunun üzerine kullanıcı installer'ı kendi
yükseltilmiş PowerShell'inde elle çalıştırdı. Kurulum komutu, `Display.
Driver` bileşenini BİLEREK dışarıda bıraktı (sürücü downgrade riskini
önlemek için) — yalnızca derleyici + geliştirme kütüphaneleri:
```
cuda_12.8.0_571.96_windows.exe -s nvcc_12.8 cudart_12.8 cuobjdump_12.8 nvprune_12.8 nvdisasm_12.8 nvprof_12.8 cublas_12.8 cublas_dev_12.8 cufft_12.8 cufft_dev_12.8 curand_12.8 curand_dev_12.8 cusolver_12.8 cusolver_dev_12.8 cusparse_12.8 cusparse_dev_12.8 npp_12.8 npp_dev_12.8 nvjpeg_12.8 nvjpeg_dev_12.8 nvrtc_12.8 nvrtc_dev_12.8 nvml_dev_12.8 nvtx_12.8 visual_studio_integration_12.8 occupancy_calculator_12.8 thrust_12.8 cccl_12.8
```
Bileşen adları installer'ın kendi `-h`'ından DOĞRULANAMADI (yükseltme
duvarına takıldı) — NVIDIA'nın CUDA 12.x Windows sessiz kurulum için
belgelediği standart meta-paket adları kullanıldı. **Doğrulama:** `nvcc.exe
--version` → 12.8.61 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\
v12.8\bin\nvcc.exe`), `nvidia-smi --query-gpu=driver_version` → **576.88,
DEĞİŞMEDİ**. `CUDA_PATH`/`CUDA_PATH_V12_8` sistem env değişkenleri ve
PATH'e `...\v12.8\bin` doğru şekilde eklenmiş (yeni açılan process'ler
görüyor — kurulumdan önce açık olan shell'ler görmüyor, beklenen davranış).

**nvdiffrast canlı doğrulandı:** `pip install <yerel_klonlanmış_repo>`
(PyPI'de nvdiffrast paketi YOK, GitHub'dan kaynak kurulumu gerekiyor) 26
saniyede derlendi (`_nvdiffrast_c.cp310-win_amd64.pyd`, site-packages
KÖKÜNE, `nvdiffrast/` alt paketine değil — normal davranış). Gerçek bir
üçgeni `RasterizeCudaContext()` + `rasterize()` ile 64x64 çözünürlükte
rasterize etti (1352 sıfır-olmayan piksel) — bu, önceden "ÇALIŞAMAZ" olarak
belgelenen tam senaryo.

**`custom_rasterizer_kernel` (Hunyuan3D-2 texgen) — gerçek bir MSVC 14.44 +
CUDA 12.8 uyumsuzluğu bulundu ve düzeltildi:** İlk derleme denemesi AYNI
`error C2872: 'std': belirsiz simge` hatasıyla `torch/csrc/dynamo/
compiled_autograd.h` içinde patladı. Minimal bir tekrar-üretim testiyle
(tek `.cu` dosyası, yalnızca `#include <torch/extension.h>` + boş kernel)
sorunun SİSTEMİK olduğu doğrulandı — custom_rasterizer'ın kendi koduyla
ilgisi yok, `torch/extension.h`'ı nvcc host-pass'ından derleyen HERHANGİ
bir `.cu` dosyası bu hatayı alır. `nvdiffrast`'ın neden etkilenmediği
incelendi: kendi `.cu` dosyaları torch header'larından tamamen bağımsız
(dar CUDA/ATen include'ları), `torch/extension.h`'a ihtiyaç duyan TEK
dosya (`PYBIND11_MODULE` makrosu için) düz bir `.cpp` — nvcc'den hiç
geçmiyor, doğrudan `cl.exe` ile derleniyor. **Yerel düzeltme** (yalnızca
vendor kopyasında, upstream'e gönderilmedi): `custom_rasterizer_kernel/
rasterizer.h`'deki `#include <torch/extension.h>` → `#include <ATen/
ATen.h>` + `#include <torch/types.h>` (dynamo'suz, hâlâ `torch::Tensor`
sağlıyor); `rasterizer.cpp`'ye (PYBIND11_MODULE'ü barındıran, `.cu`
DEĞİL) açıkça `#include <torch/extension.h>` geri eklendi. Ayrıca ayrı bir
sorun: derin bir yoldan (`scratchpad\...\custom_rasterizer\`, 283 karakter)
derleme `LINK : fatal error LNK1104: ... .lib dosyası açılamıyor` ile
patladı — Windows MAX_PATH (260) aşımı, `.obj` dosyaları yazılabildi ama
linker'ın `/IMPLIB:` çıktısı yazılamadı. Çözüm: kaynağı kısa bir yola
(`C:\cr_build\`) kopyalayıp oradan derlemek (derlenen `.pyd` zaten site-
packages'e gidiyor, build dizini kalıcı değil). Düzeltmeden sonra
`custom_rasterizer_kernel.rasterize_image()` canlı test edildi: 64x64'te
bir üçgen, 1250 sıfır-olmayan piksel.

**Bu düzeltme GENELLENEBİLİR** — InstantMesh/TRELLIS/CRM kendi CUDA
uzantılarını derlerken AYNI `torch/extension.h`+`.cu`+MSVC 14.44 hatasına
çarparlarsa, aynı iki reçete (narrow include + kısa build yolu) uygulanmalı.

**Hunyuan3D-2 texgen/paint — CUDA Toolkit engeli kalktı ama YENİ, farklı
sınıftan bir engelle karşılaşıldı (upstream checkpoint/kod uyumsuzluğu):**
`hy3dgen/texgen` tamamen vendor edildi (`vendor/hunyuan3d/hy3dgen/texgen`),
`custom_rasterizer_kernel` ve `mesh_processor` ikisi de derlendi ve
çalışıyor, delight (`Light_Shadow_Remover`, StableDiffusionInstructPix2Pix
tabanlı) alt-pipeline'ı sorunsuz yükleniyor. Ama multiview diffusion
alt-pipeline'ının UNet'i (`hunyuan3d-paint-v2-0-turbo/unet/modules.py`)
`strict=True` ile state_dict yüklerken 36 fazla anahtarla patlıyor
(`image_proj_model.*`, `*.attn2.to_{k,v}_ip.weight` — IP-Adapter
bileşenleri) çünkü `modules.py` bu bileşenleri state_dict yüklemeden ÖNCE
hiç oluşturmuyor. HuggingFace'teki GÜNCEL `main` dalı kopyasıyla
karşılaştırıldı (WebFetch ile) — birebir aynı, hâlâ düzeltilmemiş bir
upstream hatası, yerel/sürüm sorunu değil. IP-Adapter bağlantısını
Tencent'in belgelenmemiş mimarisinden tahmin ederek yeniden inşa etmek
(veya `strict=False` ile sessizce atlamak) yanlış/bozuk dokularla
sonuçlanabilir, hiçbir hata vermeden — bu yüzden denenmedi.
`adapters/mesh.py`'de `family="hunyuan3d"` hâlâ yalnızca shape üretiyor.
Ayrıntı: `vendor/hunyuan3d/NOTICE.md`.

**Bu arada çözülen, texgen'e özgü bir bağımlılık sorunu:** `diffusers`
(shapegen için kurulan `0.39.0`) `0.32.2`'ye düşürüldü — 0.39.0'ın
`diffusers.models.autoencoders` paketi eager olarak `AutoencoderRAE`'yi
import ediyor, o da `transformers.Dinov2WithRegistersConfig` istiyor (çok
yeni transformers, `transformers==5.14.1`'e zorlardı — TripoSR/SF3D/
OpenLRM'in DINO/CLIP kodlayıcıları için çok riskli bir sıçrama). `0.32.2`,
`envs/image`'de zaten `transformers==4.46.3` ile kanıtlanmış aynı çift —
yukarı doğru sürüm kovalamak yerine bilinen-iyi bir eşleşme yeniden
kullanıldı. `transformers` `4.42.3` → `4.46.3`'e (`peft>=0.17`'nin istediği
`EncoderDecoderCache` için) + `tokenizers` → `0.20.3`'e (4.46.3'ün kendi
alt sınırı) yükseltildi. Diffusers düşürüldükten sonra shapegen'in hâlâ
sorunsuz import edildiği yeniden doğrulandı (regresyon kontrolü). **Torch-
gölgeleme kazası bu süreçte İKİ KEZ daha oldu** (`accelerate`/`peft`
çözümlemesi pinsiz CPU-only torch çekti) — ikisinde de `uv pip uninstall
torch torchvision` ile düzeltildi, `2.11.0+cu128, True` yeniden doğrulandı.

**Ayrıca çözülen, texgen'e özgü bir model-dosyası sorunu (vendor kodu
DEĞİL, Model Manager indirmesi):** `hunyuan3d-paint-v2-0-turbo/unet/
modules.py` (indirilen model klasöründeki kopya) `diffusion_pytorch_
model.bin` (pickle) yüklemeyi bekliyordu ama projenin safetensors-tercih
politikası gereği yalnızca `.safetensors` indirilmişti. Yerel yama: önce
`.safetensors`'ı `safetensors.torch.load_file` ile dene, yoksa `.bin`'e
düş. Diffusers'ın `~/.cache/huggingface/modules/diffusers_modules/`
önbelleği de temizlendi (değişikliğin görülmesi için).

## Faz 9 — InstantMesh beşinci aile olarak eklendi (LRM+FlexiCubes)

CUDA Toolkit kurulumunun ilk gerçek faydası: InstantMesh (TencentARC),
`src/models/lrm_mesh.py`'de modül seviyesinde `nvdiffrast` import ediyor
ama `extract_mesh(use_texture_map=False)` (varsayılan, vertex-renkli yol)
ASLA `RasterizeCudaContext` oluşturmuyor — yalnızca `use_texture_map=True`
(gerçek UV dokusu, `xatlas_uvmap` üzerinden) oluşturuyor. Yani nvdiffrast
kurulu olmasa bile vertex-renkli üretim çalışırdı; ama artık her ikisi de
çalışıyor ve ikisi de canlı test edildi.

**Mimari — iki ayrı diffusers pipeline'ı + bir LRM reconstruction modeli:**
1. `sudo-ai/zero123plus-v1.2` (temel çoklu-görünüm difüzyon modeli, tam bir
   SD1.5-boyutlu UNet+VAE+CLIP+vision_encoder — ~4.3GB, ayrı bir HF repo)
2. `TencentARC/InstantMesh`'in kendi `diffusion_pytorch_model.bin`'i
   (zero123plus'ın UNet'i ÜZERİNE `strict=True` ile yüklenen "beyaz
   arkaplan" ince ayarı — Hunyuan3D-2'nin IP-Adapter sorununun aksine, bu
   basit bir düz state_dict yüklemesi, ek bileşen gerekmiyor)
3. `instant_mesh_base.ckpt` (LRM+FlexiCubes reconstruction modeli —
   `encoder_model_name: facebook/dino-vitb16` kullanıyor, İLK KULLANIMDA
   HF'den ayrıca indiriliyor, Model Manager'dan bağımsız — OpenLRM'in
   DINOv2 notuyla aynı desen)

**zero123plus-v1.2 gerçek Downloads Manager'la indirildi** (`POST /v1/
downloads/snapshot`, `subdir="3d"`) — otomatik olarak yalnızca
`.safetensors` varyantlarını seçti, elle pickle-karantina gerekmedi.
`custom_pipeline="zero123plus"` upstream'in bare-string deseni (diffusers'ın
HF community pipelines'tan AĞDAN çektiği) yerine `vendor/instantmesh/
zero123plus/` yerel klasörüne işaret edecek şekilde değiştirildi —
OpenLRM/Hunyuan3D-2 ile aynı "yerel kod, ağ çağrısı yok" kuralı.

**FlexiCubes tamamen vendor edilmiş, saf PyTorch** (`src/models/geometry/
rep_3d/flexicubes.py`) — ekstra bir `flexicubes` paketi ya da CUDA uzantısı
GEREKMİYOR, InstantMesh'in kendi yazarları tarafından paket içine gömülmüş.

**Model yükleme sorunsuz geçti** — Hunyuan3D-2'nin aksine hiçbir strict-
load/checkpoint uyumsuzluğu YAŞANMADI, ilk denemede çalıştı. Fark muhtemelen
InstantMesh'in checkpoint'lerinin kendi yükleme kodlarıyla (`run.py`/
`modules.py` DEĞİL, düz `torch.load`+`load_state_dict`) aynı anda
üretilmiş/test edilmiş olması.

**Canlı doğrulama (gerçek `/v1/jobs` API'si üzerinden):** `num_inference_
steps=30, view=6, export_texmap=false` ile **19sn'de** (worker soğuk
başlangıç + model yükleme dahil) 46.012 vertex/92.016 yüz. Standalone
testte `export_texmap=true` (nvdiffrast+xatlas UV pişirme) ile 42.3sn'de
59.526 vertex/95.804 yüz, gerçek `TextureVisuals` materyali doğrulandı.
**VRAM tepe noktası canlı ölçüldü: 15.590MB** (`num_inference_steps=50,
export_texmap=false`) — diğer dört aileden BELİRGİN daha ağır (zero123plus
tam bir SD1.5-boyutlu difüzyon modeli taşıdığı için); `vram_estimate()`
16.000MB'a güncellendi.

Vendor: `src/` (LRM+FlexiCubes model + geometri/renderer/encoder alt
modülleri) ve `zero123plus/` (`Zero123PlusPipeline`, `diffusers.
StableDiffusionPipeline` alt sınıfı) `vendor/instantmesh/`'e kopyalandı, kod
değişikliği yapılmadı. Lisans: Apache-2.0 (özel UI uyarısı gerekmiyor,
TripoSR'la aynı muamele). Ayrıntı: `vendor/instantmesh/NOTICE.md`.

## Faz 9 — TRELLIS altıncı aile olarak eklendi (sparse-voxel, gerçek doku)

En karmaşık kademeli liste üyesi — `setup.sh` kaolin/flash-attn/xformers/
diffoctreerast/mip-splatting gibi ağır CUDA bağımlılıkları listeliyordu,
ama gerçek kod taraması bunların çoğunun GEREKSİZ olduğunu ortaya çıkardı:

- **kaolin GEREKSİZ:** `trellis/` paketi hiç import etmiyor — yalnızca
  vendored FlexiCubes alt modülünün (`.gitmodules` üzerinden ayrıca
  fetch edilen bir git submodule, `MaxtirError/FlexiCubes`) bir şekil-
  doğrulama yardımcısı (`check_tensor`, yalnızca `assert` içinde
  kullanılıyor) için import ediliyordu — 8 satırlık yerel bir eşdeğerle
  değiştirildi.
- **flash_attn/xformers İKİSİ DE ÇALIŞMIYOR bu makinede** (test edilerek
  doğrulandı, varsayılmadı): flash_attn'in resmi Windows wheel'i yok;
  `pip install xformers` unpinned bir torch==2.13.0 çekti (yine torch-
  gölgeleme kazası, `uv pip uninstall` ile düzeltildi), derlenen sürüm
  gerçek bir attention çağrısında `requires device with capability <=
  (9, 0) but your GPU has capability (12, 0) (too new)` hatası verdi —
  RTX 5090/Blackwell henüz desteklenmiyor. TRELLIS'in kendi DENSE
  attention modülünde zaten var olan `'sdpa'` seçeneğini SPARSE (voxel)
  attention modülüne PORTLADIK — paketlenmiş değişken-uzunluklu diziyi
  max uzunluğa sıfır-doldurup bool mask'le `torch.nn.functional.
  scaled_dot_product_attention` çağırmak, sonra tekrar paketlemek
  (xformers/flash_attn'in zaten yaptığı işin elle yapılmış hâli — aynı
  matematik, farklı çekirdek). `full_attn.py`'de (flow/diffusion
  modelleri, varsayılan `attn_mode="full"`, indirilen checkpoint
  config'lerinde teyit edildi) VE `windowed_attn.py`'de (mesh decoder'ın
  GERÇEKTEN kullandığı — "hiç çağrılmıyor" varsayımı yanlış çıktı, gerçek
  bir `UnboundLocalError` ile bulundu) gerçek sdpa implementasyonu
  yazıldı; `serialized_attn.py`'de yalnızca import-zamanı çökmeyi önleyen
  bir guard (bu pipeline'da hiç çağrılmadığı doğrulandı).
- **`xformers` sonunda TAMAMEN kaldırıldı** — yalnızca işe yaramamakla
  kalmıyor, `facebookresearch/dinov2` torch.hub kodunun (TRELLIS'in
  görüntü koşullandırma modeli) kendi xformers kullanımını da bozuyordu
  (TRELLIS'in ATTN_BACKEND'inden bağımsız, Meta'nın kendi kodu) — dinov2
  zaten temiz bir `try/except ImportError` yedeğine sahip, xformers
  paketi tamamen kaldırılınca o yedek devreye giriyor.
- **spconv ÇALIŞIYOR** — `spconv-cu120` (kendi CUDA 12.0 runtime'ını
  taşıyor, sistem CUDA Toolkit'inden bağımsız) canlı test edildi, gerçek
  bir `SubMConv3d` forward geçişi çalıştı.
- **diff-gaussian-rasterization (mip-splatting) — GEÇ keşfedilen bir
  gizli bağımlılık:** `trellis/renderers/gaussian_render.py`'nin üst
  seviye import'ları saf PyTorch (CUDA uzantısı YOK gibi görünüyordu) —
  ama `GaussianRenderer.render()` fonksiyon GÖVDESİNDE gecikmeli bir
  import var, yalnızca doku pişirme sırasında (`to_glb()`'nin 100
  görünümlük gaussian render'ı) tetikleniyor. İlk okumada kaçırıldı,
  gerçek bir traceback'te bulundu. `autonomousvision/mip-splatting`
  klonlanıp `submodules/diff-gaussian-rasterization` derlendi — AYNI
  MSVC 14.44 + `torch/extension.h` + `.cu` hatası (bkz. Hunyuan3D-2
  `custom_rasterizer` notu) `rasterize_points.cu`/`.h`'de tekrar
  çıktı, aynı narrow-include düzeltmesiyle giderildi; `ext.cpp`
  (PYBIND11_MODULE'ü barındıran, nvcc'den geçmeyen tek `.cpp`) zaten
  kendi `torch/extension.h`'ına sahipti, değişmedi. **LİSANS UYARISI:**
  bu bileşen Inria/MPII Gaussian-Splatting Lisansı — yalnızca araştırma/
  değerlendirme, TİCARİ KULLANIM YOK (OpenLRM'in CC-BY-NC-4.0'ıyla aynı
  muamele, UI'da gösteriliyor).
- **`open3d` atlandı:** `trellis/pipelines/__init__.py` upstream'de hem
  image hem text-to-3D pipeline'ını eager import ediyor, text pipeline
  yalnızca open3d (~500MB) gerektiriyor ve bu adapter yalnızca image-to-3D
  destekliyor (diğer beş aileyle aynı desen) — text import'u kaldırıldı.
- **Checkpoint yükleme sorunsuz:** Hunyuan3D-2'nin IP-Adapter sorununun
  AKSİNE, TRELLIS'in her checkpoint'i (`ss_flow_img_dit`, `slat_flow_
  img_dit`, `ss_dec_conv3d`, `slat_decoder_mesh`, `slat_decoder_gs`)
  ilk denemede `strict=True` ile sorunsuz yüklendi.

**Canlı doğrulama (gerçek `/v1/jobs` API'si üzerinden):**
`num_inference_steps=25, formats=[mesh,gaussian], texture_size=1024,
simplify=0.95` ile 9.862 vertex/15.750 yüz, gerçek pişirilmiş
`TextureVisuals` materyali doğrulandı. VRAM tepe noktası (ısınmış model,
ikinci çağrı): **11.583MB** — altı aile arasında en yüksek. Isınmış
model süresi ~28sn (çoğu 100-görünümlük gaussian render + doku pişirmede,
~18sn; difüzyon örneklemesinin kendisi yalnızca ~7sn). İlk kullanımda
ayrıca ~130sn'lik bir `facebookresearch/dinov2` torch.hub indirmesi var
(bu tahmine dahil değil, OpenLRM/InstantMesh'in DINO notlarıyla aynı
desen).

Vendor: `trellis/` (MIT) + `trellis/representations/mesh/flexicubes/`
(Apache-2.0, ayrıca fetch edilen submodule) + `third_party/diff-gaussian-
rasterization/` (Inria/MPII NC lisansı) hepsi `vendor/trellis/`'e
kopyalandı. Ayrıntı ve tüm yerel yamaların tam listesi:
`vendor/trellis/NOTICE.md`.

**Kademeli listenin geri kalanı (CRM):** CUDA Toolkit engeli kalktığı
için artık teorik olarak tractable — henüz adapter yazılmadı, bu iş
devam ediyor.
