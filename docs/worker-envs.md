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

UI'da 7 satırlık ham liste yerine tek/temiz bir seçim için `VideoPanel.tsx`
yalnızca `vae` bileşenini (her pipeline'da tam olarak bir dosya, parçalanmamış)
gösterip görünen adı üst klasörden (gerçek model adı, ör. "LTX-Video") türetiyor.

**Bu makinede canlı olarak yerleştirildi:** `models/video/LTX-Video/` (indirilen
~27 GB'lık tam snapshot, HF cache'inden doğrudan kopyalandı — indirme motoru
yeniden çalıştırılmadı, Faz 7'deki ses modelleri deseniyle aynı). Doğrulama:
`HF_HUB_OFFLINE=1` + boş `HF_HOME` ile, Model Browser'dan seçilmiş gibi bir
bileşen dosyası `model_path` olarak verilip gerçek bir üretim hiçbir ağ
isteği yapmadan başarıyla tamamlandı.
