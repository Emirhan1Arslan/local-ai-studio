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
