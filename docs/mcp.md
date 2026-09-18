# MCP Sunucusu ve Uzaktan Erişim (Faz 12)

Local AI Studio'nun **birincil entegrasyon yüzeyi** MCP'dir (docs/architecture.md):
Claude Desktop, Claude Code, Cursor, VS Code, Windsurf ve her MCP istemcisi aynı araç
setiyle görsel/video/3D/ses/LLM üretir, model indirir, workflow çalıştırır.

İki transport, tek kod:

| Transport | Nerede çalışır | Kim kullanır |
|---|---|---|
| **streamable HTTP** `http://<host>:8710/mcp` | `studio-api` sürecinin içinde (mount) | Cursor, VS Code, Claude Code, özel istemciler — header verebilenler |
| **stdio** `studio-mcp` | İstemcinin başlattığı ayrı süreç; Studio'ya REST ile bağlanır | Claude Desktop ve header veremeyen istemciler; **uzak makine** senaryosu |

Her araç Studio'nun `/v1` REST API'sini çağırır (`studio_mcp.StudioClient`). İç servislere
doğrudan erişim yok — sunucu Studio ile aynı makinede de, uzak bir dizüstünde de aynıdır.

## Araçlar

| Araç | Ne yapar |
|---|---|
| `system_info` | GPU/VRAM/RAM/backend, Studio sürümü, API adresi |
| `list_models`, `search_models`, `download_model`, `install_model` | Kurulu modeller · HF Hub arama · tek dosya indir · çok dosyalı repo kur |
| `generate_image`, `edit_image` | txt2img (SDXL/SD3/FLUX otomatik aile) · img2img/inpaint/outpaint |
| `remove_background`, `upscale`, `caption`, `ocr`, `segment` | Görüntü araç seti (BiRefNet, ESRGAN, Florence-2, EasyOCR, SAM+DINO) |
| `generate_video` | LTX / Wan / CogVideoX / Hunyuan / Mochi, t2v + i2v |
| `generate_3d` | TripoSR / SF3D / OpenLRM / Hunyuan3D-2 / InstantMesh / TRELLIS / CRM |
| `transcribe_audio`, `synthesize_voice` | faster-whisper STT · Kokoro / XTTS-v2 TTS |
| `chat` | Yerel GGUF LLM sohbeti (oturumlu) |
| `run_job`, `get_job`, `wait_job`, `cancel_job`, `list_jobs` | Ham iş kuyruğu + **job-id + poll** deseni |
| `upload_file` | Uzak istemciden Studio makinesine girdi dosyası |
| `list_node_types`, `list_workflows`, `workflow_create`, `workflow_save`, `workflow_run`, `workflow_export` | Node workflow'ları |

**Uzun işler:** üretim araçları varsayılan `wait=true` ile bitene kadar bekler ve durum
geçişlerini MCP progress bildirimi olarak akıtır. `wait=false` hemen `job_id` döner;
`wait_job`/`get_job` ile takip edilir. Zaman aşımı `timeout_s` (vars. 900 sn; video 3600).

**Çıktılar:** `output_path` Studio makinesindeki dosyadır; `output_url`
(`http://<host>:<port>/outputs/<uuid>.png`) uzak istemcinin indirmesi içindir.
`generate_image(include_image=true)` PNG'yi doğrudan içerik olarak da döner (Claude
Desktop görseli gösterir).

**Model seçimi:** `model_path` boş bırakılırsa kurulu ilk uygun model kullanılır
(`checkpoint`/`llm`/`upscale`). Yoksa araç anlaşılır bir hatayla `search_models` →
`download_model`/`install_model` yolunu gösterir.

**İzin ipuçları:** araçlar MCP `ToolAnnotations` taşır — listeleme/sorgu `readOnlyHint`,
indirme `openWorldHint`, `cancel_job` `destructiveHint`. İstemciler bunlara göre onay ister.

## Kurulum tarifleri

Ön koşul: Studio çalışıyor olmalı (`uv run studio-api` ya da masaüstü uygulaması).
Masaüstü uygulaması her açılışta rastgele port + token üretir; sabit adres için backend'i
elle başlat: `uv run studio-api --port 8710` (dev: token'sız) veya `--token <gizli>`.

### Claude Desktop (stdio)

`claude_desktop_config.json` (Windows: `%APPDATA%\Claude\`, macOS: `~/Library/Application
Support/Claude/`):

```json
{
  "mcpServers": {
    "local-ai-studio": {
      "command": "uv",
      "args": ["run", "--directory", "C:/Users/<sen>/Desktop/Studio", "studio-mcp"],
      "env": {
        "STUDIO_API_URL": "http://127.0.0.1:8710",
        "STUDIO_API_TOKEN": ""
      }
    }
  }
}
```

Claude Desktop'ı yeniden başlat; "Local AI Studio" araçları çekiç simgesinde görünür.
DoD denemesi: *"Bir kedi görseli üret ve dosya yolunu söyle"* → `generate_image` çalışır,
`output_path` döner.

### Claude Code

```bash
claude mcp add --transport http local-ai-studio http://127.0.0.1:8710/mcp
# token'lı:
claude mcp add --transport http local-ai-studio http://127.0.0.1:8710/mcp \
  --header "Authorization: Bearer <token>"
```

### Cursor (`.cursor/mcp.json`) — HTTP

```json
{
  "mcpServers": {
    "local-ai-studio": {
      "url": "http://127.0.0.1:8710/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

### VS Code (`.vscode/mcp.json`)

```json
{
  "servers": {
    "local-ai-studio": {
      "type": "http",
      "url": "http://127.0.0.1:8710/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

### Windsurf (`~/.codeium/windsurf/mcp_config.json`)

```json
{
  "mcpServers": {
    "local-ai-studio": {
      "serverUrl": "http://127.0.0.1:8710/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

### Özel istemci (Python, resmi SDK)

```python
import asyncio
from mcp import Client

async def main():
    async with Client("http://127.0.0.1:8710/mcp") as c:
        r = await c.call_tool("generate_image", {"prompt": "bir kedi, fotogerçekçi"})
        print(r.structured_content or r.content[0].text)

asyncio.run(main())
```

## Uzaktan çalışma (LAN / VPN)

Studio GPU'lu makinede (A), istemci başka makinede (B):

1. **A'da Studio'yu ağa aç** — token zorunludur, yoksa otomatik üretilir ve stderr'e yazılır:
   ```bash
   uv run studio-api --host 0.0.0.0 --port 8710 --token <gizli>
   # veya: STUDIO_HOST=0.0.0.0 STUDIO_API_TOKEN=<gizli> uv run studio-api
   ```
   Loopback dışına bağlanırken `studio-api` bir **LAN modu uyarısı** basar. Yalnız
   güvendiğin ağda/VPN'de kullan; Windows Güvenlik Duvarı'nda 8710/TCP'ye izin ver.

2. **B'den bağlan** — iki yol:
   - HTTP destekleyen istemci (Cursor/VS Code/Claude Code): `http://<A-ip>:8710/mcp` +
     `Authorization: Bearer <gizli>` (yukarıdaki tariflerde adresi değiştir).
   - Claude Desktop (header veremez): B'ye yalnız `studio-mcp` paketini kur — Studio'nun
     kendisine, torch'a vb. gerek yok — ve stdio köprüsünü uzak adrese yönlendir:
     ```bash
     uv tool install ./backend/studio_mcp      # ya da: pip install ./backend/studio_mcp
     ```
     ```json
     {
       "mcpServers": {
         "local-ai-studio": {
           "command": "studio-mcp",
           "env": { "STUDIO_API_URL": "http://<A-ip>:8710", "STUDIO_API_TOKEN": "<gizli>" }
         }
       }
     }
     ```

3. **Dosyalar:** girdi görselleri B'de ise `upload_file` ile A'ya gönder (dönen `path`'i
   kullan); çıktılar A'da durur, `output_url` ile B'ye indir.

`studio-mcp --transport http --host 0.0.0.0 --port 8711` ile köprü, bağımsız bir HTTP MCP
sunucusu olarak da çalışır (Studio'nun `/mcp` mount'u yeterli olmadığında — ör. Studio'yu
ağa açmadan, aynı makinedeki bir tünel arkasında).

## Güvenlik notları

- `/mcp`, REST `/v1` ile **aynı** Bearer token kapısının arkasındadır (`_BearerGuard`);
  token yokken (dev modu) ikisi de açıktır — bu yalnız loopback'te kabul edilir, `studio-api`
  loopback dışına token'sız çıkmayı reddeder (otomatik token üretir).
- `/outputs/*` token'sız servis edilir (UI `<img src>` ile okur); dosya adları tahmin
  edilemez uuid'lerdir.
- MCP'nin DNS-rebinding koruması mount'ta kapalıdır: LAN modunda `Host` başlığı IP/makine
  adı olur; koruma token kapısıyla sağlanır.
- XTTS-v2 (CPML), OpenLRM (CC-BY-NC), TRELLIS doku pişirme (NC) gibi ticari-olmayan lisanslı
  motorlar araç açıklamalarında işaretlidir; ses klonlamada `consent=true` zorunludur.
