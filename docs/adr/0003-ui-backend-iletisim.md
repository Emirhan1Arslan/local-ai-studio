# ADR-0003: UI ↔ Backend İletişimi — REST + WebSocket, Token

**Durum:** accepted · 2026-07-18

**Karar:** localhost üzerinde REST `/v1` (kontrol) + WebSocket `/v1/ws` (olay/stream), tek
kullanımlık token ile.

**Gerekçe:** Aynı API hem UI hem dış istemciler (SDK, MCP) tarafından kullanılır → tek yüzey,
çift iş yok.

**Sonuçlar:** WS olay şemaları (`job.*`, `download.*`, `system.*`, `log`) sürümlü ve kararlı
tutulmalıdır; UI'a özel gizli endpoint açılmaz.
