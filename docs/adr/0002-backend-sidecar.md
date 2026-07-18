# ADR-0002: Backend Süreç Modeli — Sidecar

**Durum:** accepted · 2026-07-18

**Karar:** Python backend'i Tauri tarafından yönetilen sidecar süreçtir; başlat/izle/kapat +
çökmede otomatik yeniden başlatma Tauri'nin sorumluluğudur.

**Gerekçe:** Çökme izolasyonu; bağımsız güncellenebilirlik; aynı backend'in headless (Docker)
modda tek başına çalışabilmesi.

**Sonuçlar:** Port + tek kullanımlık token pazarlığı gerekir (ADR-0003); süreç sızıntısı
testleri Faz 1 DoD'sindedir.
