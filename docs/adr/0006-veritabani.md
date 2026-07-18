# ADR-0006: Veritabanı — SQLite + SQLAlchemy 2 + Alembic

**Durum:** accepted · 2026-07-18

**Karar:** SQLite + SQLAlchemy 2 + Alembic; vektörler için sqlite-vec (büyük korpus: LanceDB).

**Gerekçe:** Sıfır kurulum, tek dosya, taşınabilir proje yedekleri.

**Sonuçlar:** Çok kullanıcılı sunucu senaryosu hedef değildir; WAL modu + tek yazar deseni
kullanılır.
