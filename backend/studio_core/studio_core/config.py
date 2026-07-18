"""Şema tabanlı ayarlar (ADR: Pydantic Settings) ve kullanıcı veri dizinleri.

Kullanıcı verisi repo dışında yaşar: ~/LocalAIStudio/{models,projects,cache,logs,plugins,envs}
models/ altı plan.md §3'teki şemaya uyar.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_SUBDIRS = [
    "image",
    "video",
    "3d",
    "tts",
    "stt",
    "ocr",
    "embeddings",
    "llm",
    "lora",
    "controlnet",
    "checkpoints",
    "vae",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default_factory=lambda: Path.home() / "LocalAIStudio")
    host: str = "127.0.0.1"
    port: int = 8710
    # Sidecar başlatırken Tauri tarafından üretilir; boşsa auth kapalıdır (yalnız dev).
    api_token: str = ""
    log_level: str = "INFO"
    # Kapalı/lisans onaylı HF repoları için (STUDIO_HF_TOKEN); asla loglanmaz.
    hf_token: str = ""
    # system.telemetry yayın aralığı (sn)
    telemetry_interval: float = 2.0
    # Aynı anda çalışacak worker sayısı (Faz 3'te cihaz başına 1)
    worker_concurrency: int = 1
    # İş türü → worker Python yolu override'ı (JSON: STUDIO_WORKER_PYTHONS)
    # Boşsa ~/LocalAIStudio/envs/<tür> venv'i, o da yoksa ana venv kullanılır.
    worker_pythons: dict[str, str] = {}
    # Boştaki worker'ın (ve yüklü modelinin) bellekte kalma süresi (sn)
    worker_idle_timeout: float = 300.0

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def plugins_dir(self) -> Path:
        return self.data_dir / "plugins"

    @property
    def envs_dir(self) -> Path:
        return self.data_dir / "envs"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "studio.db"

    def ensure_dirs(self) -> None:
        """Tüm kullanıcı veri dizinlerini (idempotent) oluşturur."""
        for d in (
            self.data_dir,
            self.projects_dir,
            self.cache_dir,
            self.logs_dir,
            self.plugins_dir,
            self.envs_dir,
            self.outputs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        for sub in MODEL_SUBDIRS:
            (self.models_dir / sub).mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
