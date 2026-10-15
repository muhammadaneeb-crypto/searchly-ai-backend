"""Application settings, loaded from environment variables / backend/.env."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_KEYS = {"", "your_groq_api_key_here", "changeme"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- GroqCloud (the ONLY external AI provider) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_seconds: float = Field(45, gt=0)
    groq_max_retries: int = Field(2, ge=0, le=5)

    # --- Server ---
    backend_host: str = "0.0.0.0"
    backend_port: int = Field(8000, ge=1, le=65535)
    log_level: str = "INFO"

    # --- Uploads ---
    max_upload_mb: int = Field(20, ge=1, le=200)
    max_audio_mb: int = Field(25, ge=1, le=100)
    ocr_languages: str = "eng"  # e.g. "eng+urd" if you installed the Urdu Tesseract data

    # --- Web search / page fetching ---
    search_region: str = "wt-wt"
    search_max_queries: int = Field(3, ge=1, le=5)
    search_results_per_query: int = Field(6, ge=1, le=15)
    search_timeout_seconds: float = Field(15, gt=0)
    pages_to_fetch: int = Field(5, ge=0, le=10)
    page_timeout_seconds: float = Field(10, gt=0)
    page_max_bytes: int = 2_000_000
    page_text_chars: int = 6000
    max_concurrent_fetches: int = Field(4, ge=1, le=10)
    respect_robots: bool = True
    cache_ttl_seconds: int = 900
    max_context_chars: int = 24000
    user_agent: str = "SearchlyAI-Bot/1.0 (research assistant; respects robots.txt)"

    # --- Local speech recognition (optional) ---
    whisper_enabled: bool = True
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_audio_bytes(self) -> int:
        return self.max_audio_mb * 1024 * 1024

    @property
    def groq_configured(self) -> bool:
        return self.groq_api_key.strip() not in PLACEHOLDER_KEYS

    def config_problems(self) -> list[str]:
        """Human-readable problems. Never includes secret values."""
        problems = []
        if not self.groq_configured:
            problems.append("GROQ_API_KEY is missing or still the placeholder. Chat will not work until you set it in backend/.env.")
        if not self.groq_model.strip():
            problems.append("GROQ_MODEL is empty.")
        if not self.groq_base_url.startswith("https://"):
            problems.append("GROQ_BASE_URL should start with https://")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
