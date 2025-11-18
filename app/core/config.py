from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # IMPORTANTE: agora a API key vem da env RESTARTAI_OPENAI_KEY,
    # não mais de OPENAI_API_KEY.
    OPENAI_API_KEY: str = Field("", alias="RESTARTAI_OPENAI_KEY")

    MODEL: str = "gpt-4o-mini"

    INTERNAL_KEY: str = "minha-internal-key"

    OPENAI_TIMEOUT: int = 10
    OPENAI_MAX_TOKENS: int = 80
    RESUME_MAX_TOKENS: int = 512
    OPENAI_TEMPERATURE: float = 0.2
    OPENAI_BASE_URL: str = ""

    CACHE_TTL_SECONDS: int = 60
    CACHE_MAX_KEYS: int = 128

    RATE_LIMIT_PER_MIN: int = 6

    CB_MAX_FAILS: int = 3
    CB_WINDOW_SEC: int = 60
    CB_COOLDOWN_SEC: int = 30

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
