from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centraliza todas as configurações do app.

    - Lê valores do .env automaticamente
    - Permite override por variável de ambiente
    """

    # Config do Pydantic Settings (v2)
    model_config = SettingsConfigDict(
        extra="ignore",          # ignora variáveis extra no ambiente
        env_file=".env",         # arquivo padrão
        env_file_encoding="utf-8",
    )

    # OpenAI
    OPENAI_API_KEY: str = ""
    MODEL: str = "gpt-4o-mini"
    OPENAI_TIMEOUT: float = 10.0
    OPENAI_MAX_TOKENS: int = 80
    RESUME_MAX_TOKENS: int = 512
    OPENAI_TEMPERATURE: float = 0.2
    OPENAI_BASE_URL: str | None = None  

    # Segurança / Auth interna
    INTERNAL_KEY: str = ""

    # Cache em memória
    CACHE_TTL_SECONDS: int = 900  # 15 minutos
    CACHE_MAX_KEYS: int = 1000

    # Rate limit (pedidos por minuto por usuário)
    RATE_LIMIT_PER_MIN: int = 6

    # Circuit breaker para erros de OpenAI
    CB_MAX_FAILS: int = 3
    CB_WINDOW_SEC: int = 60
    CB_COOLDOWN_SEC: int = 30


@lru_cache
def get_settings() -> Settings:
    """
    Retorna uma instância única de Settings (cacheada).
    """
    return Settings()
