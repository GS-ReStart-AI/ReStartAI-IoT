from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from openai import OpenAI

from app.core.config import get_settings


class AIServiceError(RuntimeError):
    """
    Erro genérico da camada de IA.

    - Carrega uma mensagem "public_msg" que podemos enviar para o cliente
    - É usado pelos services para diferenciar erro de negócio x erro da IA
    """

    def __init__(self, msg: str):
        super().__init__(msg)
        self.public_msg = msg


def _client() -> OpenAI:
    """
    Cria uma instância do cliente OpenAI usando as configs do .env.

    - Usa OPENAI_API_KEY (obrigatório)
    - Usa OPENAI_TIMEOUT
    - Usa OPENAI_BASE_URL se você estiver atrás de um proxy/gateway
    """
    settings = get_settings()

    kw: Dict[str, Any] = {
        "api_key": settings.OPENAI_API_KEY,
        "timeout": settings.OPENAI_TIMEOUT,
    }

    if settings.OPENAI_BASE_URL:
        kw["base_url"] = settings.OPENAI_BASE_URL

    return OpenAI(**kw)


async def chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> str:
    """
    Wrapper assíncrono para chamar o modelo de chat.

    - Recebe mensagens já prontas (system/user/assistant)
    - Usa o MODEL e TEMPERATURE das configs
    - Roda o client síncrono em thread separada (asyncio.to_thread)
    """
    settings = get_settings()

    if not settings.OPENAI_API_KEY:
        raise AIServiceError("OPENAI_API_KEY ausente")

    cli = _client()

    try:
        # Cliente é síncrono; usamos to_thread para não travar o loop async
        resp = await asyncio.to_thread(
            cli.chat.completions.create,
            model=settings.MODEL,
            messages=messages,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # Qualquer erro da OpenAI vira AIServiceError
        raise AIServiceError(str(e))

    # Conteúdo da primeira escolha
    content = resp.choices[0].message.content or ""
    return content
