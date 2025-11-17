from __future__ import annotations

import uuid

from fastapi import Depends, Header, HTTPException, status

from app.core.config import get_settings
from app.core.rate_limit import check_rate


async def assert_internal_key(
    x_internal_key: str | None = Header(default=None, alias="X-Internal-Key"),
) -> None:
    """
    Valida a chave interna enviada no header X-Internal-Key.

    - Compara com INTERNAL_KEY do .env
    - Se não bater, devolve 401
    """
    settings = get_settings()

    if not settings.INTERNAL_KEY:
        # Config errada/ausente
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuração interna inválida (INTERNAL_KEY ausente).",
        )

    if not x_internal_key or x_internal_key != settings.INTERNAL_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Não autorizado.",
        )


def get_request_id(
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> str:
    """
    Garante que sempre exista um request_id.

    - Se o cliente mandar X-Request-Id, usa ele
    - Senão, gera um UUID aleatório
    """
    return x_request_id or str(uuid.uuid4())


def apply_rate_limit(user_id: str) -> None:
    """
    Aplica rate limit por usuário usando o core.rate_limit.

    - Converte o erro genérico em HTTP 429 lá na rota
    """
    check_rate(user_id)
