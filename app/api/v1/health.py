from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.circuit import cb_is_open

router = APIRouter()


@router.get("/healthz")
async def healthz():
    """
    Health check simples.

    - Usa para saber se o processo está de pé
    - Não chama OpenAI
    """
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.MODEL,
    }


@router.get("/readyz")
async def readyz():
    """
    Readiness check.

    - Indica se a API está pronta para receber tráfego real
    - Verifica se há API key e se o circuito está fechado
    """
    settings = get_settings()

    ready = bool(settings.OPENAI_API_KEY) and not cb_is_open()

    return {
        "ready": ready,
        "circuit_open": cb_is_open(),
        "has_api_key": bool(settings.OPENAI_API_KEY),
    }
