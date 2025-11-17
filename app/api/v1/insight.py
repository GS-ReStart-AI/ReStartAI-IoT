from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import assert_internal_key, get_request_id, apply_rate_limit
from app.core.openai_client import AIServiceError
from app.domain.models import InsightRequest, InsightResponse
from app.services.insight_service import generate_insight

router = APIRouter()


@router.post(
    "/insight",
    response_model=InsightResponse,
)
async def create_insight(
    req: InsightRequest,
    _auth: None = Depends(assert_internal_key),
    request_id: str = Depends(get_request_id),
):
    """
    Endpoint chamado pelo backend C# para gerar um insight.

    - Valida X-Internal-Key (auth interna)
    - Gera/recebe X-Request-Id (útil para logs/futuro)
    - Aplica rate limit por userId
    - Chama o service de IA
    """
    # Rate limit por usuário (userId vem do corpo)
    try:
        apply_rate_limit(req.userId)
    except RuntimeError:
        # Estourou o limite de chamadas por minuto
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests for this user. Try again later.",
        )

    try:
        # Chama a camada de serviço (regra de negócio + OpenAI)
        result = await generate_insight(req)
        return result

    except AIServiceError as e:
        # Erros vindos da IA viram 502 (bad gateway)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erro ao gerar insight: {e.public_msg}",
        )
