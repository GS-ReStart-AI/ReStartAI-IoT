from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import assert_internal_key, get_request_id
from app.core.openai_client import AIServiceError
from app.domain.models import ResumeSummaryRequest, ResumeSummaryResponse
from app.services.resume_service import generate_resume_summary

router = APIRouter()


@router.post(
    "/resume-summary",
    response_model=ResumeSummaryResponse,
)
async def create_resume_summary(
    req: ResumeSummaryRequest,
    _auth: None = Depends(assert_internal_key),
    request_id: str = Depends(get_request_id),
):
    """
    Endpoint chamado pelo backend C# para analisar o currículo.

    - Valida X-Internal-Key (auth interna)
    - Gera/recebe X-Request-Id
    - Chama o service de resumo de currículo
    """
    try:
        result = await generate_resume_summary(req)
        return result

    except AIServiceError as e:
        # Erros vindos da IA viram 502 para o cliente
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erro ao gerar resumo de currículo: {e.public_msg}",
        )
