from __future__ import annotations

import textwrap
from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.deps import assert_internal_key, get_request_id, apply_rate_limit
from app.core.openai_client import AIServiceError
from app.domain.models import ResumeSummaryRequest, ResumeSummaryResponse
from app.services.resume_service import generate_resume_summary

router = APIRouter()


@router.post(
    "/resume-summary",
    response_model=ResumeSummaryResponse,
)
async def post_resume_summary(
    req: ResumeSummaryRequest,
    response: Response,
    _auth: None = Depends(assert_internal_key),
    request_id: str = Depends(get_request_id),
):
    response.headers["X-Request-Id"] = request_id
    apply_rate_limit(req.usuarioId)

    # DEBUG: ver o que está chegando do C#
    preview = (req.curriculoTexto or "").replace("\r", " ").replace("\n", " ")
    preview = preview[:400]
    print("\n====== CURRÍCULO RECEBIDO ======")
    print(f"usuarioId={req.usuarioId}")
    print(textwrap.fill(preview, width=100))
    print("====== FIM CURRÍCULO ======\n")

    try:
        result = await generate_resume_summary(req)
        return result

    except AIServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.public_msg,
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Falha temporária ao gerar resumo de currículo. Tente novamente mais tarde.",
        )
