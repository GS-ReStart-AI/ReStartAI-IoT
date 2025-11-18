from __future__ import annotations

import json
import re
import logging
from typing import Any, Dict, List

from app.core.circuit import cb_is_open, cb_on_failure, cb_on_success
from app.core.config import get_settings
from app.core.openai_client import chat_completion, AIServiceError
from app.core.prompts import RESUME_SYSTEM_PROMPT, RESUME_USER_TEMPLATE
from app.domain.models import ResumeSummaryRequest, ResumeSummaryResponse

log = logging.getLogger(__name__)


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Tenta extrair JSON da resposta da IA.

    - Primeiro tenta dar json.loads direto.
    - Se falhar, tenta achar o primeiro bloco {...} com regex.
    """
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _validate_resume_out(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza/valida o JSON de saída do modelo para caber em ResumeSummaryResponse.
    """
    if not isinstance(obj, dict):
        raise AIServiceError("Formato inválido para resumo de currículo")

    areas = obj.get("areas") or []
    if not isinstance(areas, list):
        areas = [areas]
    areas = [str(a).strip() for a in areas if str(a).strip()]

    best_role = obj.get("best_role") or obj.get("bestRole") or ""
    best_role = str(best_role).strip()

    roles = obj.get("roles") or []
    if not isinstance(roles, list):
        roles = [roles]
    roles = [str(r).strip() for r in roles if str(r).strip()]

    seniority = obj.get("seniority") or ""
    seniority = str(seniority).strip()

    years = obj.get("years_of_experience") or obj.get("yearsOfExperience") or 0
    try:
        years_int = int(years)
    except (TypeError, ValueError):
        years_int = 0
    if years_int < 0:
        years_int = 0

    skills = obj.get("skills_detected") or obj.get("skillsDetected") or []
    if not isinstance(skills, list):
        skills = [skills]
    skills = [str(s).strip() for s in skills if str(s).strip()]

    raw_queries = obj.get("job_search_queries") or obj.get("jobSearchQueries") or []
    norm_queries: List[Dict[str, Any]] = []

    if isinstance(raw_queries, list):
        for q in raw_queries:
            if not isinstance(q, dict):
                continue

            title = str(q.get("title") or "").strip()
            query = str(q.get("query") or "").strip()
            platforms = q.get("platforms") or []
            if not isinstance(platforms, list):
                platforms = [platforms]
            platforms = [str(p).strip() for p in platforms if str(p).strip()]

            if not title and not query:
                continue

            norm_queries.append(
                {
                    "title": title,
                    "query": query,
                    "platforms": platforms,
                }
            )

    if not best_role and roles:
        best_role = roles[0]

    return {
        "areas": [str(a) for a in areas][:5],
        "best_role": str(best_role),
        "roles": [str(r) for r in roles][:8],
        "seniority": str(seniority),
        "years_of_experience": years_int,
        "skills_detected": [str(s) for s in skills][:30],
        "job_search_queries": norm_queries[:5],
    }


async def generate_resume_summary(req: ResumeSummaryRequest) -> ResumeSummaryResponse:
    """
    Função principal chamada pela rota /resume-summary.

    Fluxo:
    - Verifica circuit breaker
    - Monta schema + mensagens para o modelo
    - Tenta até 2 chamadas à IA
    - Normaliza a saída com _validate_resume_out
    """
    settings = get_settings()

    if cb_is_open():
        raise AIServiceError("Circuito aberto")

    schema = (
        '{"areas":["string"],'
        '"best_role":"string",'
        '"roles":["string"],'
        '"seniority":"junior|pleno|senior|estagio",'
        '"years_of_experience":1,'
        '"skills_detected":["string"],'
        '"job_search_queries":[{"title":"string","query":"string","platforms":["linkedin","indeed","gupy"]}]}'
    )

    messages = [
        {"role": "system", "content": RESUME_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RESUME_USER_TEMPLATE.format(
                schema=schema,
                resume_text=req.curriculoTexto,
            ),
        },
    ]

    last_error: Exception | None = None

    for _ in range(2):
        try:
            raw = await chat_completion(
                messages=messages,
                max_tokens=settings.RESUME_MAX_TOKENS,
            )
            # log bruto (só pra debug; se quiser, pode comentar depois)
            log.info("RAW RESUME RESPONSE: %s", raw)

            obj = _extract_json(raw)
            out_dict = _validate_resume_out(obj)

            cb_on_success()
            return ResumeSummaryResponse(**out_dict)

        except (json.JSONDecodeError, AIServiceError, TimeoutError, Exception) as e:
            last_error = e
            cb_on_failure()
            log.exception("Erro ao gerar resumo de currículo: %s", e)

            messages.append(
                {
                    "role": "system",
                    "content": "Responda apenas o JSON especificado, sem texto extra.",
                }
            )
            continue

    raise AIServiceError(f"Falha ao gerar resumo de currículo: {last_error!r}")
