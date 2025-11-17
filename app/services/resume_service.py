from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.core.circuit import cb_is_open, cb_on_failure, cb_on_success
from app.core.config import get_settings
from app.core.openai_client import chat_completion, AIServiceError
from app.core.prompts import RESUME_SYSTEM_PROMPT, RESUME_USER_TEMPLATE
from app.domain.models import ResumeSummaryRequest, ResumeSummaryResponse


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Tenta extrair JSON da resposta da IA.

    - Primeiro tenta json.loads direto
    - Se falhar, tenta pegar o primeiro {...} com regex
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
    Normaliza e valida a saída da IA para resumo de currículo.

    Garante tipos corretos, valores padrão e limites básicos.
    """
    if not isinstance(obj, dict):
        raise AIServiceError("Formato inválido para resumo de currículo")

    areas = obj.get("areas") or []
    roles = obj.get("roles") or []
    best_role = obj.get("best_role") or (roles[0] if roles else "")
    seniority = obj.get("seniority") or "junior"
    years = obj.get("years_of_experience") or 0
    skills = obj.get("skills_detected") or []
    queries = obj.get("job_search_queries") or []

    if not isinstance(areas, list):
        areas = [str(areas)]
    if not isinstance(roles, list):
        roles = [str(roles)]
    if not isinstance(skills, list):
        skills = [str(skills)]

    norm_queries: List[Dict[str, Any]] = []
    for q in queries:
        if not isinstance(q, dict):
            continue

        title = str(q.get("title") or "Vagas recomendadas")
        query = str(q.get("query") or best_role or "vagas junior")
        platforms = q.get("platforms") or ["linkedin"]

        if not isinstance(platforms, list):
            platforms = [str(platforms)]

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
        "years_of_experience": int(years) if years is not None else 0,
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
            obj = _extract_json(raw)
            out_dict = _validate_resume_out(obj)

            cb_on_success()
            return ResumeSummaryResponse(**out_dict)

        except (json.JSONDecodeError, AIServiceError, TimeoutError, Exception) as e:
            last_error = e
            cb_on_failure()
            messages.append(
                {
                    "role": "system",
                    "content": "Responda apenas o JSON especificado, sem texto extra.",
                }
            )
            continue

    raise AIServiceError(f"Falha ao gerar resumo de currículo: {last_error!r}")
