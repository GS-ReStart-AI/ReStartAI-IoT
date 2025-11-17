from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from app.core.cache import cache_get, cache_put, cache_key_from_insight
from app.core.circuit import cb_is_open, cb_on_failure, cb_on_success
from app.core.openai_client import chat_completion, AIServiceError
from app.core.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from app.domain.models import (
    ActionTag,
    InsightRequest,
    InsightResponse,
)


def _build_messages(req: InsightRequest) -> List[Dict[str, str]]:
    """
    Monta as mensagens de system + user para enviar ao modelo.
    Deixa toda a regra de texto encapsulada aqui.
    """
    metrics_str = json.dumps(req.metrics.model_dump(mode="json"), ensure_ascii=False)
    events_str = json.dumps(
        [e.model_dump(mode="json") for e in req.lastEvents],
        ensure_ascii=False,
    )
    profile_str = json.dumps(req.profile.model_dump(mode="json"), ensure_ascii=False)
    bestopp_str = (
        json.dumps(req.bestOpportunity.model_dump(mode="json"), ensure_ascii=False)
        if req.bestOpportunity
        else "null"
    )

    user_content = USER_TEMPLATE.format(
        metrics=metrics_str,
        events=events_str,
        profile=profile_str,
        bestopp=bestopp_str,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Tenta extrair JSON da resposta do modelo.

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


def _normalize_action_tag(tag: Any) -> str:
    """
    Garante que actionTag seja um dos valores permitidos.
    Se vier inválido, escolhe um fallback neutro.
    """
    if isinstance(tag, ActionTag):
        return tag.value

    if isinstance(tag, str):
        tag_lower = tag.strip().lower()
        if tag_lower in {"apply", "explore", "study"}:
            return tag_lower

    # fallback padrão
    return "explore"


def _validate_insight_payload(obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Valida e normaliza o JSON vindo da IA.

    - Garante presença de insight e actionTag
    - Corta insight para 120 chars se passar
    - Normaliza actionTag para um dos 3 valores
    """
    if not isinstance(obj, dict):
        raise AIServiceError("Resposta da IA inválida (esperado dict).")

    if "insight" not in obj:
        raise AIServiceError('Resposta da IA sem campo "insight".')

    insight = str(obj["insight"]).strip()
    if not insight:
        raise AIServiceError("Insight vazio.")

    if len(insight) > 120:
        insight = insight[:120].rstrip()

    tag_raw = obj.get("actionTag", "explore")
    action_tag = _normalize_action_tag(tag_raw)

    return insight, action_tag


async def generate_insight(req: InsightRequest) -> InsightResponse:
    """
    Função principal chamada pela rota /insight.

    Fluxo:
    - Verifica circuit breaker
    - Verifica cache (mesma entrada -> mesma saída)
    - Chama OpenAI (com até 2 tentativas)
    - Atualiza circuit breaker
    - Salva no cache
    """
    if cb_is_open():
        raise AIServiceError("Serviço de IA temporariamente indisponível (circuito aberto).")

    cache_key = cache_key_from_insight(req)
    cached = cache_get(cache_key)
    if cached:
        return InsightResponse(**cached)

    messages = _build_messages(req)
    last_error: Exception | None = None

    for _ in range(2):
        try:
            raw = await chat_completion(
                messages=messages,
                max_tokens=80,
            )
            obj = _extract_json(raw)
            insight, action_tag = _validate_insight_payload(obj)

            result = InsightResponse(
                insight=insight,
                actionTag=ActionTag(action_tag),
            )

            cb_on_success()
            cache_put(
                cache_key,
                {
                    "insight": result.insight,
                    "actionTag": result.actionTag.value,
                },
            )
            return result

        except Exception as e:
            last_error = e
            cb_on_failure()
            # Refina instrução para próxima tentativa
            messages.append(
                {
                    "role": "system",
                    "content": (
                        'Atenção: responda APENAS um JSON válido com as chaves '
                        '"insight" (string, <=120 chars) e "actionTag" '
                        '("apply" | "explore" | "study").'
                    ),
                }
            )

    raise AIServiceError(f"Falha ao gerar insight: {last_error!r}")
