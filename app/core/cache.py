from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, Optional, Tuple

from app.core.config import get_settings
from app.domain.models import InsightRequest

# Estrutura simples de cache em memória:
_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}


def _now() -> float:
    return time.time()


def cache_key_from_insight(req: InsightRequest) -> str:
    """
    Gera uma chave de cache estável a partir do conteúdo do InsightRequest.

    - Usa apenas campos relevantes para o insight
    - Serializa para JSON com sort_keys=True
    - Hash SHA-256 para ficar curto e consistente
    """
    base = {
        "userId": req.userId,
        "metrics": req.metrics.model_dump(mode="json"),
        "lastEvents": [e.model_dump(mode="json") for e in req.lastEvents],
        "profile": req.profile.model_dump(mode="json"),
        "bestOpportunity": (
            req.bestOpportunity.model_dump(mode="json")
            if req.bestOpportunity
            else None
        ),
    }

    raw = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_get(key: str) -> Optional[Dict[str, str]]:
    """
    Lê um item do cache.

    - Remove automaticamente se estiver expirado
    - Retorna o dict salvo ou None se não existir/vencer
    """
    item = _cache.get(key)
    if not item:
        return None

    exp_ts, value = item
    if _now() > exp_ts:
        # Expirou: remove e retorna None
        _cache.pop(key, None)
        return None

    return value


def cache_put(key: str, value: Dict[str, str]) -> None:
    """
    Salva um item no cache com TTL definido nas configs.

    - value deve ser serializável em JSON (ex.: resposta do insight)
    - Se estourar CACHE_MAX_KEYS, remove um item qualquer
    """
    settings = get_settings()
    ttl = settings.CACHE_TTL_SECONDS

    # Limita quantidade de chaves em memória
    if len(_cache) >= settings.CACHE_MAX_KEYS:
        # Estratégia simples: remove o primeiro item arbitrário
        first_key = next(iter(_cache))
        _cache.pop(first_key, None)

    exp_ts = _now() + ttl
    _cache[key] = (exp_ts, value)
