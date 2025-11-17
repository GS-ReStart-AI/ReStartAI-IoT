from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict

from app.core.config import get_settings

# Armazena, para cada userId, os timestamps (monotonic) das últimas chamadas
_requests: Dict[str, Deque[float]] = {}


def check_rate(user_id: str) -> None:
    """
    Aplica rate limit por usuário.

    - Usa RATE_LIMIT_PER_MIN do Settings
    - Janela deslizante de 60 segundos
    - Se passar do limite, dispara RuntimeError("rate_limit_exceeded")
    """
    settings = get_settings()
    now = time.monotonic()

    # Fila de timestamps para esse usuário
    q = _requests.setdefault(user_id, deque())

    # Remove chamadas mais antigas que 60s
    while q and now - q[0] > 60:
        q.popleft()

    # Se já atingiu o limite, bloqueia
    if len(q) >= settings.RATE_LIMIT_PER_MIN:
        raise RuntimeError("rate_limit_exceeded")

    # Registra a chamada atual
    q.append(now)
