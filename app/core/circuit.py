from __future__ import annotations

import time
from typing import Dict

from app.core.config import get_settings

# Estado único do circuit breaker em memória
_state: Dict[str, float] = {
    "first_ts": 0.0,     # primeiro erro na janela
    "fails": 0.0,        # quantidade de falhas na janela
    "open_until": 0.0,   # se > 0, circuito está aberto até esse instante
}


def cb_is_open() -> bool:
    """
    Verifica se o circuito está aberto.

    - Quando aberto, pedimos para o service não chamar a OpenAI
    - Se o cooldown já passou, reabrimos (half-open -> closed)
    """
    settings = get_settings()
    now = time.monotonic()

    # Nunca abriu
    if _state["open_until"] == 0.0:
        return False

    # Cooldown passou, fecha de novo
    if now >= _state["open_until"]:
        _state["open_until"] = 0.0
        _state["fails"] = 0
        _state["first_ts"] = 0.0
        return False

    # Ainda dentro do período de bloqueio
    return True


def cb_on_success() -> None:
    """
    Deve ser chamado quando a chamada à OpenAI deu certo.

    - Reseta o contador de falhas
    - Fecha o circuito se estivesse meio aberto
    """
    _state["fails"] = 0
    _state["first_ts"] = 0.0
    _state["open_until"] = 0.0


def cb_on_failure() -> None:
    """
    Deve ser chamado quando a chamada à OpenAI falha.

    - Conta falhas dentro de uma janela (CB_WINDOW_SEC)
    - Se passar de CB_MAX_FAILS, abre o circuito por CB_COOLDOWN_SEC
    """
    settings = get_settings()
    now = time.monotonic()

    # Se a primeira falha é antiga, iniciamos uma janela nova
    if _state["first_ts"] == 0.0 or now - _state["first_ts"] > settings.CB_WINDOW_SEC:
        _state["first_ts"] = now
        _state["fails"] = 1
        return

    # Ainda dentro da janela: incrementa falhas
    _state["fails"] += 1

    # Se estourou o limite, abre o circuito
    if _state["fails"] >= settings.CB_MAX_FAILS:
        _state["open_until"] = now + settings.CB_COOLDOWN_SEC
