"""
Camada de infraestrutura (core).

Aqui ficam:
- config.py        -> Settings / variáveis de ambiente
- cache.py         -> cache em memória
- rate_limit.py    -> rate limit por usuário
- circuit.py       -> circuit breaker para a IA
- openai_client.py -> cliente da OpenAI
- prompts.py       -> textos dos prompts
"""

from .config import get_settings  
