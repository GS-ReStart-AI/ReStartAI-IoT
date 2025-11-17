"""
Pacote principal da aplicação FastAPI.

- Esta pasta contém:
    - main.py      -> ponto de entrada da API
    - api/         -> rotas (endpoints)
    - core/        -> configurações e infraestrutura
    - domain/      -> modelos e contratos
    - services/    -> regras de negócio
"""

# Reexporta a app para facilitar imports opcionais
from .main import app  # noqa: F401
