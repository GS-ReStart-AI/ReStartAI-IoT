"""
Camada de serviços (regras de negócio).

- insight_service.py -> geração de insight
- resume_service.py  -> geração de resumo de currículo
"""

from .insight_service import generate_insight  # noqa: F401
from .resume_service import generate_resume_summary  # noqa: F401
