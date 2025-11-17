"""
Camada de domínio.

- models.py -> contratos de entrada e saída (Pydantic models)
"""

from .models import (  
    ActionTag,
    Metrics,
    Event,
    Profile,
    BestOpportunity,
    InsightRequest,
    InsightResponse,
    JobSearchQuery,
    ResumeSummaryRequest,
    ResumeSummaryResponse,
)
