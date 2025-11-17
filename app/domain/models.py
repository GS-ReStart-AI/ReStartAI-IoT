from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class ActionTag(str, Enum):
    """
    Tipos de ação que o insight pode sugerir.
    Mantém exatamente os mesmos valores usados hoje.
    """
    apply = "apply"
    explore = "explore"
    study = "study"


class Metrics(BaseModel):
    """
    Métricas agregadas de comportamento do usuário.
    Mesmo formato do app.py original.
    """
    jobsViewedToday: int = Field(ge=0)
    applyClicksToday: int = Field(ge=0)
    lastEventAt: datetime


class Event(BaseModel):
    """
    Evento recente do usuário (tipo + timestamp).
    Usado na lista lastEvents.
    """
    type: str
    ts: datetime


class Profile(BaseModel):
    """
    Perfil simplificado do usuário.
    """
    areas: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    gaps: List[str] = Field(default_factory=list)


class BestOpportunity(BaseModel):
    """
    Melhor vaga/oportunidade atual para o usuário.
    Opcional na requisição.
    """
    role: Optional[str] = None
    city: Optional[str] = None
    match: Optional[int] = Field(default=None, ge=0, le=100)
    missingSkill: Optional[str] = None


class InsightRequest(BaseModel):
    """
    Payload que o C# envia para gerar um insight.
    Extra="forbid" impede campos desconhecidos.
    """
    model_config = ConfigDict(extra="forbid")

    userId: str
    metrics: Metrics
    lastEvents: List[Event] = Field(default_factory=list)
    profile: Profile
    bestOpportunity: Optional[BestOpportunity] = None


class InsightResponse(BaseModel):
    """
    Resposta enviada de volta para o C# / mobile.
    Mesmo contrato: insight + actionTag.
    """
    model_config = ConfigDict(extra="forbid")

    insight: str = Field(max_length=120)
    actionTag: ActionTag


class JobSearchQuery(BaseModel):
    """
    Sugestões de buscas de vaga com título, query e plataformas.
    Usado na resposta de resumo de currículo.
    """
    title: str
    query: str
    platforms: List[str] = Field(default_factory=list)


class ResumeSummaryRequest(BaseModel):
    """
    Requisição para analisar currículo.
    Mantém os MESMOS nomes de campos que seu C# usa hoje:
    - usuarioId
    - curriculoTexto
    """
    model_config = ConfigDict(extra="forbid")

    usuarioId: str
    curriculoTexto: str = Field(min_length=50, max_length=20000)


class ResumeSummaryResponse(BaseModel):
    """
    Resposta estruturada da análise de currículo.
    """
    model_config = ConfigDict(extra="ignore")

    areas: List[str] = Field(default_factory=list)
    best_role: str
    roles: List[str] = Field(default_factory=list)
    seniority: str
    years_of_experience: int = Field(ge=0)
    skills_detected: List[str] = Field(default_factory=list)
    job_search_queries: List[JobSearchQuery] = Field(default_factory=list)
