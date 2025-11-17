from fastapi import FastAPI

# Importa as rotas organizadas em módulos separados
from app.api.v1 import insight, resume, health

APP_TITLE = "ReStartAI-Insight-Min+"
APP_VERSION = "1.3.0"


def create_app() -> FastAPI:
    """
    Cria e configura a aplicação FastAPI.

    Aqui a gente:
    - define título e versão da API
    - registra os routers (rotas) de cada módulo:
        - insight: /insight
        - resume: /resume-summary
        - health: /healthz e /readyz
    """
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
    )

    # Cada router cuida de um grupo de endpoints
    app.include_router(insight.router, prefix="", tags=["insight"])
    app.include_router(resume.router, prefix="", tags=["resume"])
    app.include_router(health.router, prefix="", tags=["health"])

    return app


# Objeto usado pelo Uvicorn para rodar o servidor
app = create_app()
