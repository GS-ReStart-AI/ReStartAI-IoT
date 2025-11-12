from __future__ import annotations
import os, asyncio, json, re, time, hashlib, uuid
from enum import Enum
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import deque

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status, Depends, Response
from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI
from openai._exceptions import OpenAIError
import logging

# ---------- Config ----------
load_dotenv(override=True)

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
MODEL = (os.getenv("MODEL") or "gpt-4o-mini").strip()
INTERNAL_KEY = (os.getenv("INTERNAL_KEY") or "").strip()
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "10"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "80"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))           
CACHE_MAX_KEYS = int(os.getenv("CACHE_MAX_KEYS", "1000"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "6"))
CB_MAX_FAILS = int(os.getenv("CB_MAX_FAILS", "3"))
CB_WINDOW_SEC = int(os.getenv("CB_WINDOW_SEC", "60"))
CB_COOLDOWN_SEC = int(os.getenv("CB_COOLDOWN_SEC", "30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s insight %(message)s",
)
log = logging.getLogger("restartai.insight")

# ---------- Schemas ----------
class ActionTag(str, Enum):
    apply = "apply"
    explore = "explore"
    study = "study"

class Metrics(BaseModel):
    jobsViewedToday: int = Field(ge=0)
    applyClicksToday: int = Field(ge=0)
    lastEventAt: datetime

class Event(BaseModel):
    type: str
    ts: datetime

class Profile(BaseModel):
    areas: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    gaps: List[str] = Field(default_factory=list)

class BestOpportunity(BaseModel):
    role: Optional[str] = None
    city: Optional[str] = None
    match: Optional[int] = Field(default=None, ge=0, le=100)
    missingSkill: Optional[str] = None

class InsightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    userId: str
    metrics: Metrics
    lastEvents: List[Event] = Field(default_factory=list)
    profile: Profile
    bestOpportunity: Optional[BestOpportunity] = None

class InsightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insight: str = Field(max_length=120)
    actionTag: ActionTag

# ---------- App ----------
app = FastAPI(title="ReStartAI-Insight-Min+", version="1.1.0")

# ---------- Security ----------
async def assert_internal_key(x_internal_key: str | None = Header(default=None, alias="X-Internal-Key")) -> None:
    if not INTERNAL_KEY:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Config inválida")
    if not x_internal_key or x_internal_key != INTERNAL_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Não autorizado")

# ---------- Correlation-ID ----------
def get_request_id(x_request_id: str | None = Header(default=None, alias="X-Request-Id")) -> str:
    return x_request_id or str(uuid.uuid4())

# ---------- Rate limit por usuário ----------
_rate: Dict[str, deque] = {}
def check_rate(user_id: str) -> None:
    now = time.monotonic()
    q = _rate.setdefault(user_id, deque())
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Muitas requisições")
    q.append(now)

# ---------- Cache TTL + idempotência ----------
_cache: Dict[str, tuple[float, Dict[str, str]]] = {}

def _cache_key(req: InsightRequest) -> str:
    base = {
        "metrics": req.metrics.model_dump(mode="json"),
        "lastEvents": [e.model_dump(mode="json") for e in req.lastEvents],
        "profile": req.profile.model_dump(mode="json"),
        "bestOpportunity": req.bestOpportunity.model_dump(mode="json") if req.bestOpportunity else None,
    }
    s = json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cache_get(key: str) -> Optional[Dict[str, str]]:
    item = _cache.get(key)
    if not item: return None
    exp, val = item
    if time.time() > exp:
        _cache.pop(key, None)
        return None
    return val

def cache_put(key: str, val: Dict[str, str]) -> None:
    if len(_cache) >= CACHE_MAX_KEYS:
        _cache.pop(next(iter(_cache)))  # LRU básica por inserção
    _cache[key] = (time.time() + CACHE_TTL_SECONDS, val)

# ---------- Circuit breaker ----------
_cb_state = {"fails": 0, "first_ts": 0.0, "open_until": 0.0}

def cb_is_open() -> bool:
    return time.time() < _cb_state["open_until"]

def cb_on_success() -> None:
    _cb_state.update({"fails": 0, "first_ts": 0.0, "open_until": 0.0})

def cb_on_failure() -> None:
    now = time.time()
    if _cb_state["first_ts"] == 0.0 or now - _cb_state["first_ts"] > CB_WINDOW_SEC:
        _cb_state["first_ts"] = now
        _cb_state["fails"] = 1
    else:
        _cb_state["fails"] += 1
    if _cb_state["fails"] >= CB_MAX_FAILS:
        _cb_state["open_until"] = now + CB_COOLDOWN_SEC

# ---------- Prompts ----------
SYSTEM_PROMPT = (
    'Voce e o "Gerador de Insights" do ReStart.AI, um app que ajuda pessoas a se recolocar rapido.\n'
    "Contexto do produto:\n"
    "- O app mostra ao usuario uma unica \"melhor oportunidade\" (papel/cargo) com % de match calculada no backend C#.\n"
    "- O backend C# usa regra deterministica: +2 por must, +1 por nice, -1 por gap. Voce NAO recalcula score.\n"
    "Voce gera uma UNICA frase curta (<=120 caracteres) que recomenda a proxima acao com base em IoB e perfil.\n"
    "A frase aparece no RESUMO/NOTIFICACAO. Botao de acao: \"Pesquisar vagas desse papel\".\n\n"
    "Regras OBRIGATORIAS:\n"
    '1) Responda APENAS JSON exatamente: {"insight":"...","actionTag":"apply|explore|study"}\n'
    "2) PT-BR, direto, sem emojis e sem quebras de linha.\n"
    "3) Use apenas metrics, lastEvents, profile, bestOpportunity. Nao invente PII.\n"
    '4) Personalize citando papel/area e cidade quando possivel (ex.: "Analista de CX Jr em Sao Paulo").\n'
    '5) actionTag: "apply" se viu varias vagas e nao aplicou ou acabou de ver a melhor; "explore" baixa atividade; "study" se houver gap/missingSkill.\n'
    "6) Dados insuficientes: dica generica acionavel (explore), <=120 chars.\n"
    '7) Nao inclua chaves alem de "insight" e "actionTag".'
)
USER_TEMPLATE = (
    "Gere o insight considerando os dados do usuario no ReStart.AI.\n\n"
    "metrics:\n{metrics}\n\n"
    "lastEvents:\n{events}\n\n"
    "profile:\n{profile}\n\n"
    "bestOpportunity (opcional):\n{bestopp}\n"
)

# ---------- OpenAI ----------
class AIServiceError(RuntimeError):
    def __init__(self, msg: str): super().__init__(msg); self.public_msg = msg

def _client() -> OpenAI:
    kw = {"api_key": OPENAI_API_KEY, "timeout": OPENAI_TIMEOUT}
    if OPENAI_BASE_URL: kw["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kw)

def _build_messages(req: InsightRequest) -> List[Dict[str, str]]:
    payload = USER_TEMPLATE.format(
        metrics=json.dumps(req.metrics.model_dump(mode="json"), ensure_ascii=False),
        events=json.dumps([e.model_dump(mode="json") for e in req.lastEvents], ensure_ascii=False),
        profile=json.dumps(req.profile.model_dump(mode="json"), ensure_ascii=False),
        bestopp=json.dumps(req.bestOpportunity.model_dump(mode="json") if req.bestOpportunity else None, ensure_ascii=False),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": payload}]

_ALLOWED = {"apply", "explore", "study"}

def _extract_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m: raise
        return json.loads(m.group(0))

def _validate_out(obj: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(obj, dict): raise AIServiceError("Formato inválido")
    k = set(obj.keys())
    if k != {"insight", "actionTag"}:
        if "insight" in obj and "actionTag" in obj:
            obj = {"insight": obj["insight"], "actionTag": obj["actionTag"]}
        else:
            raise AIServiceError("Formato inválido")
    insight = str(obj["insight"]).replace("\n", " ").strip()
    tag = str(obj["actionTag"]).strip()
    if not (0 < len(insight) <= 120): raise AIServiceError("Insight fora do limite")
    if tag not in _ALLOWED: raise AIServiceError("actionTag inválida")
    return {"insight": insight, "actionTag": tag}

async def _generate(req: InsightRequest) -> Dict[str, str]:
    if not OPENAI_API_KEY: raise AIServiceError("OPENAI_API_KEY ausente")
    if cb_is_open(): raise AIServiceError("Circuito aberto")
    msgs = _build_messages(req)
    cli = _client()
    for _ in range(2):
        try:
            r = await asyncio.to_thread(
                cli.chat.completions.create,
                model=MODEL, messages=msgs,
                temperature=OPENAI_TEMPERATURE, max_tokens=OPENAI_MAX_TOKENS,
            )
            content = r.choices[0].message.content or ""
            out = _validate_out(_extract_json(content))
            cb_on_success()
            return out
        except (OpenAIError, TimeoutError, json.JSONDecodeError, AIServiceError):
            cb_on_failure()
            msgs += [{"role": "system", "content": 'Responda apenas JSON com "insight" e "actionTag". Insight <=120.'}]
            continue
    raise AIServiceError("Falha ao gerar insight")

# ---------- Fallback inteligente ----------
def _choose_tag(req: InsightRequest) -> str:
    has_gap = bool(req.profile.gaps) or (req.bestOpportunity and req.bestOpportunity.missingSkill)
    if has_gap: return "study"
    many_views_no_apply = req.metrics.jobsViewedToday >= 3 and req.metrics.applyClicksToday == 0
    saw_best = any(e.type == "view_best_opportunity" for e in req.lastEvents)
    if many_views_no_apply or saw_best: return "apply"
    return "explore"

def _phrase(tag: str, req: InsightRequest) -> str:
    role = (req.bestOpportunity.role if req.bestOpportunity and req.bestOpportunity.role else (req.profile.roles[0] if req.profile.roles else None))
    city = (req.bestOpportunity.city if req.bestOpportunity and req.bestOpportunity.city else req.profile.city)
    tgt = f"{role} em {city}" if role and city else (role or city or "na sua área")
    if tag == "apply":
        s = f"Aplique para {tgt}; você já analisou boas opções hoje."
    elif tag == "study":
        gap = (req.bestOpportunity.missingSkill if req.bestOpportunity and req.bestOpportunity.missingSkill else (req.profile.gaps[0] if req.profile.gaps else "uma skill chave"))
        s = f"Estude {gap} para elevar seu match e abrir mais vagas em {tgt}."
    else:
        s = f"Explore novas vagas {tgt} e ajuste filtros para melhorar o match."
    return s[:120]

def _fallback(req: InsightRequest) -> InsightResponse:
    tag = _choose_tag(req)
    return InsightResponse(insight=_phrase(tag, req), actionTag=tag)  # type: ignore[arg-type]

# ---------- Endpoints ----------
@app.post("/insight", response_model=InsightResponse)
async def post_insight(
    req: InsightRequest,
    response: Response,
    _auth: None = Depends(assert_internal_key),
    request_id: str = Depends(get_request_id),
):
    response.headers["X-Request-Id"] = request_id
    check_rate(req.userId)

    cache_key = _cache_key(req)
    cached = cache_get(cache_key)
    if cached:
        log.info(f"hit cache rid={request_id} uid={req.userId} tag={cached['actionTag']}")
        return InsightResponse(**cached)

    t0 = time.perf_counter()
    try:
        out = await _generate(req)
        took = int((time.perf_counter() - t0) * 1000)
        cache_put(cache_key, out)
        log.info(f"ok rid={request_id} uid={req.userId} tag={out['actionTag']} ms={took}")
        return InsightResponse(**out)
    except AIServiceError as e:
        out = _fallback(req).model_dump()
        took = int((time.perf_counter() - t0) * 1000)
        cache_put(cache_key, out)
        log.warning(f"fallback rid={request_id} uid={req.userId} reason='{e.public_msg}' tag={out['actionTag']} ms={took}")
        return InsightResponse(**out)
    except HTTPException:
        raise
    except Exception as e:
        out = _fallback(req).model_dump()
        took = int((time.perf_counter() - t0) * 1000)
        cache_put(cache_key, out)
        log.error(f"error rid={request_id} uid={req.userId} err={type(e).__name__} tag={out['actionTag']} ms={took}")
        return InsightResponse(**out)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL}

@app.get("/readyz")
async def readyz():
    ready = bool(OPENAI_API_KEY) and not cb_is_open()
    return {"ready": ready, "circuit_open": cb_is_open()}
