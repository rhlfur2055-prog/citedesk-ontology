from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .llm import AnthropicLLM, FakeLLM
from .ontology import Ontology, UnknownRole
from .pipeline import Pipeline
from .store import Store

WEB = Path(__file__).resolve().parents[2] / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    use_ontology: bool = True


def build_pipeline(s: Settings) -> Pipeline:
    llm = FakeLLM() if s.use_fake_llm else AnthropicLLM(s.model, s.max_output_tokens)
    return Pipeline(Store(s.db_path), llm, s, Ontology.load(s.ontology_path))


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    app = FastAPI(title="citedesk", version="0.2.0")
    state: dict = {"p": pipeline}

    def get_pipeline() -> Pipeline:
        if state["p"] is None:  # 첫 요청 때 만든다 -> 키가 없어도 /health 는 뜬다
            state["p"] = build_pipeline(get_settings())
        return state["p"]

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/ask")
    def ask(req: AskRequest, x_role: str = Header(default="employee")) -> dict:
        # 데모용: 역할을 헤더로 받는다. 실제 배포에서는 로그인 토큰에서 꺼내야 한다.
        try:
            return get_pipeline().ask(req.question, x_role, req.use_ontology).__dict__
        except UnknownRole:
            raise HTTPException(403, f"알 수 없는 역할: {x_role}")

    @app.get("/stats")
    def stats() -> dict:
        return get_pipeline().store.stats()

    @app.get("/audit")
    def audit() -> list[dict]:
        return get_pipeline().store.audit()

    @app.get("/graph.json")
    def graph(x_role: str = Header(default="employee")) -> JSONResponse:
        onto = get_pipeline().onto
        try:
            return JSONResponse(onto.to_graph(onto.allowed_levels(x_role)))
        except UnknownRole:
            raise HTTPException(403, f"알 수 없는 역할: {x_role}")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    if WEB.is_dir():
        app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
    return app


app = create_app()
