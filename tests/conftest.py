from pathlib import Path

import pytest

from citedesk.config import Settings
from citedesk.llm import FakeLLM, LLMResult
from citedesk.ontology import Ontology
from citedesk.pipeline import Pipeline
from citedesk.store import Store

DATA = Path(__file__).resolve().parents[1] / "data"


class SpyLLM(FakeLLM):
    """LLM 에 실제로 넘어간 프롬프트를 기록한다 (권한 밖 본문이 새는지 확인용)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.prompts.append(user)
        return super().complete(system, user)


@pytest.fixture
def onto() -> Ontology:
    return Ontology.load(DATA / "ontology.toml")


@pytest.fixture
def spy() -> SpyLLM:
    return SpyLLM()


@pytest.fixture
def pipeline(tmp_path, onto, spy) -> Pipeline:
    s = Settings(db_path=str(tmp_path / "t.db"), use_fake_llm=True)
    store = Store(s.db_path)
    store.ingest_dir(str(DATA / "corpus"), onto.level_of, s.max_chunk_chars)
    return Pipeline(store, spy, s, onto)
