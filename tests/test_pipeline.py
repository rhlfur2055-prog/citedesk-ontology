from fastapi.testclient import TestClient

from citedesk.api import create_app
from citedesk.chunker import chunk_markdown


def test_ontology_finds_paraphrase_that_plain_search_misses(pipeline):
    q = "패스워드는 얼마나 길어야 하나요?"
    assert pipeline.retrieve(q, "employee", use_ontology=False).hits == []
    with_onto = pipeline.retrieve(q, "employee", use_ontology=True)
    assert with_onto.hits[0].source == "security.md"
    assert [c.id for c in with_onto.concepts] == ["credential"]


def test_answer_has_valid_citation(pipeline):
    a = pipeline.ask("연차는 며칠 전까지 신청해야 하나요?")
    assert a.grounded and a.citations and a.citations[0]["source"] == "vacation.md"


def test_no_hits_means_no_llm_call_and_zero_cost(pipeline, spy):
    a = pipeline.ask("회사 주식 매수선택권은 어떻게 행사하나요?")
    assert a.refused and not a.denied and a.cost_usd == 0 and spy.prompts == []


def test_stats_count_queries_and_denials(pipeline):
    pipeline.ask("연차는 며칠 전까지 신청해야 하나요?")
    pipeline.ask("주니어 개발자 연봉 밴드가 얼마인가요?")
    st = pipeline.store.stats()
    assert st["queries"] == 2 and st["denied"] == 1


def test_chunker_keeps_heading_with_each_piece():
    cs = chunk_markdown("a.md", "# T\n\n## 가\n내용1\n\n## 나\n내용2")
    assert [(c.heading, c.text) for c in cs] == [("가", "내용1"), ("나", "내용2")]


def test_api_ask_and_role_header(pipeline):
    c = TestClient(create_app(pipeline))
    r = c.post("/ask", json={"question": "주니어 개발자 연봉 밴드가 얼마인가요?"}, headers={"X-Role": "hr"})
    assert r.status_code == 200 and not r.json()["denied"]
    assert c.post("/ask", json={"question": "연차 신청 방법"}, headers={"X-Role": "intern"}).status_code == 403
    g = c.get("/graph.json").json()
    assert {n["kind"] for n in g["nodes"]} == {"concept", "document", "team"}
