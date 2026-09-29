import pytest

from citedesk.ontology import UnknownRole
from citedesk.pipeline import DENIED

Q = "주니어 개발자 연봉 밴드가 얼마인가요?"


def test_employee_is_denied_and_llm_never_called(pipeline, spy):
    a = pipeline.ask(Q, role="employee")
    assert a.denied and a.refused and a.answer == DENIED
    assert spy.prompts == []


def test_restricted_text_never_reaches_llm_for_any_low_role(pipeline, spy):
    for role in ("employee", "guest"):
        for q in (Q, "성과급은 언제 얼마나 지급되나요?", "연봉 밴드와 성과급 알려줘", "연차는 며칠 전 신청?"):
            pipeline.ask(q, role=role)
    assert all("4,000만원" not in p and "성과급은 연 1회" not in p for p in spy.prompts)


def test_hr_can_read_restricted(pipeline):
    a = pipeline.ask(Q, role="hr")
    assert not a.denied and any(c["source"] == "payroll.md" for c in a.citations)


def test_denied_answer_carries_no_document_text(pipeline):
    a = pipeline.ask(Q, role="guest")
    assert a.citations == [] and "4,000" not in a.answer


def test_denied_answer_does_not_name_restricted_concepts_or_docs(pipeline, onto):
    assert pipeline.ask(Q, role="employee").concepts == []
    assert [c["id"] for c in pipeline.ask(Q, role="hr").concepts] == ["salary_band"]
    names = {n["id"] for n in onto.to_graph(onto.allowed_levels("employee"))["nodes"]}
    assert "payroll.md" not in names and "salary_band" not in names
    assert "payroll.md" in {n["id"] for n in onto.to_graph(onto.allowed_levels("hr"))["nodes"]}


def test_unknown_role_is_rejected(pipeline):
    with pytest.raises(UnknownRole):
        pipeline.ask(Q, role="intern")


def test_audit_log_records_denied(pipeline):
    pipeline.ask(Q, role="employee")
    row = pipeline.store.audit(1)[0]
    assert row["role"] == "employee" and row["denied"] == 1 and "salary_band" in row["concepts"]
