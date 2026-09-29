import pytest

from citedesk.ontology import RESTRICTED, Concept, Ontology, OntologyError, UnknownRole


def test_link_finds_concepts_by_alias(onto):
    ids = {c.id for c in onto.link("점심값은 얼마까지 내주나요?")}
    assert "meal_allowance" in ids


def test_expand_includes_child_terms(onto):
    _, terms = onto.expand("휴가는 어떻게 쓰나요?")  # 상위 개념 '휴가' -> 하위 연차·병가의 문서 표현까지
    assert "병가는" in terms and "연차휴가" in terms


def test_governing_doc_is_inherited_from_parent(onto):
    assert onto.governing_doc("sick_leave") == "vacation.md"


def test_unknown_document_is_restricted(onto):
    assert onto.level_of("not-registered.md") == RESTRICTED


def test_unknown_role_raises(onto):
    with pytest.raises(UnknownRole):
        onto.allowed_levels("intern")


def test_cycle_is_rejected():
    a = Concept("a", "A", (), (), "b", "d.md")
    b = Concept("b", "B", (), (), "a", None)
    with pytest.raises(OntologyError, match="순환"):
        Ontology({"a": a, "b": b}, {"d.md": {"level": "public", "owner": "x"}}, {})


def test_concept_without_any_governing_doc_is_rejected():
    with pytest.raises(OntologyError, match="규율하는 문서"):
        Ontology({"a": Concept("a", "A", (), (), None, None)}, {}, {})


def test_mermaid_mentions_every_relation_kind(onto):
    m = onto.to_mermaid()
    assert m.startswith("graph LR") and "is_a" in m and "governed_by" in m and "owned_by" in m
