import json
from dataclasses import dataclass
from pathlib import Path

from .pipeline import Pipeline


@dataclass
class EvalReport:
    n: int
    n_answerable: int
    n_unanswerable: int
    n_denied: int
    hit_at_k: float          # 정답 문서가 검색 상위 k 안에 있는 비율 (LLM 불필요, 결정적)
    top1: float              # 정답 문서가 1등인 비율
    denied_ok: float | None  # 권한 밖 질문을 차단한 비율 (LLM 불필요, 결정적)
    answer_correct: float | None    # 답에 기대 키워드가 모두 들어있는 비율 (LLM 필요)
    citation_source_ok: float | None  # 인용에 정답 문서가 포함된 비율
    refusal_ok: float | None  # 답할 수 없는 질문을 거절한 비율
    failures: list[str]

    def to_markdown(self) -> str:
        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.0%}"

        rows = [
            ("검색 hit@k (정답 문서가 상위 k 안)", pct(self.hit_at_k)),
            ("검색 top-1", pct(self.top1)),
            ("권한 밖 질문 차단률", pct(self.denied_ok)),
            ("답변 정확도 (기대 키워드 전부 포함)", pct(self.answer_correct)),
            ("인용에 정답 문서 포함", pct(self.citation_source_ok)),
            ("답 없는 질문 거절률", pct(self.refusal_ok)),
        ]
        head = (f"질문 {self.n}개 (답 있음 {self.n_answerable} / 답 없음 {self.n_unanswerable}"
                f" / 권한 밖 {self.n_denied})\n\n")
        body = "| 지표 | 결과 |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
        if self.failures:
            body += "\n\n실패 사례:\n" + "\n".join(f"- {f}" for f in self.failures)
        return head + body


def load_golden(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def run_eval(p: Pipeline, golden: list[dict], use_llm: bool, use_ontology: bool = True) -> EvalReport:
    """골든 한 줄: question, source(정답 문서 또는 null), keywords, [role=employee], [expect="denied"]"""
    denied_q = [g for g in golden if g.get("expect") == "denied"]
    ans_q = [g for g in golden if g.get("expect") != "denied" and g["source"]]
    no_q = [g for g in golden if g.get("expect") != "denied" and not g["source"]]
    hit = top1 = correct = cite_ok = refused_ok = denied_ok = 0
    failures: list[str] = []

    for g in ans_q:
        role = g.get("role", "employee")
        hits = p.retrieve(g["question"], role, use_ontology).hits
        srcs = [h.source for h in hits]
        if g["source"] in srcs:
            hit += 1
        else:
            failures.append(f"검색 실패: {g['question']!r} (기대 {g['source']}, 실제 {srcs})")
        if srcs[:1] == [g["source"]]:
            top1 += 1
        if use_llm:
            a = p.ask(g["question"], role, use_ontology)
            if not a.refused and all(k in a.answer for k in g["keywords"]):
                correct += 1
            else:
                failures.append(f"답변 오류: {g['question']!r} -> {a.answer[:60]!r}")
            if any(c["source"] == g["source"] for c in a.citations):
                cite_ok += 1

    for g in denied_q:
        if p.retrieve(g["question"], g.get("role", "employee"), use_ontology).denied:
            denied_ok += 1
        else:
            failures.append(f"차단 실패: {g['question']!r} (role={g.get('role', 'employee')})")

    for g in no_q:
        if not use_llm:
            continue  # 거절 여부는 LLM 이 정한다 -> 검색 전용 모드에서는 재지 않는다
        a = p.ask(g["question"], g.get("role", "employee"), use_ontology)
        if a.refused:
            refused_ok += 1
        else:
            failures.append(f"거절 실패(환각 의심): {g['question']!r} -> {a.answer[:60]!r}")

    na, nn, nd = max(len(ans_q), 1), max(len(no_q), 1), max(len(denied_q), 1)
    return EvalReport(
        n=len(golden), n_answerable=len(ans_q), n_unanswerable=len(no_q), n_denied=len(denied_q),
        hit_at_k=hit / na, top1=top1 / na,
        denied_ok=denied_ok / nd if denied_q else None,
        answer_correct=correct / na if use_llm else None,
        citation_source_ok=cite_ok / na if use_llm else None,
        refusal_ok=refused_ok / nn if use_llm and no_q else None,
        failures=failures,
    )
