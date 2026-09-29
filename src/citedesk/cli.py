import argparse

import uvicorn

from .api import build_pipeline
from .config import get_settings
from .evaluate import load_golden, run_eval
from .ontology import Ontology
from .store import Store


def main() -> None:
    ap = argparse.ArgumentParser(prog="citedesk")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="폴더의 .md 문서를 색인한다 (접근 등급은 온톨로지에서 온다)")
    p_ing.add_argument("directory")

    p_ask = sub.add_parser("ask", help="질문 하나를 던진다")
    p_ask.add_argument("question")
    p_ask.add_argument("--role", default="employee")
    p_ask.add_argument("--no-ontology", action="store_true", help="온톨로지 검색 확장을 끈다 (비교용)")

    p_ev = sub.add_parser("eval", help="정답 세트로 평가한다")
    p_ev.add_argument("golden", nargs="+")
    p_ev.add_argument("--retrieval-only", action="store_true", help="LLM 없이 검색만 평가")
    p_ev.add_argument("--no-ontology", action="store_true", help="온톨로지 검색 확장을 끈다 (비교용)")

    p_gr = sub.add_parser("graph", help="온톨로지를 mermaid 로 출력한다")
    p_gr.add_argument("--json", action="store_true", help="3D 뷰어가 쓰는 JSON 으로 출력")

    p_srv = sub.add_parser("serve", help="API 서버 + 3D 그래프 뷰어 실행")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)

    a = ap.parse_args()
    s = get_settings()

    if a.cmd == "ingest":
        onto = Ontology.load(s.ontology_path)
        files, chunks = Store(s.db_path).ingest_dir(a.directory, onto.level_of, s.max_chunk_chars)
        print(f"{files}개 문서 -> {chunks}개 조각 색인 완료 ({s.db_path})")
    elif a.cmd == "ask":
        r = build_pipeline(s).ask(a.question, a.role, use_ontology=not a.no_ontology)
        print(r.answer)
        if r.concepts:
            print("  개념: " + ", ".join(f"{c['label']}({c['doc']})" for c in r.concepts))
        for c in r.citations:
            print(f"  [{c['n']}] {c['source']} > {c['heading']}")
    elif a.cmd == "eval":
        p = build_pipeline(s) if not a.retrieval_only else build_pipeline(
            s.model_copy(update={"use_fake_llm": True}))
        for path in a.golden:
            print(f"### {path}  (온톨로지 {'끔' if a.no_ontology else '켬'})\n")
            print(run_eval(p, load_golden(path), use_llm=not a.retrieval_only,
                           use_ontology=not a.no_ontology).to_markdown() + "\n")
    elif a.cmd == "graph":
        onto = Ontology.load(s.ontology_path)
        print(onto.to_json() if a.json else onto.to_mermaid())
    elif a.cmd == "serve":
        uvicorn.run("citedesk.api:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
