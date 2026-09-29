.PHONY: run test lint eval

run:            ## 서버 + 3D 그래프 (http://127.0.0.1:8000), 키 없이 FakeLLM 으로 뜬다
	uv sync
	CITEDESK_USE_FAKE_LLM=true uv run citedesk ingest data/corpus
	CITEDESK_USE_FAKE_LLM=true uv run citedesk serve

test:
	uv run pytest -q

lint:
	uv run ruff check .

eval:           ## 온톨로지 켬/끔 비교 (LLM 불필요, 결정적)
	uv run citedesk eval data/golden.jsonl data/golden_paraphrase.jsonl data/golden_access.jsonl --retrieval-only
	uv run citedesk eval data/golden.jsonl data/golden_paraphrase.jsonl data/golden_access.jsonl --retrieval-only --no-ontology
