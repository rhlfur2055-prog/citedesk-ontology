import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk, chunk_markdown

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    source UNINDEXED, heading UNINDEXED, access UNINDEXED, text, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY, ts REAL, role TEXT, question TEXT, concepts TEXT, hits INTEGER,
    denied INTEGER, refused INTEGER, grounded INTEGER, tokens_in INTEGER, tokens_out INTEGER,
    latency_ms REAL, cost_usd REAL
);
"""


@dataclass(frozen=True)
class Hit:
    source: str
    heading: str
    text: str
    score: float  # 클수록 관련 있음


def _fts_query(question: str, extra_terms: Sequence[str] = ()) -> str | None:
    # 한글은 조사·어미가 붙어 단어 통째로는 안 맞는다("한도가" vs "한도까지").
    # 단어를 3글자 조각으로 잘라 OR 로 묶으면 겹치는 조각이 많은 문서가 BM25 로 위로 올라온다.
    grams: list[str] = []
    for tok in re.findall(r"\w+", question):
        grams += [tok[i:i + 3] for i in range(len(tok) - 2)]
    grams += [t for t in extra_terms if len(t) >= 3]  # 온톨로지가 붙여 준 '문서 쪽 표현'은 통째로 구문 검색
    if not grams:
        return None
    return " OR ".join('"' + g.replace('"', '""') + '"' for g in dict.fromkeys(grams))


class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)

    def reset_chunks(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def add_chunks(self, chunks: list[Chunk], level_of: Callable[[str], str]) -> int:
        self.conn.executemany(
            "INSERT INTO chunks(source, heading, access, text) VALUES (?,?,?,?)",
            [(c.source, c.heading, level_of(c.source), c.text) for c in chunks],
        )
        self.conn.commit()
        return len(chunks)

    def ingest_dir(self, directory: str, level_of: Callable[[str], str], max_chars: int = 500) -> tuple[int, int]:
        self.reset_chunks()  # 같은 폴더를 다시 넣어도 중복이 쌓이지 않게 한다
        files = sorted(Path(directory).glob("**/*.md"))
        total = 0
        for f in files:
            total += self.add_chunks(chunk_markdown(f.name, f.read_text(encoding="utf-8"), max_chars), level_of)
        return len(files), total

    def search(self, question: str, k: int, levels: Sequence[str], extra_terms: Sequence[str] = (),
               *, allowed: bool = True) -> list[Hit]:
        """levels 에 속한 조각만(allowed=True) 또는 속하지 않은 조각만(allowed=False) 찾는다.
        접근 필터는 SQL 안에서 걸리므로 권한 밖 본문은 파이썬 메모리에 올라오지도 않는다."""
        q = _fts_query(question, extra_terms)
        if q is None:
            return []
        if allowed and not levels:
            return []
        marks = ",".join("?" * len(levels)) or "''"
        op = "IN" if allowed else "NOT IN"
        rows = self.conn.execute(
            f"SELECT source, heading, text, bm25(chunks) FROM chunks WHERE chunks MATCH ? "
            f"AND access {op} ({marks}) ORDER BY bm25(chunks) LIMIT ?",
            (q, *levels, k),
        ).fetchall()
        # SQLite 의 bm25 는 작을수록(더 음수일수록) 좋다 -> 부호를 뒤집어 '클수록 좋음'으로 통일
        return [Hit(s, h, t, -sc) for s, h, t, sc in rows]

    def log_query(self, **f) -> None:
        self.conn.execute(
            "INSERT INTO queries(ts, role, question, concepts, hits, denied, refused, grounded,"
            " tokens_in, tokens_out, latency_ms, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), f["role"], f["question"], f["concepts"], f["hits"], int(f["denied"]),
             int(f["refused"]), int(f["grounded"]), f["tokens_in"], f["tokens_out"],
             f["latency_ms"], f["cost_usd"]),
        )
        self.conn.commit()

    def stats(self) -> dict:
        r = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(refused),0), COALESCE(SUM(denied),0), COALESCE(SUM(grounded),0), "
            "COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
            "COALESCE(AVG(latency_ms),0), COALESCE(SUM(cost_usd),0) FROM queries"
        ).fetchone()
        n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "chunks": n_chunks, "queries": r[0], "refused": r[1], "denied": r[2], "grounded": r[3],
            "tokens_in": r[4], "tokens_out": r[5], "avg_latency_ms": round(r[6], 1),
            "cost_usd": round(r[7], 6),
        }

    def audit(self, limit: int = 20) -> list[dict]:
        cur = self.conn.execute(
            "SELECT ts, role, question, concepts, hits, denied FROM queries ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(zip(("ts", "role", "question", "concepts", "hits", "denied"), row)) for row in cur]
