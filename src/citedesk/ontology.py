import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

RESTRICTED = "restricted"  # 등록되지 않은 문서의 기본 등급: 모르면 잠근다


class OntologyError(ValueError):
    pass


class UnknownRole(KeyError):
    pass


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    aliases: tuple[str, ...]
    terms: tuple[str, ...]
    parent: str | None
    governed_by: str | None


class Ontology:
    def __init__(self, concepts: dict[str, Concept], documents: dict[str, dict], roles: dict[str, dict]):
        self.concepts, self.documents, self.roles = concepts, documents, roles
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "Ontology":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        concepts = {
            cid: Concept(
                cid, c["label"], tuple(c.get("aliases", ())), tuple(c.get("terms", ())),
                c.get("parent"), c.get("governed_by"),
            )
            for cid, c in raw.get("concepts", {}).items()
        }
        return cls(concepts, raw.get("documents", {}), raw.get("roles", {}))

    def _validate(self) -> None:
        for c in self.concepts.values():
            if c.parent is not None and c.parent not in self.concepts:
                raise OntologyError(f"{c.id}: 없는 부모 개념 {c.parent!r}")
            if c.governed_by is not None and c.governed_by not in self.documents:
                raise OntologyError(f"{c.id}: 등록되지 않은 문서 {c.governed_by!r}")
            seen, cur = {c.id}, c.parent  # is_a 순환 검사
            while cur is not None:
                if cur in seen:
                    raise OntologyError(f"{c.id}: is_a 순환")
                seen.add(cur)
                cur = self.concepts[cur].parent
            if not self.governing_doc(c.id):
                raise OntologyError(f"{c.id}: 규율하는 문서를 찾을 수 없음 (본인·조상 모두 governed_by 없음)")

    # ---- 그래프 질의 ----
    def governing_doc(self, cid: str) -> str | None:
        cur: str | None = cid
        while cur is not None:
            c = self.concepts[cur]
            if c.governed_by:
                return c.governed_by
            cur = c.parent
        return None

    def descendants(self, cid: str) -> list[str]:
        out = [cid]
        for c in self.concepts.values():
            if c.parent == cid:
                out += self.descendants(c.id)
        return out

    def link(self, question: str) -> list[Concept]:
        """질문에 별칭이 들어 있는 개념을 찾는다. 대소문자는 무시한다."""
        q = question.lower()
        return [c for c in self.concepts.values() if any(a.lower() in q for a in (c.label, *c.aliases))]

    def expand(self, question: str) -> tuple[list[Concept], list[str]]:
        """연결된 개념과, 그 개념(및 하위 개념)이 문서에서 쓰는 말 목록."""
        linked = self.link(question)
        terms: dict[str, None] = {}
        for c in linked:
            for did in self.descendants(c.id):
                for t in self.concepts[did].terms:
                    terms[t] = None
        return linked, list(terms)

    # ---- 접근 제어 ----
    def allowed_levels(self, role: str) -> tuple[str, ...]:
        if role not in self.roles:
            raise UnknownRole(role)
        return tuple(self.roles[role]["levels"])

    def level_of(self, source: str) -> str:
        return self.documents.get(source, {}).get("level", RESTRICTED)

    # ---- 내보내기 ----
    def visible(self, cid: str, levels: tuple[str, ...]) -> bool:
        """개념을 규율하는 문서를 이 역할이 볼 수 있어야 개념도 보인다 (이름만으로도 존재가 드러나므로)."""
        doc = self.governing_doc(cid)
        return doc is not None and self.level_of(doc) in levels

    def to_graph(self, levels: tuple[str, ...] | None = None) -> dict:
        """3D 뷰어와 mermaid 가 같이 쓰는 노드/간선 목록. levels 를 주면 볼 수 없는 문서·개념은 뺀다."""
        show = (lambda cid: True) if levels is None else (lambda cid: self.visible(cid, levels))
        nodes, links = [], []
        for c in self.concepts.values():
            if not show(c.id):
                continue
            nodes.append({"id": c.id, "label": c.label, "kind": "concept"})
            if c.parent:
                links.append({"source": c.id, "target": c.parent, "rel": "is_a"})
            if c.governed_by:
                links.append({"source": c.id, "target": c.governed_by, "rel": "governed_by"})
        owners: set[str] = set()
        for name, d in self.documents.items():
            if levels is not None and d["level"] not in levels:
                continue
            nodes.append({"id": name, "label": name, "kind": "document", "level": d["level"]})
            owners.add(d["owner"])
            links.append({"source": name, "target": "team:" + d["owner"], "rel": "owned_by"})
        nodes += [{"id": "team:" + o, "label": o, "kind": "team"} for o in sorted(owners)]
        return {"nodes": nodes, "links": links}

    def to_mermaid(self) -> str:
        g = self.to_graph()
        safe = {n["id"]: f"n{i}" for i, n in enumerate(g["nodes"])}
        lines = ["graph LR"]
        for n in g["nodes"]:
            lb = n["label"]
            lines.append(f'  {safe[n["id"]]}["{lb}"]' if n["kind"] == "concept"
                         else f'  {safe[n["id"]]}[["{lb}"]]')
        for e in g["links"]:
            lines.append(f'  {safe[e["source"]]} -->|{e["rel"]}| {safe[e["target"]]}')
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.to_graph(), ensure_ascii=False)
