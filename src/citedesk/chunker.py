import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    text: str


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        if buf and len(buf) + len(para) + 2 > limit:
            parts.append(buf)
            buf = ""
        while len(para) > limit:  # 문단 하나가 한도를 넘으면 문장 경계에서 자른다
            cut = max(para.rfind(". ", 0, limit), para.rfind("다. ", 0, limit))
            cut = cut + 2 if cut > 0 else limit
            parts.append(para[:cut].strip())
            para = para[cut:]
        buf = f"{buf}\n\n{para}".strip() if buf else para
    if buf.strip():
        parts.append(buf)
    return parts


def chunk_markdown(source: str, text: str, max_chars: int = 500) -> list[Chunk]:
    """제목 단위로 나누고, 긴 섹션은 문단 단위로 다시 나눈다. 각 조각은 자기 제목을 들고 다닌다."""
    chunks: list[Chunk] = []
    heading = source
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            for piece in _split_long(content, max_chars):
                chunks.append(Chunk(source, heading, piece))
        body.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
        else:
            body.append(line)
    flush()
    return chunks
