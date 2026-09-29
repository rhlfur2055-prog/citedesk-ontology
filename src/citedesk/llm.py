import re
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


class LLM(Protocol):
    def complete(self, system: str, user: str) -> LLMResult: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int):
        import anthropic  # API 키 없이 테스트할 때는 import 하지 않도록 지연 로딩

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResult:
        t0 = time.perf_counter()
        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResult(text, r.usage.input_tokens, r.usage.output_tokens,
                         (time.perf_counter() - t0) * 1000)


class FakeLLM:
    """키 없이 돌리는 테스트/데모용. 첫 번째 근거 조각의 첫 문장을 그대로 인용한다."""

    def complete(self, system: str, user: str) -> LLMResult:
        m = re.search(r"\[1\]\s*(.+)", user)
        sentence = (m.group(1).split(". ")[0] if m else "").strip()
        return LLMResult(f"{sentence} [1]", len(user) // 4, len(sentence) // 4, 1.0)
