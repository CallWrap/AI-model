"""LLM 백엔드 추상화. 소유: 김성진. 이재혁(F3)도 사용 — 시그니처 변경 시 반드시 공유.

구현체는 anthropic.py(개발용)와 ollama.py(온프렘 검증용)로 추가 예정.
.env의 LLM_BACKEND=anthropic|ollama 로 스위칭한다. (명세 4.6)
"""

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete(self, prompt: str, schema: type[T]) -> T:
        """prompt를 보내고 schema에 맞는 구조화 출력을 받는다.

        구현체는 파싱 실패 시 1회 재시도하고, 그래도 실패하면 예외를 던진다.
        부분 실패 처리(failures 기록)는 호출자(pipeline)의 책임이다.
        """
        ...
