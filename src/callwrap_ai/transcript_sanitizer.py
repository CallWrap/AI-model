"""표시·요약용 전사 비속어 마스킹.

감정 분석은 원본 전사를 사용한다. 이 모듈은 LLM을 호출하지 않으며 원본 객체도 수정하지 않는다.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from callwrap_ai.schemas import Transcript

MASK = "[비속어]"
PATTERN_FILE = Path(__file__).resolve().parent / "config" / "profanity_patterns.json"


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern[str], ...]:
    data = json.loads(PATTERN_FILE.read_text(encoding="utf-8"))
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in data["patterns"])


def sanitize_text(text: str) -> str:
    """비속어 패턴을 ``[비속어]``로 바꾼다."""
    sanitized = text
    for pattern in _patterns():
        sanitized = pattern.sub(MASK, sanitized)
    return sanitized


def sanitize_transcript(transcript: Transcript) -> Transcript:
    """발화 메타데이터를 보존한 마스킹 전사 복사본을 반환한다."""
    utterances = [
        utterance.model_copy(update={"text": sanitize_text(utterance.text)})
        for utterance in transcript.utterances
    ]
    return transcript.model_copy(update={"utterances": utterances})
