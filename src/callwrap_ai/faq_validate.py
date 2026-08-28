"""Gemma 3 12B로 EXAONE FAQ 초안을 원본 사례에 대조 검증한다."""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from callwrap_ai.transcript_sanitizer import sanitize_text

ROOT = Path(__file__).resolve().parents[2]
DEV_PROMPT = ROOT / "eval/prompts/faq_validate_v1.md"
PACKAGED_PROMPT = Path(__file__).resolve().parent / "prompts" / "faq_validate_v1.md"
DEFAULT_MODEL = "gemma3:12b"
API = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
TIMEOUT = int(os.environ.get("CALLWRAP_LLM_TIMEOUT", "180"))


def _prompt() -> str:
    env_override = os.environ.get("CALLWRAP_FAQ_VALIDATOR_PROMPT")
    candidates = [Path(env_override)] if env_override else [DEV_PROMPT, PACKAGED_PROMPT]
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8").split("\n## 프롬프트\n", 1)[1].strip()
        except (OSError, IndexError):
            continue
    raise FileNotFoundError("FAQ 검증 프롬프트 파일을 찾지 못했습니다")


def _ask(prompt: str, model: str) -> str | None:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 42},
        }
    ).encode()
    request = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    try:
        response = json.loads(urllib.request.urlopen(request, timeout=TIMEOUT).read())["response"]
        return response if isinstance(response, str) else None
    except (
        urllib.error.URLError,
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None


def _format_cases(cases: list[dict]) -> str:
    return "\n".join(
        f"[{index}] 문의: {sanitize_text(str(case.get('inquiry') or '(문의 내용 없음)'))}\n"
        f"    처리: {sanitize_text(str(case.get('resolution') or '(처리 내용 없음)'))}"
        for index, case in enumerate(cases)
    )


def _parse(raw: str, num_cases: int, max_items: int) -> list[dict] | None:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return None
    data = payload["items"]

    result: list[dict] = []
    for item in data[:max_items]:
        if not isinstance(item, dict):
            return None
        question, answer = item.get("question"), item.get("answer")
        if not (
            isinstance(question, str)
            and question.strip()
            and isinstance(answer, str)
            and answer.strip()
        ):
            return None
        raw_indices = item.get("source_indices")
        if not isinstance(raw_indices, list):
            return None
        indices = [
            index
            for index in raw_indices
            if isinstance(index, int) and 0 <= index < num_cases
        ]
        if not indices:
            return None
        result.append(
            {
                "question": sanitize_text(question.strip()),
                "answer": sanitize_text(answer.strip()),
                "source_indices": indices,
            }
        )
    return result


def validate_faq(
    subcategory: str,
    cases: list[dict],
    candidate: list[dict],
    model: str | None = None,
) -> list[dict] | None:
    """검증된 FAQ 목록. 검증 실패는 None, 전부 부적합은 빈 목록을 반환한다."""
    if not cases or not candidate:
        return []
    try:
        prompt = _prompt().format(
            subcategory=subcategory,
            cases=_format_cases(cases),
            candidate=json.dumps(candidate, ensure_ascii=False),
        )
    except (OSError, IndexError, KeyError, ValueError):
        return None
    raw = _ask(prompt, model or os.environ.get("CALLWRAP_VALIDATOR_MODEL", DEFAULT_MODEL))
    if not raw:
        return None
    return _parse(raw, len(cases), len(candidate))
