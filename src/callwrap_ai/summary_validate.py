"""Gemma 3 12B로 EXAONE 요약·FAQ 필요 여부를 원문 대조 검증한다."""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from callwrap_ai.schemas import Judgment, Summary, Transcript
from callwrap_ai.transcript_sanitizer import sanitize_text

ROOT = Path(__file__).resolve().parents[2]
DEV_PROMPT = ROOT / "eval/prompts/summary_validate_v1.md"
PACKAGED_PROMPT = Path(__file__).resolve().parent / "prompts" / "summary_validate_v1.md"
DEFAULT_MODEL = "gemma3:12b"
API = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
TIMEOUT = int(os.environ.get("CALLWRAP_LLM_TIMEOUT", "120"))


def _prompt() -> str:
    env_override = os.environ.get("CALLWRAP_SUMMARY_VALIDATOR_PROMPT")
    candidates = [Path(env_override)] if env_override else [DEV_PROMPT, PACKAGED_PROMPT]
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8").split("\n## 프롬프트\n", 1)[1].strip()
        except (OSError, IndexError):
            continue
    raise FileNotFoundError("요약 검증 프롬프트 파일을 찾지 못했습니다")


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


def _parse_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate(
    transcript: Transcript,
    candidate: tuple[Judgment[Summary], Judgment[bool]],
    model: str | None = None,
) -> tuple[Judgment[Summary], Judgment[bool]] | None:
    """정확하면 초안을 유지하고, 오류가 있으면 Gemma 교정본을 반환한다."""
    summary, faq = candidate
    dialog = "\n".join(
        f"[{utterance.index}] {utterance.speaker.value}: {utterance.text}"
        for utterance in transcript.utterances
    )
    candidate_json = json.dumps(
        {
            "title": summary.value.title,
            "inquiry": summary.value.inquiry,
            "resolution": summary.value.resolution,
            "needs_faq": faq.value,
            "evidence_indices": summary.evidence_indices,
        },
        ensure_ascii=False,
    )
    try:
        prompt = _prompt().format(dialog=dialog, candidate=candidate_json)
    except (OSError, IndexError, KeyError, ValueError):
        return None
    raw = _ask(prompt, model or os.environ.get("CALLWRAP_VALIDATOR_MODEL", DEFAULT_MODEL))
    if not raw:
        return None
    data = _parse_json(raw)
    if data is None or not isinstance(data.get("valid"), bool):
        return None
    if data["valid"]:
        return candidate

    fields = ("title", "inquiry", "resolution")
    if not all(isinstance(data.get(field), str) and data[field].strip() for field in fields):
        return None
    if not isinstance(data.get("needs_faq"), bool):
        return None

    valid_indices = {utterance.index for utterance in transcript.utterances}
    raw_evidence = data.get("evidence_indices")
    evidence = (
        [index for index in raw_evidence if isinstance(index, int) and index in valid_indices][:3]
        if isinstance(raw_evidence, list)
        else []
    )
    corrected = Summary(
        **{field: sanitize_text(data[field].strip()) for field in fields}
    )
    return (
        Judgment(value=corrected, confidence=0.0, evidence_indices=evidence),
        Judgment(value=data["needs_faq"], confidence=0.0, evidence_indices=evidence),
    )
