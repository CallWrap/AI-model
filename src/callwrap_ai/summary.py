"""F1 요약·F4 후속처리·FAQ필요 — EXAONE 1회 호출로 함께 판정한다.

기본 프롬프트는 평가에서 선택한 v6이다. 이후 프롬프트를 개선해도
`CALLWRAP_SUMMARY_PROMPT`만 바꾸면 analyze() 계약과 코드는 그대로 유지된다.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from callwrap_ai.schemas import Judgment, Summary, Transcript

ROOT = Path(__file__).resolve().parents[2]
#: 로컬 개발/레포 체크아웃 상태에서 쓰는 경로 — 편집 소스는 여기.
DEV_PROMPT = ROOT / "eval/prompts/summary_v6.md"
#: 패키지 설치 상태(server-fastapi가 git 소스로 callwrap-ai를 uv sync할 때)에서 쓰는 경로.
# pyproject.toml의 [tool.hatch.build.targets.wheel]이 packages=["src/callwrap_ai"]만 포함해서
# eval/은 휠에 안 들어간다(실제로 빌드해서 확인함 — eval 관련 파일 0개). DEV_PROMPT만 보고 있으면
# 이 파일을 git 소스로 설치한 환경(예: server-fastapi Docker 이미지)에서 read_text가 조용히
# OSError를 던지고, 그게 _ask 호출 전에 잡혀서 "모델이 판단 못 함"과 똑같이 보인다 — 실제로는
# 프롬프트 파일을 못 찾은 것뿐인데 구분이 안 됐던 버그. 그래서 내용이 같은 사본을
# src/callwrap_ai/prompts/ 밑에 두고 순서대로 시도한다.
PACKAGED_PROMPT = Path(__file__).resolve().parent / "prompts" / "summary_v6.md"
DEFAULT_MODEL = "exaone3.5:7.8b"
API = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
TIMEOUT = int(os.environ.get("CALLWRAP_LLM_TIMEOUT", "120"))


def _prompt() -> str:
    env_override = os.environ.get("CALLWRAP_SUMMARY_PROMPT")
    candidates = [Path(env_override)] if env_override else [DEV_PROMPT, PACKAGED_PROMPT]
    last_error: Exception | None = None
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
            return text.split("\n## 프롬프트\n", 1)[1].strip()
        except OSError as e:
            last_error = e
            continue
    raise FileNotFoundError(f"요약 프롬프트 파일을 찾지 못했습니다 (시도한 경로: {candidates})") from last_error


def _ask(prompt: str, model: str) -> str | None:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": 42}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    try:
        response = json.loads(
            urllib.request.urlopen(req, timeout=TIMEOUT).read()
        )["response"]
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


def _parse(raw: str, transcript: Transcript):
    if not isinstance(raw, str):
        return None
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    fields = ("title", "inquiry", "resolution")
    if not all(isinstance(data.get(f), str) and data[f].strip() for f in fields):
        return None
    if not all(isinstance(data.get(f), bool) for f in ("needs_followup", "needs_faq")):
        return None

    valid = {u.index for u in transcript.utterances}
    raw_evidence = data.get("evidence_indices")
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    evidence = [i for i in raw_evidence
                if isinstance(i, int) and i in valid][:3]
    confidence = 0.0  # 캘리브레이션 전에는 LLM self-report 확률을 만들지 않는다.
    common = {"confidence": confidence, "evidence_indices": evidence}
    return (
        Judgment(
            value=Summary(**{f: data[f].strip() for f in fields}),
            **common,
        ),
        Judgment(value=data["needs_followup"], **common),
        Judgment(value=data["needs_faq"], **common),
    )


def summarize(transcript: Transcript, model: str | None = None):
    """Transcript -> (요약, 후속처리, FAQ필요). 실패하면 None."""
    dialog = "\n".join(
        f"[{u.index}] {u.speaker.value}: {u.text}" for u in transcript.utterances
    )
    try:
        prompt = _prompt().format(dialog=dialog)
    except (OSError, IndexError, KeyError, ValueError):
        return None
    selected_model = model or os.environ.get("CALLWRAP_LLM_MODEL", DEFAULT_MODEL)
    raw = _ask(prompt, selected_model)
    if not isinstance(raw, str) or not raw.strip():
        return None
    result = _parse(raw, transcript)
    if result is not None:
        return result

    raw = _ask(prompt + "\n\n반드시 JSON 하나만 출력하세요.", selected_model)
    return _parse(raw, transcript) if isinstance(raw, str) and raw.strip() else None
