"""F1 요약·F4 후속처리·FAQ필요 — EXAONE 1회 호출로 함께 판정한다.

기본 프롬프트는 평가에서 선택한 v6이다. 이후 프롬프트를 개선해도
`CALLWRAP_SUMMARY_PROMPT`만 바꾸면 analyze() 계약과 코드는 그대로 유지된다.

**confidence는 모델의 토큰 확률에서 뽑는다.** ollama `/api/generate`에 `logprobs: true`를
주면 생성한 토큰마다 logprob이 함께 온다. 그 값의 기하평균이 confidence다 — 분류 모델의
softmax와 같은 종류의 값이고, 모델에게 "얼마나 확신하냐"고 되묻는 self-report가 아니다
(명세 9.3이 금지한 건 후자다). 추가 호출이 없어 비용도 그대로다.

⚠️ **캘리브레이션된 정확도 확률이 아니다.** 토큰 확률은 "이 문장을 이렇게 쓸 확신"이지
"이 판정이 맞을 확률"이 아니다. 3-tier UI 임계값(90/60)에 물리기 전에 골든셋으로 보정이
필요하다 (명세 9.3, Step 4). ollama가 logprobs를 안 주는 구버전이면 0.0으로 떨어진다.
"""

import json
import math
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


def _ask(prompt: str, model: str) -> tuple[str, list[dict]] | None:
    """(응답 본문, 토큰별 logprob 목록)을 돌려준다. 실패하면 None.

    `logprobs`를 모르는 구버전 ollama는 이 필드를 무시하고 응답만 준다 — 그때는
    빈 리스트가 되어 confidence만 0.0으로 떨어지고 요약 자체는 정상 동작한다.
    """
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "logprobs": True,
                       "options": {"temperature": 0, "seed": 42}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    try:
        payload = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
        response = payload["response"]
        if not isinstance(response, str):
            return None
        logprobs = payload.get("logprobs")
        return response, logprobs if isinstance(logprobs, list) else []
    except (
        urllib.error.URLError,
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None


#: JSON 값의 위치를 찾는 패턴. 문자열 값은 이스케이프(\") 를 건너뛰며 잡는다.
_STRING_VALUE = '"{key}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"'
_BOOL_VALUE = '"{key}"\\s*:\\s*(true|false)'


def _token_spans(logprobs: list[dict]) -> list[tuple[int, int, float]]:
    """토큰 목록 -> [(시작 문자위치, 끝 문자위치, logprob)]. 형식이 다르면 빈 리스트.

    토큰을 이어붙인 문자열이 곧 응답 본문이라(ollama 실측 확인) 문자 위치로 구간을 잡을 수 있다.
    """
    spans: list[tuple[int, int, float]] = []
    pos = 0
    for item in logprobs:
        if not isinstance(item, dict):
            return []
        token, logprob = item.get("token"), item.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, (int, float)):
            return []  # 일부만 읽어서 반쪽짜리 확률을 만드느니 통째로 포기한다
        spans.append((pos, pos + len(token), float(logprob)))
        pos += len(token)
    return spans


def _confidence(spans, ranges: list[tuple[int, int]]) -> float | None:
    """주어진 문자 구간들에 걸친 토큰들의 기하평균 확률. 겹치는 토큰이 없으면 None."""
    picked = [lp for start, end in ranges
              for s, e, lp in spans if s < end and e > start]
    if not picked:
        return None
    return round(min(1.0, math.exp(sum(picked) / len(picked))), 4)


def _value_range(text: str, key: str, pattern: str) -> tuple[int, int] | None:
    match = re.search(pattern.format(key=key), text, re.S)
    return match.span(1) if match else None


def _parse(raw: str, logprobs: list[dict], transcript: Transcript):
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
    # 판정별로 근거가 된 토큰 구간이 다르므로 confidence도 따로 뽑는다 — 요약은 세 문장
    # 전체의 기하평균, 후속처리·FAQ는 각자 true/false 토큰 하나의 확률이다.
    # logprobs가 없으면(구버전 ollama 등) 전부 0.0 — 없는 값을 지어내지 않는다.
    spans = _token_spans(logprobs)
    text = "".join(item["token"] for item in logprobs) if spans else ""

    def ranges(pattern: str, *keys: str) -> list[tuple[int, int]]:
        found = (_value_range(text, key, pattern) for key in keys)
        return [r for r in found if r is not None]

    summary_confidence = _confidence(spans, ranges(_STRING_VALUE, *fields)) or 0.0
    followup_confidence = _confidence(spans, ranges(_BOOL_VALUE, "needs_followup")) or 0.0
    faq_confidence = _confidence(spans, ranges(_BOOL_VALUE, "needs_faq")) or 0.0

    return (
        Judgment(
            value=Summary(**{f: data[f].strip() for f in fields}),
            confidence=summary_confidence,
            evidence_indices=evidence,
        ),
        Judgment(
            value=data["needs_followup"],
            confidence=followup_confidence,
            evidence_indices=evidence,
        ),
        Judgment(
            value=data["needs_faq"],
            confidence=faq_confidence,
            evidence_indices=evidence,
        ),
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
    answer = _ask(prompt, selected_model)
    if answer is None or not answer[0].strip():
        return None
    result = _parse(*answer, transcript)
    if result is not None:
        return result

    answer = _ask(prompt + "\n\n반드시 JSON 하나만 출력하세요.", selected_model)
    if answer is None or not answer[0].strip():
        return None
    return _parse(*answer, transcript)
