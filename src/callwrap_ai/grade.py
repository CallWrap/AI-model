"""F3 상담등급 — 발화별 감정 점수를 2축 규칙으로 칭찬/악성/일반에 매핑한다.

온디바이스 LLM(ollama)에 민원인 발화의 감정을 매기게 하고, 그 점수를 규칙에 넣는다.
LLM 호출은 **통화당 1회**다 (발화마다 부르면 7배가 된다).

판정 규칙과 임계값은 `config/grade_rule.json`에 있다. 코드에 박지 않는 이유는
3-tier UI의 90/60처럼 근거 없는 값이 굳어지는 것을 막기 위해서다 (명세 4.5).

ollama가 꺼져 있거나 응답이 깨지면 None을 돌린다. 호출부가 부분 실패로 처리한다
(명세 4.1) — 등급 하나 때문에 분석 전체를 날리지 않는다.

**인터넷이 필요 없다.** ollama는 127.0.0.1에만 붙고 모델은 로컬 디스크에 있다.
"""

import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from callwrap_ai.preprocess import strip_emotion_markers
from callwrap_ai.schemas import ConsultationGrade, Judgment, Speaker, Transcript

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULE = ROOT / "config/grade_rule.json"
DEFAULT_PROMPT = ROOT / "eval/prompts/sentiment_v1.md"
API = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
TIMEOUT = int(os.environ.get("CALLWRAP_LLM_TIMEOUT", "120"))


@lru_cache(maxsize=1)
def _rule() -> dict:
    path = Path(os.environ.get("CALLWRAP_GRADE_RULE", DEFAULT_RULE))
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _prompt() -> str:
    path = Path(os.environ.get("CALLWRAP_SENTIMENT_PROMPT", DEFAULT_PROMPT))
    return path.read_text(encoding="utf-8").split("\n## 프롬프트\n", 1)[1].strip()


def _ask(prompt: str, model: str) -> str | None:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": 42}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())["response"]
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return None  # 서버 꺼짐·타임아웃·응답 이상 — 부분 실패로 넘긴다


def _parse(raw: str, n: int) -> list[float] | None:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    by_n = {int(d["n"]): float(d["score"]) for d in arr if "n" in d and "score" in d}
    if len(by_n) < n:
        return None
    return [max(-1.0, min(1.0, by_n.get(i + 1, 0.0))) for i in range(n)]


def score_utterances(texts: list[str], model: str | None = None) -> list[float] | None:
    """민원인 발화 목록 -> 발화별 감정 점수(-1~1). 실패 시 None."""
    if not texts:
        return None
    rule = _rule()
    lines = "\n".join(f"{i+1}. {strip_emotion_markers(t)}" for i, t in enumerate(texts))
    raw = _ask(_prompt().format(lines=lines), model or rule["model"])
    return _parse(raw, len(texts)) if raw else None


def judge(transcript: Transcript, model: str | None = None) -> Judgment[ConsultationGrade] | None:
    """Transcript -> 상담등급 판정. 실패하면 None (부분 실패).

    2축 규칙:
      score_min  < abusive_max_score_min  → 악성콜  가장 격한 발화가 얼마나 격했나
      score_tail > praise_min_score_tail  → 칭찬콜  통화가 어떻게 끝났나

    한 축으로는 안 된다. score_min만 쓰면 칭찬콜을 전부 일반으로 흘리고,
    score_tail만 쓰면 악성 Recall이 0.85로 떨어진다.
    """
    cust = [u for u in transcript.utterances if u.speaker is Speaker.CUSTOMER]
    scores = score_utterances([u.text for u in cust], model)
    if not scores:
        return None

    rule = _rule()
    low, high = rule["abusive_max_score_min"], rule["praise_min_score_tail"]
    k = max(1, len(scores) // 3)
    s_min, s_tail = min(scores), sum(scores[-k:]) / k

    if s_min < low:
        grade, margin, span = ConsultationGrade.ABUSIVE, low - s_min, 1.0 + low
        pick = scores.index(s_min)
    elif s_tail > high:
        grade, margin, span = ConsultationGrade.PRAISE, s_tail - high, 1.0 - high
        pick = scores.index(max(scores))
    else:
        grade = ConsultationGrade.NORMAL
        margin, span = min(s_min - low, high - s_tail), max(high - low, 1e-6)
        pick = scores.index(s_min)

    # 확신도는 임계값에서 떨어진 거리다. **캘리브레이션된 확률이 아니다** —
    # 3-tier UI 임계값에 물리기 전에 보정이 필요하다 (명세 9.3, Step 4).
    conf = max(0.0, min(1.0, margin / span)) if span > 0 else 0.0
    return Judgment(value=grade, confidence=round(conf, 4), evidence_indices=[cust[pick].index])


def is_ready(model: str | None = None) -> bool:
    """ollama에 모델이 올라와 있는지. 서버 기동 시 확인용."""
    try:
        tags = json.loads(urllib.request.urlopen(
            API.replace("/api/generate", "/api/tags"), timeout=5).read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False
    want = model or _rule()["model"]
    return any(m.get("name") == want for m in tags.get("models", []))
