"""FAQ 배치 합성 — needs_faq=true로 쌓인 콜들을 소분류 단위로 모아 EXAONE에 한 번에 넘겨
대표 질문/답변으로 합성한다. summary.py와 호출 방식(로컬 Ollama)은 같지만, 콜 하나가 아니라
여러 콜을 한 번에 입력받는다는 점이 다르다 — summary.py의 _ask/_parse 패턴을 그대로 따르되
합치지는 않았다(별도 트리거·별도 실패 모드라 독립적으로 유지하는 게 더 명확하다고 판단).

FastAPI의 배치 생성기(POST /internal/faq/generate-batch)가 소분류별로 이 함수를 호출한다.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from callwrap_ai.transcript_sanitizer import sanitize_text

ROOT = Path(__file__).resolve().parents[2]
#: 로컬 개발/레포 체크아웃 상태에서 쓰는 경로 — 편집 소스는 여기(summary_v1~v6.md와 같은 자리).
DEV_PROMPT = ROOT / "eval/prompts/faq_v1.md"
#: 패키지 설치 상태(server-fastapi가 git 소스로 callwrap-ai를 uv sync할 때)에서 쓰는 경로.
# pyproject.toml의 [tool.hatch.build.targets.wheel]이 packages=["src/callwrap_ai"]만 포함해서
# eval/은 휠에 안 들어간다 — 실제로 빌드해서 확인함(휠 안에 eval 관련 파일 0개). 그래서 이 파일과
# 내용이 동일한 사본을 src/callwrap_ai/prompts/ 밑에 따로 둔다. (summary.py의 eval/prompts/*.md도
# 같은 문제를 겪고 있을 가능성이 높다 — 별도로 확인·수정 필요, 여기서는 건드리지 않았다.)
PACKAGED_PROMPT = Path(__file__).resolve().parent / "prompts" / "faq_v1.md"
DEFAULT_MODEL = "exaone3.5:7.8b"
API = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
# 여러 사례를 한 번에 넣어 요약 1건보다 입력이 커진다 — summary.py 기본(120s)보다 여유를 둔다.
TIMEOUT = int(os.environ.get("CALLWRAP_LLM_TIMEOUT", "180"))

#: 소분류 하나에 후보가 아주 많이 쌓여도 프롬프트가 무한정 커지지 않게 상한을 둔다.
MAX_CASES = 20


def _prompt_template() -> str:
    env_override = os.environ.get("CALLWRAP_FAQ_PROMPT")
    candidates = [Path(env_override)] if env_override else [DEV_PROMPT, PACKAGED_PROMPT]
    last_error: Exception | None = None
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
            return text.split("\n## 프롬프트\n", 1)[1].strip()
        except OSError as e:
            last_error = e
            continue
    raise FileNotFoundError(f"FAQ 프롬프트 파일을 찾지 못했습니다 (시도한 경로: {candidates})") from last_error


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


def _format_cases(cases: list[dict]) -> str:
    lines = []
    for i, c in enumerate(cases):
        inquiry = sanitize_text((c.get("inquiry") or "").strip()) or "(문의 내용 없음)"
        resolution = sanitize_text((c.get("resolution") or "").strip()) or "(처리 내용 없음)"
        lines.append(f"[{i}] 문의: {inquiry}\n    처리: {resolution}")
    return "\n".join(lines)


def _parse(raw: str, num_cases: int) -> list[dict] | None:
    if not isinstance(raw, str):
        return None
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None

    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if not (isinstance(question, str) and question.strip()
                and isinstance(answer, str) and answer.strip()):
            continue
        raw_indices = item.get("source_indices")
        indices = (
            [i for i in raw_indices if isinstance(i, int) and 0 <= i < num_cases]
            if isinstance(raw_indices, list) else []
        )
        result.append({
            "question": sanitize_text(question.strip()),
            "answer": sanitize_text(answer.strip()),
            "source_indices": indices,
        })
    return result or None


def generate_faq(subcategory: str, cases: list[dict], model: str | None = None) -> list[dict] | None:
    """소분류 하나의 문의/처리 사례들 → FAQ 후보 리스트. 실패하면 None (호출부가 그 소분류만 건너뛴다).

    cases: [{"inquiry": str, "resolution": str}, ...] — Spring의 candidates 응답을 그대로 넣으면 된다.
    반환: [{"question": str, "answer": str, "source_indices": [int, ...]}, ...]
    source_indices는 입력 cases 리스트(잘라내기 전 아니라 실제로 프롬프트에 들어간 순서) 기준 인덱스다.
    """
    if not cases:
        return None
    limited = cases[:MAX_CASES]
    try:
        template = _prompt_template()
        prompt = template.format(
            subcategory=subcategory,
            count=len(limited),
            cases=_format_cases(limited),
        )
    except (OSError, IndexError, KeyError, ValueError):
        return None

    selected_model = model or os.environ.get("CALLWRAP_LLM_MODEL", DEFAULT_MODEL)
    raw = _ask(prompt, selected_model)
    if not isinstance(raw, str) or not raw.strip():
        return None
    result = _parse(raw, len(limited))
    if result is not None:
        return result

    raw = _ask(prompt + "\n\n반드시 JSON 배열 하나만 출력하세요.", selected_model)
    return _parse(raw, len(limited)) if isinstance(raw, str) and raw.strip() else None
