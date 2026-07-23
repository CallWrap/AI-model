"""v1 유지분 재라벨링 — 구 체계 대본에 새 8종·계층 분류 라벨 '초안'을 단다.

curated_100.jsonl의 v1 대본(구 라벨 + 소분류·요청제목 없음)을 LLM에 넣어
새 체계 라벨 초안을 받는다. 이 초안은 사람 검수 대상이며 정답이 아니다 (명세 9.1).
v2는 생성 시 이미 새 체계 라벨이 있으므로 대상이 아니다.

실행:  uv run python eval/relabel_v1.py           # 전체 (이어하기 지원)
       uv run python eval/relabel_v1.py --smoke    # 첫 배치만
출력:  ai/data/scenarios/relabel_v1/batch_NNN.json  (배치 원본)
       ai/data/scenarios/v1_label_drafts.jsonl      (검증 통과 초안)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from callwrap_ai.schemas import (
    SUBCATEGORIES_BY_CATEGORY,
    ConsultationCategory,
    ConsultationGrade,
    ConsultationSubcategory,
    ConsultationType,
)

AI_DIR = Path(__file__).resolve().parents[1]
SCN_DIR = AI_DIR / "data/scenarios"
PROMPT_PATH = AI_DIR / "eval/prompts/relabel_v1.md"
BATCH_DIR = SCN_DIR / "relabel_v1"
OUT = SCN_DIR / "v1_label_drafts.jsonl"
BATCH_SIZE = 3
MODEL = "sonnet"

VALID_TYPES = {t.value for t in ConsultationType}
VALID_CATEGORIES = {c.value for c in ConsultationCategory}
VALID_GRADES = {g.value for g in ConsultationGrade}


def load_v1() -> list[dict]:
    rows = [json.loads(line) for line in (SCN_DIR / "curated_100.jsonl").open(encoding="utf-8")]
    return [r for r in rows if r["_origin"] == "v1"]


def dialogue_text(scn: dict) -> str:
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in scn["utterances"])


def build_prompt(base: str, batch: list[dict]) -> str:
    parts = [base, "\n---\n## 라벨링할 대본\n"]
    for s in batch:
        parts.append(f"### 대본 {s['_id']}")
        parts.append(dialogue_text(s))
        parts.append("")
    parts.append(f"\n위 {len(batch)}개 대본에 대해 각 1개씩, JSON 배열로만 출력하세요.")
    return "\n".join(parts)


def load_oauth_token() -> str:
    env_file = AI_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CLAUDE_CODE_OAUTH_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def call_claude(prompt: str) -> str:
    env_cmd = ["env", "-u", "ANTHROPIC_BASE_URL", "-u", "CLAUDECODE", "-u", "CLAUDE_CODE_SESSION_ID"]
    token = load_oauth_token()
    if token:
        env_cmd.append(f"CLAUDE_CODE_OAUTH_TOKEN={token}")
    r = subprocess.run(
        env_cmd + ["claude", "-p", "--model", MODEL],
        input=prompt, capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI 실패: {r.stderr[:300]}")
    return r.stdout


def parse(raw: str, batch: list[dict]) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        raise ValueError("JSON 배열 없음")
    ids = {s["_id"] for s in batch}
    ok = []
    for d in json.loads(m.group(0)):
        did = d.get("id")
        if did not in ids:
            print(f"    무효: id={did}")
            continue
        cat, sub = d.get("consultation_category"), d.get("consultation_subcategory")
        if d.get("consultation_type") not in VALID_TYPES or cat not in VALID_CATEGORIES:
            print(f"    무효({did}): 유형/중분류")
            continue
        # 계층 정합: 소분류가 그 중분류에 속하는지
        cat_enum = ConsultationCategory(cat)
        try:
            sub_enum = ConsultationSubcategory(sub)
        except ValueError:
            print(f"    무효({did}): 소분류={sub}")
            continue
        if sub_enum not in SUBCATEGORIES_BY_CATEGORY[cat_enum]:
            print(f"    무효({did}): {sub} not in {cat}")
            continue
        if d.get("grade") not in VALID_GRADES or not d.get("title"):
            print(f"    무효({did}): 등급/제목")
            continue
        d["label_status"] = "machine_draft"  # 사람 검수 대상
        ok.append(d)
    return ok


def main() -> None:
    smoke = "--smoke" in sys.argv
    base = PROMPT_PATH.read_text(encoding="utf-8")
    v1 = load_v1()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = [v1[i:i + BATCH_SIZE] for i in range(0, len(v1), BATCH_SIZE)]
    if smoke:
        batches = batches[:1]

    all_ok = []
    for bi, batch in enumerate(batches):
        out = BATCH_DIR / f"batch_{bi:03d}.json"
        if out.exists():
            try:
                all_ok += parse(out.read_text(encoding="utf-8"), batch)
                print(f"[{bi + 1}/{len(batches)}] 캐시 사용")
                continue
            except Exception:
                pass
        try:
            raw = call_claude(build_prompt(base, batch))
        except Exception as e:
            print(f"[{bi + 1}/{len(batches)}] 호출 실패: {e}")
            continue
        out.write_text(raw, encoding="utf-8")
        try:
            got = parse(raw, batch)
        except Exception as e:
            print(f"[{bi + 1}/{len(batches)}] 파싱 실패: {e}")
            continue
        all_ok += got
        print(f"[{bi + 1}/{len(batches)}] {len(got)}/{len(batch)}건 (누적 {len(all_ok)})")

    with OUT.open("w", encoding="utf-8") as f:
        for d in all_ok:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n저장: {len(all_ok)}건 -> {OUT.name}")


if __name__ == "__main__":
    main()
