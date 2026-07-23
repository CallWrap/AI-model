"""상담 시나리오 생성기 v2 — 기관 확정 체계(장애인활동지원, 유형 8종, 중분류 6·소분류 29)로
부족분을 생성한다. 라벨 체계는 callwrap_ai.schemas를 단일 출처로 삼는다.

v1에서 살린 85건(triage_v1.py)의 중분류가 대상자관리·결제에 편중돼 있어,
신규분은 비어 있는 중분류(지자체사업관리·제공인력관리·부정사용 등)를 채우도록 배분한다.

실행:  uv run python eval/generate_scenarios.py           # 전체 (이어하기 지원)
       uv run python eval/generate_scenarios.py --smoke   # 첫 배치만

출력:  ai/data/scenarios/batches_v2/batch_NNN.json  (배치별 원본 응답)
       ai/data/scenarios/scenarios_v2.jsonl         (검증 통과분)
"""

import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from callwrap_ai.schemas import (
    SUBCATEGORIES_BY_CATEGORY,
    ConsultationCategory,
    ConsultationGrade,
    ConsultationSubcategory,
    ConsultationType,
)

AI_DIR = Path(__file__).resolve().parents[1]
FAQ_PATH = AI_DIR / "data/processed/faq_all.jsonl"
PROMPT_PATH = AI_DIR / "eval/prompts/scenario_v2.md"
OUT_DIR = AI_DIR / "data/scenarios"
BATCH_DIR = OUT_DIR / "batches_v2"
MERGED = OUT_DIR / "scenarios_v2.jsonl"

BATCH_SIZE = 3
MODEL = "sonnet"
SEED = 21

# 소재 FAQ는 장애인활동지원에 적용되는 출처만 사용 (doc3=일상돌봄, web2=세종 제외)
IN_SCOPE_FAQ = ("doc1", "doc2", "web1", "web3")

# 중분류별 신규 생성량 — v1 유지분의 빈 곳을 메운다
TOPUP = {
    ConsultationCategory.MUNICIPAL: 12,  # 유지분 1건
    ConsultationCategory.PROVIDER_STAFF: 11,  # 4건
    ConsultationCategory.FRAUD: 10,  # 4건
    ConsultationCategory.PROVIDER_ORG: 8,  # 8건
    ConsultationCategory.RECIPIENT: 2,  # 35건
    ConsultationCategory.PAYMENT: 2,  # 31건
}

# 중분류별 소재 FAQ 검색 키워드
CATEGORY_KEYWORDS = {
    ConsultationCategory.PAYMENT: ["결제", "단말기", "청구", "과오", "가산", "교통비", "지급"],
    ConsultationCategory.RECIPIENT: ["대상자", "수급자", "본인부담", "바우처", "자격", "신청"],
    ConsultationCategory.FRAUD: ["부정", "신고", "포상금", "허위", "환수"],
    ConsultationCategory.PROVIDER_ORG: ["제공기관", "지정", "등록", "휴업", "폐업"],
    ConsultationCategory.PROVIDER_STAFF: ["활동지원사", "제공인력", "교육", "결격", "급여등록"],
    ConsultationCategory.MUNICIPAL: ["지자체", "예탁금", "행복e음", "시군구", "시·군·구", "조례"],
}

# 중분류별 주 문의자 역할
CALLER_BY_CATEGORY = {
    ConsultationCategory.MUNICIPAL: "지자체 담당 공무원",
    ConsultationCategory.PROVIDER_STAFF: "제공기관 담당자",
    ConsultationCategory.PROVIDER_ORG: "제공기관 담당자",
    ConsultationCategory.FRAUD: "제공기관 담당자",
    ConsultationCategory.PAYMENT: "제공기관 담당자",
    ConsultationCategory.RECIPIENT: "이용자(수급자 또는 보호자)",
}

# v1 유지분에서 적었던 유형을 앞에 둬 신규분에 더 자주 배정
TYPE_ORDER = [
    ConsultationType.ETC,
    ConsultationType.CUSTOMER_PC,
    ConsultationType.POLICY,
    ConsultationType.USAGE,
    ConsultationType.DATA_CHECK,
    ConsultationType.SYSTEM_ERROR,
    ConsultationType.DATA_FIX,
    ConsultationType.IMPROVEMENT,
]
BOUNDARY_PLAN = ["mixed"] * 6 + ["topic_shift"] * 3 + ["smalltalk"] * 3 + ["ambiguous"] * 3

VALID_TYPES = {t.value for t in ConsultationType}
VALID_CATEGORIES = {c.value for c in ConsultationCategory}
VALID_SUBCATEGORIES = {s.value for s in ConsultationSubcategory}
VALID_GRADES = {g.value for g in ConsultationGrade}


def build_specs() -> list[dict]:
    rng = random.Random(SEED)
    specs: list[dict] = []
    for cat, count in TOPUP.items():
        subs = sorted(SUBCATEGORIES_BY_CATEGORY[cat], key=lambda s: s.value)
        for i in range(count):
            specs.append(
                {
                    "category": cat,
                    "subcategory": subs[i % len(subs)],  # 소분류 고르게 순회
                    "caller": CALLER_BY_CATEGORY[cat],
                }
            )
    rng.shuffle(specs)

    boundary = BOUNDARY_PLAN + ["normal"] * (len(specs) - len(BOUNDARY_PLAN))
    for i, s in enumerate(specs):
        s["id"] = f"v2-{i + 1:03d}"
        s["consultation_type"] = TYPE_ORDER[i % len(TYPE_ORDER)].value
        s["scenario_type"] = boundary[i]
        s["grade"] = (
            ConsultationGrade.ABUSIVE.value
            if i % 8 == 1
            else ConsultationGrade.PRAISE.value
            if i % 8 == 5
            else ConsultationGrade.NORMAL.value
        )
        s["needs_followup"] = i % 10 in (0, 3, 7)
    return specs


def sample_faqs(faqs: list[dict], cat: ConsultationCategory, rng: random.Random, k: int = 2) -> list[dict]:
    kws = CATEGORY_KEYWORDS[cat]
    matched = [f for f in faqs if any(kw in f["question"] + f["answer"] for kw in kws)]
    return rng.sample(matched or faqs, min(k, len(matched or faqs)))


def build_batch_prompt(base: str, batch: list[dict], faqs: list[dict], rng: random.Random) -> str:
    parts = [base, "\n---\n## 이번에 생성할 시나리오 스펙\n"]
    for s in batch:
        seeds = sample_faqs(faqs, s["category"], rng)
        parts.append(f"### 스펙 {s['id']}")
        parts.append(
            json.dumps(
                {
                    "id": s["id"],
                    "scenario_type": s["scenario_type"],
                    "consultation_type": s["consultation_type"],
                    "consultation_category": s["category"].value,
                    "consultation_subcategory": s["subcategory"].value,
                    "grade": s["grade"],
                    "needs_followup": s["needs_followup"],
                    "caller": s["caller"],
                },
                ensure_ascii=False,
            )
        )
        parts.append("소재 FAQ:")
        for f in seeds:
            parts.append(f"- [{f['id']}] Q: {f['question']}\n  A: {f['answer'][:600]}")
        parts.append("")
    parts.append(f"\n위 {len(batch)}개 스펙에 대해 각 1개씩, JSON 배열로만 출력하세요.")
    return "\n".join(parts)


def load_oauth_token() -> str:
    """ai/.env의 CLAUDE_CODE_OAUTH_TOKEN (claude setup-token으로 발급)."""
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
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI 실패: {r.stderr[:300]}")
    return r.stdout


def parse_scenarios(raw: str, batch: list[dict]) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        raise ValueError("JSON 배열 없음")
    arr = json.loads(m.group(0))
    spec_by_id = {s["id"]: s for s in batch}
    ok = []
    for scn in arr:
        spec = spec_by_id.get(scn.get("spec_id"))
        if not spec:
            print(f"    무효: 알 수 없는 spec_id={scn.get('spec_id')}")
            continue
        lb, utts = scn.get("labels", {}), scn.get("utterances", [])
        sid = spec["id"]
        if lb.get("consultation_type") not in VALID_TYPES:
            print(f"    무효({sid}): 유형={lb.get('consultation_type')}")
            continue
        if lb.get("consultation_category") not in VALID_CATEGORIES:
            print(f"    무효({sid}): 중분류={lb.get('consultation_category')}")
            continue
        if lb.get("consultation_subcategory") not in VALID_SUBCATEGORIES:
            print(f"    무효({sid}): 소분류={lb.get('consultation_subcategory')}")
            continue
        if lb.get("grade") not in VALID_GRADES or len(utts) < 8:
            print(f"    무효({sid}): 등급/발화수")
            continue
        if not scn.get("summary_gold", {}).get("title"):
            print(f"    무효({sid}): 요청제목 없음")
            continue
        for i, u in enumerate(utts):
            u["index"] = i
        scn["spec"] = {
            "id": sid,
            "consultation_type": spec["consultation_type"],
            "consultation_category": spec["category"].value,
            "consultation_subcategory": spec["subcategory"].value,
            "grade": spec["grade"],
            "scenario_type": spec["scenario_type"],
            "needs_followup": spec["needs_followup"],
            "caller": spec["caller"],
        }
        ok.append(scn)
    return ok


def main() -> None:
    smoke = "--smoke" in sys.argv
    rng = random.Random(SEED + 1)
    faqs = [
        json.loads(line)
        for line in FAQ_PATH.open(encoding="utf-8")
        if json.loads(line)["id"].startswith(IN_SCOPE_FAQ)
    ]
    base = PROMPT_PATH.read_text(encoding="utf-8")
    specs = build_specs()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    batches = [specs[i : i + BATCH_SIZE] for i in range(0, len(specs), BATCH_SIZE)]
    if smoke:
        batches = batches[:1]

    all_ok = []
    for bi, batch in enumerate(batches):
        out = BATCH_DIR / f"batch_{bi:03d}.json"
        if out.exists():
            try:
                all_ok += parse_scenarios(out.read_text(encoding="utf-8"), batch)
                print(f"[{bi + 1}/{len(batches)}] 캐시 사용")
                continue
            except Exception:
                pass
        try:
            raw = call_claude(build_batch_prompt(base, batch, faqs, rng))
        except Exception as e:
            print(f"[{bi + 1}/{len(batches)}] 호출 실패: {e}")
            continue
        out.write_text(raw, encoding="utf-8")
        try:
            got = parse_scenarios(raw, batch)
        except Exception as e:
            print(f"[{bi + 1}/{len(batches)}] 파싱 실패: {e}")
            continue
        all_ok += got
        print(f"[{bi + 1}/{len(batches)}] {len(got)}/{len(batch)}건 성공 (누적 {len(all_ok)})")

    with MERGED.open("w", encoding="utf-8") as f:
        for s in all_ok:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n저장: {len(all_ok)}건 -> {MERGED}")
    print("유형:", dict(Counter(s["labels"]["consultation_type"] for s in all_ok)))
    print("중분류:", dict(Counter(s["labels"]["consultation_category"] for s in all_ok)))
    print("등급:", dict(Counter(s["labels"]["grade"] for s in all_ok)))


if __name__ == "__main__":
    main()
