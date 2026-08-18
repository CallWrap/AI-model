"""상담등급 데이터셋 조립 — 원본 골든셋 + 증강 생성분을 하나로 합친다.

    uv run python eval/build_grade_dataset.py

입력
  eval/datasets/golden_category.jsonl   대본 (중분류 골든셋과 공유)
  eval/datasets/gold_draft.jsonl        등급 라벨 초안
  data/scenarios/scenarios_sub.jsonl    소분류 증강분 (등급 라벨도 함께 생성됨)

출력
  eval/datasets/grade_dataset.jsonl     합본. scenario_id / grade / source / text /
                                        customer_utterances (고객 발화만 추린 것)

감정분석은 **민원인(고객) 발화만** 대상으로 한다. 상담사 발화에는 "안 됩니다" 같은
안내 표현이 많아 그대로 넣으면 불만으로 잡힌다. 그래서 조립 시점에 갈라 둔다.

등급 라벨은 전부 검수 전이다 (초안 + 생성 시점 값). 성능을 주장할 때 단서를 붙인다.
감정 표지 제거는 저장하지 않고 소비 시점에 preprocess.strip_emotion_markers()로 건다.
포함/제거 두 버전을 비교해야 하기 때문이다.
"""

import json
import re
from collections import Counter
from pathlib import Path

from callwrap_ai.schemas import ConsultationGrade

AI = Path(__file__).resolve().parents[1]
DATASETS = AI / "eval/datasets"
GENERATED = AI / "data/scenarios/scenarios_sub.jsonl"
OUT = DATASETS / "grade_dataset.jsonl"

VALID = {g.value for g in ConsultationGrade}
COUNSELOR = {"상담원", "상담사"}

# 장애인활동지원이 아닌 다른 바우처 사업 — 소분류와 같은 기준으로 제외한다.
OUT_OF_SCOPE = {"scn-115", "scn-128", "scn-019", "scn-121"}


def split_speakers(text: str) -> list[str]:
    """대본 평문 -> 고객 발화만. 화자 라벨이 15종이라 상담사만 골라내고 나머지를 고객으로 본다."""
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^([^:]{1,20}):\s*(.*)$", ln)
        if m and m.group(1).strip() not in COUNSELOR and m.group(2).strip():
            out.append(m.group(2).strip())
    return out


def main() -> None:
    draft = {r["scenario_id"]: r for r in map(json.loads, (DATASETS / "gold_draft.jsonl").open(encoding="utf-8"))}

    rows, dropped = [], Counter()
    for line in (DATASETS / "golden_category.jsonl").open(encoding="utf-8"):
        g = json.loads(line)
        sid = g["scenario_id"]
        if sid in OUT_OF_SCOPE:
            dropped["스코프 밖"] += 1
            continue
        grade = draft.get(sid, {}).get("grade")
        if grade not in VALID:
            dropped["등급 라벨 없음"] += 1
            continue
        rows.append({"scenario_id": sid, "grade": grade, "source": draft[sid].get("label_status", "draft"),
                     "text": g["text"], "customer_utterances": split_speakers(g["text"])})

    seen = {r["scenario_id"] for r in rows}
    for line in GENERATED.open(encoding="utf-8"):
        s = json.loads(line)
        sid = s.get("spec_id") or s.get("id")
        grade = s.get("labels", {}).get("grade")
        if sid in seen or grade not in VALID:
            dropped["중복/등급 없음"] += 1
            continue
        text = "\n".join(f"{u['speaker']}: {u['text']}" for u in s["utterances"])
        cust = [u["text"].strip() for u in s["utterances"]
                if u["speaker"].strip() not in COUNSELOR and u["text"].strip()]
        rows.append({"scenario_id": sid, "grade": grade, "source": "generated",
                     "text": text, "customer_utterances": cust})
        seen.add(sid)

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"합본 {len(rows)}건 -> {OUT.relative_to(AI)}")
    print("  등급:", dict(Counter(r["grade"] for r in rows)))
    print("  출처:", dict(Counter(r["source"] for r in rows)))
    if dropped:
        print("  제외:", dict(dropped))

    n_cust = [len(r["customer_utterances"]) for r in rows]
    print(f"\n고객 발화 총 {sum(n_cust)}개 — 통화당 평균 {sum(n_cust)/len(rows):.1f}개 "
          f"(최소 {min(n_cust)} / 최대 {max(n_cust)})")
    empty = [r["scenario_id"] for r in rows if not r["customer_utterances"]]
    if empty:
        print(f"  ⚠️ 고객 발화가 0개인 건 {len(empty)}건 — 감정 점수를 낼 수 없다: {empty}")

    marked = sum(1 for r in rows if re.search(r"\((욕설|한숨)\)", r["text"]))
    print(f"\n감정 표지가 든 통화 {marked}건 ({marked/len(rows):.1%}) — 학습 시 제거 대상")
    by_grade = Counter(r["grade"] for r in rows if re.search(r"\((욕설|한숨)\)", r["text"]))
    print("  표지가 든 통화의 등급:", dict(by_grade))


if __name__ == "__main__":
    main()
