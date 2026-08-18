"""소분류 학습셋 조립 — 확정 골든셋 + 증강 생성분을 하나로 합친다.

    uv run python eval/build_subcategory_dataset.py

입력
  eval/datasets/golden_subcategory.jsonl   확정분 (source=human / draft)
  data/scenarios/scenarios_sub.jsonl       --sub-topup 생성분

출력
  eval/datasets/subcategory_dataset.jsonl  합본 (source=human / draft / generated)

source를 남기는 이유:
  human    사람이 검수한 라벨
  draft    검수 전 초안. 학습에는 쓰되 성능을 주장할 때 단서를 붙인다 (명세 9.1)
  generated 이번에 생성한 증강분. 검증셋에 섞이면 "내가 만든 걸 내가 맞히는" 상태가
            되므로, 평가 시 이 표시로 갈라서 두 숫자를 나란히 낸다

생성분은 스펙(요청한 라벨)이 아니라 실제 출력 labels를 정답으로 삼되, 계층이
어긋나면 버린다. 요청과 다르게 나온 건은 리포트에 남긴다.
"""

import json
from collections import Counter
from pathlib import Path

from callwrap_ai.schemas import SUBCATEGORIES_BY_CATEGORY

AI = Path(__file__).resolve().parents[1]
GOLDEN = AI / "eval/datasets/golden_subcategory.jsonl"
GENERATED = AI / "data/scenarios/scenarios_sub.jsonl"
OUT = AI / "eval/datasets/subcategory_dataset.jsonl"

BY_CAT = {c.value: {s.value for s in subs} for c, subs in SUBCATEGORIES_BY_CATEGORY.items()}


def main() -> None:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    seen = {r["scenario_id"] for r in rows}
    dropped, drifted = Counter(), []

    for line in GENERATED.open(encoding="utf-8"):
        s = json.loads(line)
        sid = s.get("spec_id") or s.get("id")
        if sid in seen:
            dropped["id 중복"] += 1
            continue
        lb, spec = s.get("labels", {}), s.get("spec", {})
        cat, sub = lb.get("consultation_category"), lb.get("consultation_subcategory")

        if cat not in BY_CAT or sub not in BY_CAT[cat]:
            dropped["계층 위반"] += 1
            continue
        if sub != spec.get("consultation_subcategory"):
            drifted.append((sid, spec.get("consultation_subcategory"), sub))

        rows.append({
            "scenario_id": sid, "category": cat, "subcategory": sub, "source": "generated",
            "text": "\n".join(f"{u['speaker']}: {u['text']}" for u in s["utterances"]),
        })
        seen.add(sid)

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"합본 {len(rows)}건 -> {OUT.relative_to(AI)}")
    print("  출처:", dict(Counter(r["source"] for r in rows)))
    if dropped:
        print("  제외:", dict(dropped))
    if drifted:
        print(f"  요청과 다른 소분류로 생성된 건 {len(drifted)}건 (계층은 맞아 유지):")
        for sid, want, got in drifted[:8]:
            print(f"    {sid}  요청={want} → 실제={got}")

    cnt = Counter(r["subcategory"] for r in rows)
    thin = {s: cnt.get(s, 0) for subs in BY_CAT.values() for s in subs if cnt.get(s, 0) < 5}
    print(f"\n소분류 {len(cnt)}/29종 등장, 최소 {min(cnt.values())}건 / 최대 {max(cnt.values())}건")
    if thin:
        print("  5건 미만 (검증셋에 못 들어갈 수 있음):", dict(sorted(thin.items(), key=lambda x: x[1])))


if __name__ == "__main__":
    main()
