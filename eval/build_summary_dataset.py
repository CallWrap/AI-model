"""요약·후속처리·FAQ필요 데이터셋 조립 — F1·F4 평가용.

    uv run python eval/build_summary_dataset.py

입력
  eval/datasets/golden_category.jsonl   대본
  eval/datasets/gold_draft.jsonl        요약·needs_followup·needs_faq 라벨 (원본 100건)
  data/scenarios/scenarios_sub.jsonl    소분류 증강분 (summary_gold·labels 보유)

출력
  eval/datasets/summary_dataset.jsonl

명세 Step 2에 따라 **요약(F1)·후속처리(F4)·FAQ필요를 LLM 1회 호출로 묶어** 평가한다.
같은 문맥을 세 번 읽을 이유가 없다.

요약은 화자를 갈라야 한다 — 문의내용은 고객 발화, 처리내용은 상담사 발화 기준이다.
그래서 발화를 index·role·text로 보존한다. 감정 분석과 달리 **양쪽 발화가 다 필요하다.**

라벨은 전부 검수 전이다. 특히 요약 정답은 대본을 생성할 때 함께 만들어진 것이라
**참고 기준이지 절대 정답이 아니다** (명세 9.1). ROUGE 점수를 성능 근거로 쓰지 않는다.

`needs_faq`는 "일반화 가능한 문의인가"로 해석한다. "기존 FAQ에 없는가"가 아니다 —
대본 99/100이 FAQ에서 생성돼 전부 매칭되므로 그 정의로는 변별이 불가능하다.
빈도 누적과 기존 FAQ 매칭은 대시보드 영역이다 (docs/decisions.md 2026-07-15).
"""

import json
import re
from collections import Counter
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
DATASETS = AI / "eval/datasets"
GENERATED = AI / "data/scenarios/scenarios_sub.jsonl"
OUT = DATASETS / "summary_dataset.jsonl"

COUNSELOR = {"상담원", "상담사"}
OUT_OF_SCOPE = {"scn-115", "scn-128", "scn-019", "scn-121"}  # 장애인활동지원이 아닌 사업
FIELDS = ("title", "inquiry", "resolution")


def split_utterances(text: str) -> list[dict]:
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^([^:]{1,20}):\s*(.*)$", ln)
        if m and m.group(2).strip():
            role = "상담사" if m.group(1).strip() in COUNSELOR else "고객"
            out.append({"index": len(out), "role": role, "text": m.group(2).strip()})
    return out


def valid_summary(s: dict | None) -> bool:
    return bool(s) and all(str(s.get(f, "")).strip() for f in FIELDS)


def main() -> None:
    draft = {r["scenario_id"]: r for r in map(json.loads, (DATASETS / "gold_draft.jsonl").open(encoding="utf-8"))}

    rows, dropped = [], Counter()
    for line in (DATASETS / "golden_category.jsonl").open(encoding="utf-8"):
        g = json.loads(line)
        sid = g["scenario_id"]
        if sid in OUT_OF_SCOPE:
            dropped["스코프 밖"] += 1
            continue
        d = draft.get(sid, {})
        if not valid_summary(d.get("summary")):
            dropped["요약 정답 없음"] += 1
            continue
        if d.get("needs_followup") is None or d.get("needs_faq") is None:
            dropped["bool 라벨 없음"] += 1
            continue
        rows.append({
            "scenario_id": sid, "source": d.get("label_status", "draft"),
            "summary": {f: d["summary"][f] for f in FIELDS},
            "needs_followup": bool(d["needs_followup"]), "needs_faq": bool(d["needs_faq"]),
            "text": g["text"], "utterances": split_utterances(g["text"]),
        })

    seen = {r["scenario_id"] for r in rows}
    for line in GENERATED.open(encoding="utf-8"):
        s = json.loads(line)
        sid = s.get("spec_id") or s.get("id")
        lb = s.get("labels", {})
        if sid in seen or not valid_summary(s.get("summary_gold")):
            dropped["중복/요약 없음"] += 1
            continue
        if lb.get("needs_followup") is None or lb.get("needs_faq") is None:
            dropped["bool 라벨 없음"] += 1
            continue
        rows.append({
            "scenario_id": sid, "source": "generated",
            "summary": {f: s["summary_gold"][f] for f in FIELDS},
            "needs_followup": bool(lb["needs_followup"]), "needs_faq": bool(lb["needs_faq"]),
            "text": "\n".join(f"{u['speaker']}: {u['text']}" for u in s["utterances"]),
            "utterances": [{"index": i, "role": "상담사" if u["speaker"].strip() in COUNSELOR else "고객",
                            "text": u["text"].strip()} for i, u in enumerate(s["utterances"])],
        })
        seen.add(sid)

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"합본 {len(rows)}건 -> {OUT.relative_to(AI)}")
    print("  출처:", dict(Counter(r["source"] for r in rows)))
    if dropped:
        print("  제외:", dict(dropped))
    print("  needs_followup:", dict(Counter(r["needs_followup"] for r in rows)))
    print("  needs_faq     :", dict(Counter(r["needs_faq"] for r in rows)))

    for f in FIELDS:
        lens = [len(r["summary"][f]) for r in rows]
        print(f"  {f:<11} 평균 {sum(lens)/len(lens):5.1f}자  (최소 {min(lens)} / 최대 {max(lens)})")
    n_utt = [len(r["utterances"]) for r in rows]
    cust = [sum(1 for u in r["utterances"] if u["role"] == "고객") for r in rows]
    print(f"  발화 평균 {sum(n_utt)/len(rows):.1f}개 (고객 {sum(cust)/len(rows):.1f} / 상담사 {(sum(n_utt)-sum(cust))/len(rows):.1f})")


if __name__ == "__main__":
    main()
