"""골든셋 큐레이션 — 시나리오 130건 → 검수용 100건 선별.

전략:
  1. 품질 필터: 발화 10개 미만, 4자 미만 초단발화 포함 콜은 후순위
  2. 희소 우선 보존: 경계 케이스, 악성/칭찬콜, 희소 중분류(부정사용·지자체 등)는
     최대한 살리고, 과대표집된 셀(normal + 대상자관리/결제 v1)에서 30건을 덜어낸다
  3. v1/v2 라벨 상태를 유지해 다음 단계(LLM 초안 라벨링)로 넘긴다

선별 결과는 라벨을 확정하지 않는다 — 사람 검수가 정답을 정한다 (명세 9.1).

실행: uv run python eval/curate.py
출력: ai/data/scenarios/curated_100.jsonl
"""

import json
from collections import Counter
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
SCN_DIR = AI_DIR / "data/scenarios"
TARGET = 100


def load() -> list[dict]:
    scns = []
    for name in ("scenarios_v2.jsonl", "scenarios_v1_kept.jsonl"):
        for line in (SCN_DIR / name).open(encoding="utf-8"):
            s = json.loads(line)
            s["_id"] = s.get("spec_id") or s.get("id")
            s["_origin"] = "v2" if name.startswith("scenarios_v2") else "v1"
            scns.append(s)
    return scns


def quality_ok(s: dict) -> bool:
    u = s["utterances"]
    return len(u) >= 10 and all(len(x["text"]) >= 4 for x in u)


def cell(s: dict) -> tuple:
    """과대표집 판단용 키: (출처, 시나리오형, 중분류추정)."""
    cat = s["labels"]["consultation_category"] if s["_origin"] == "v2" else s.get("category_guess", "?")
    return (s["_origin"], s["scenario_type"], cat)


def main() -> None:
    scns = load()
    good = [s for s in scns if quality_ok(s)]
    poor = [s for s in scns if not quality_ok(s)]
    print(f"입력 {len(scns)} (품질 통과 {len(good)} / 후순위 {len(poor)})")

    # 보존 우선순위 점수 — 높을수록 지킨다
    def score(s: dict) -> float:
        sc = 0.0
        if s["scenario_type"] != "normal":
            sc += 3  # 경계 케이스는 귀하다
        grade = s["labels"]["grade"] if s["_origin"] == "v2" else s.get("legacy_labels", {}).get("grade")
        if grade in ("악성콜", "칭찬콜"):
            sc += 2  # 희소 등급
        if s["_origin"] == "v2":
            sc += 1  # 신 체계 라벨이라 검수 부담 적음 — 약간 우대
        return sc

    # 과대표집 셀에서 우선 덜어내기: 점수 낮은 것부터 제거 대상
    ranked = sorted(good, key=lambda s: (score(s), s["_id"]))
    drop_n = max(0, len(good) - TARGET)

    # 셀별 상한을 두어 특정 셀만 남지 않게 — 점수 낮은 것부터 훑되 셀 균형 유지
    keep, dropped = list(good), []
    cell_count = Counter(cell(s) for s in good)
    for s in ranked:
        if len(keep) <= TARGET:
            break
        c = cell(s)
        # 그 셀에 2건 이하만 남았으면 보존(다양성), 아니면 제거
        if cell_count[c] > 2:
            keep.remove(s)
            dropped.append(s)
            cell_count[c] -= 1

    # 아직 초과면 점수 낮은 순으로 마저 제거
    for s in ranked:
        if len(keep) <= TARGET:
            break
        if s in keep:
            keep.remove(s)
            dropped.append(s)

    keep = keep[:TARGET] if len(keep) > TARGET else keep

    out = SCN_DIR / "curated_100.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in keep:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"선별 {len(keep)}건 -> {out.name} (제거 {len(dropped)})")
    print("  출처:", dict(Counter(s["_origin"] for s in keep)))
    print("  시나리오형:", dict(Counter(s["scenario_type"] for s in keep)))
    v2grade = Counter(s["labels"]["grade"] for s in keep if s["_origin"] == "v2")
    v1grade = Counter(s.get("legacy_labels", {}).get("grade") for s in keep if s["_origin"] == "v1")
    print("  등급:", dict(v2grade + v1grade))


if __name__ == "__main__":
    main()
