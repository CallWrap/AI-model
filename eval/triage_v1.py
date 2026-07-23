"""시나리오 v1 선별 — 기관 확정 체계(장애인활동지원 스코프, 유형 8종) 기준으로
살릴 대본과 폐기할 대본을 가른다.

폐기 사유:
  out_of_scope_seed  : 소재 FAQ가 전부 타 사업(일상돌봄·세종 지역사회서비스)
  off_scope_terms    : 대화 본문에 타 사업 고유 용어가 실제로 등장
  removed_type       : 폐지된 '타 기관 자료 확인' 유형

살린 대본의 구 라벨은 legacy_labels로 옮긴다 — 새 체계 라벨은 사람이 다시
붙인다 (개발명세 9.1).

실행: python ai/eval/triage_v1.py
출력: ai/data/scenarios/scenarios_v1_kept.jsonl  (+ 콘솔에 부족분 리포트)
"""

import json
import re
from collections import Counter
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
SRC = AI_DIR / "data/scenarios/scenarios_v1.jsonl"
DST = AI_DIR / "data/scenarios/scenarios_v1_kept.jsonl"

OUT_SEED_PREFIX = ("doc3", "web2")  # 일상돌봄, 세종 지역사회서비스
REMOVED_TYPE = "타 기관 자료 확인"
OFF_SCOPE_TERMS = re.compile(
    r"일상돌봄|가족돌봄청년|돌봄필요\s*중장년|특화서비스|기본서비스|지투|세종|지역사회서비스\s*투자"
)

# 구 유형 → 신 유형(8종) 표기 매핑. 참고용 힌트일 뿐 확정 라벨이 아니다.
TYPE_HINT = {
    "사용법 문의": "사용법문의",
    "자료 확인": "자료확인",
    "자료 보정": "자료보정",
    "시스템 오류": "시스템오류",
    "정책 및 지침 문의": "정책 및 지침문의",
    "기능 개선": "기능개선",
    "고객PC 환경": "고객PC환경",
    "기타": "기타",
}

# 중분류 추정용 키워드 — 부족분 스펙을 짜기 위한 어림값. 정식 라벨이 아니다.
CATEGORY_HINTS = {
    "결제및비용지급": ["결제", "단말기", "청구", "과오", "가산단가", "교통비", "비용 지급", "예외지급"],
    "대상자관리": ["대상자", "수급자", "본인부담금", "바우처 생성", "가상계좌", "자격", "신청", "캡"],
    "부정사용": ["부정", "신고", "포상금", "허위"],
    "제공기관관리": ["제공기관", "기관 지정", "기관 등록", "기관 운영", "휴업", "폐업"],
    "제공인력관리": ["제공인력", "활동지원사", "교육", "자격증", "급여등록", "채용"],
    "지자체사업관리": ["지자체", "예탁금", "행복e음", "시군구", "사업 운영"],
}


def guess_category(text: str) -> str:
    scores = {c: sum(text.count(k) for k in kws) for c, kws in CATEGORY_HINTS.items()}
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] else "미분류"


def main() -> None:
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8")]
    kept, dropped = [], []

    for r in rows:
        seeds = r.get("seed_faq_ids") or []
        text = " ".join(u["text"] for u in r["utterances"])
        old_type = r["labels"]["consultation_type"]

        if old_type == REMOVED_TYPE:
            reason = "removed_type"
        elif seeds and all(s.startswith(OUT_SEED_PREFIX) for s in seeds):
            reason = "out_of_scope_seed"
        elif OFF_SCOPE_TERMS.search(text):
            reason = "off_scope_terms"
        else:
            reason = None

        if reason:
            dropped.append((r["spec_id"], reason))
            continue

        kept.append(
            {
                "id": r["spec_id"],
                "origin": "v1",
                "scenario_type": r["scenario_type"],
                "seed_faq_ids": seeds,
                "utterances": r["utterances"],
                # 요약은 초안. title(요청제목)은 v1에 없어 라벨링 때 채운다.
                "summary_draft": r.get("summary_gold", {}),
                # 구 체계 라벨 — 참고용. 확정 라벨은 사람이 다시 붙인다.
                "legacy_labels": r["labels"],
                "type_hint": TYPE_HINT.get(old_type),
                "category_guess": guess_category(text),
            }
        )

    with DST.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"유지 {len(kept)}건 / 폐기 {len(dropped)}건 -> {DST.name}")
    print("폐기 사유:", dict(Counter(reason for _, reason in dropped)))
    print()
    print("=== 유지분 현황 (신규 생성 스펙 설계용) ===")
    print("유형(신표기):", dict(Counter(r["type_hint"] for r in kept)))
    print("등급:", dict(Counter(r["legacy_labels"]["grade"] for r in kept)))
    print("시나리오형:", dict(Counter(r["scenario_type"] for r in kept)))
    print("중분류 추정:", dict(Counter(r["category_guess"] for r in kept)))


if __name__ == "__main__":
    main()
