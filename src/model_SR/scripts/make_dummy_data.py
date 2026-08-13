"""API 키 없이 파이프라인을 스모크 테스트하기 위한 더미 데이터 생성기.

**학습용이 아니다.** 템플릿 조합으로 만든 가짜 대화이므로 모델 품질과는
무관하다. 배선(전처리 -> 학습 -> 캘리브레이션)이 끊기지 않는지 확인하는 용도다.

실제 학습 데이터는 callwrap.generate 로 만든다.

사용:
    python scripts/make_dummy_data.py --out data/dummy.jsonl -n 400
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from callwrap.dataset import save_jsonl          # noqa: E402
from callwrap.labels import LABELS, collapse     # noqa: E402
from callwrap.prompts import CALLER_ROLES, SUBCATEGORIES  # noqa: E402

OPENERS = [
    "안녕하세요, 문의드릴 게 있어서 전화드렸습니다.",
    "여보세요? 상담 좀 받으려고요.",
    "네 안녕하세요, 바쁘신데 죄송합니다.",
    "제가 지금 급해서요, 확인 좀 부탁드릴게요.",
]

CUSTOMER_LINES: dict[str, list[str]] = {
    "사용법문의": [
        "제공인력 등록을 어느 메뉴에서 하는지 모르겠어요.",
        "결제 화면이 어디에 있는지 알려주실 수 있나요?",
        "이 항목에는 뭘 입력해야 하는 건가요?",
    ],
    "시스템오류": [
        "저장 버튼을 눌러도 아무 반응이 없습니다.",
        "결제 금액이 실제보다 이상하게 계산돼서 나옵니다.",
        "다른 직원 컴퓨터에서도 똑같이 오류 화면이 뜹니다.",
    ],
    "고객PC환경": [
        "공동인증서가 인식이 안 됩니다.",
        "지문인식기에 빨간불만 계속 들어와 있어요.",
        "저희 사무실 컴퓨터에서만 이런 증상이 나옵니다.",
    ],
    "자료확인": [
        "지난달 결제 내역 좀 조회해 주실 수 있나요?",
        "이 대상자 자격이 언제까지인지 확인 부탁드립니다.",
        "접수한 건이 지금 어떻게 진행되고 있는지 알고 싶습니다.",
    ],
    "자료보정": [
        "제가 급여를 잘못 등록했는데 정정해 주세요.",
        "과오결제된 건을 취소해 주실 수 있을까요?",
        "대상자 정보가 잘못 들어가 있어서 수정 요청드립니다.",
    ],
    "정책 및 지침문의": [
        "본인부담금 면제 대상이 되는 기준이 어떻게 되나요?",
        "가산단가를 적용받으려면 어떤 요건을 충족해야 하나요?",
        "부정사용 신고 포상금은 규정상 얼마인가요?",
    ],
    "기능개선": [
        "목록을 엑셀로 내려받는 기능이 있으면 좋겠습니다.",
        "한 번에 여러 건을 등록할 수 있게 바꿔주실 수 없나요?",
        "화면에 담당자 이름도 같이 나오게 해주시면 편할 것 같아요.",
    ],
    "기타": [
        "여기가 국민연금공단 맞나요? 잘못 건 것 같습니다.",
        "상담센터 운영시간이 어떻게 되나요?",
        "지난번에 도와주셔서 감사하다고 말씀드리려고 전화했어요.",
    ],
}

AGENT_LINES: dict[str, list[str]] = {
    "사용법문의": [
        "좌측 메뉴에서 제공인력관리를 선택하신 다음 등록 버튼을 누르시면 됩니다.",
        "화면 상단의 조회 조건을 먼저 설정하시고 검색을 눌러 보세요.",
    ],
    "시스템오류": [
        "확인해 보니 서버 쪽 문제로 보입니다. 개발팀에 장애 접수해 드리겠습니다.",
        "다른 사용자분들도 동일 증상이 보고되고 있습니다. 처리 후 연락드리겠습니다.",
    ],
    "고객PC환경": [
        "혹시 다른 컴퓨터에서도 동일하게 나오시나요? 그쪽에서는 정상이라면 단말 문제입니다.",
        "보안 프로그램을 삭제하고 재설치하신 뒤 다시 시도해 보시겠어요?",
    ],
    "자료확인": [
        "제가 지금 조회해 드리겠습니다. 잠시만 기다려 주세요.",
        "확인해 보니 해당 건은 정상적으로 등록되어 있습니다.",
    ],
    "자료보정": [
        "정정 요청 접수해서 처리해 드리겠습니다.",
        "해당 건 취소 처리하고 결과는 문자로 안내드리겠습니다.",
    ],
    "정책 및 지침문의": [
        "지침상으로는 해당 요건을 모두 충족해야 적용이 가능합니다.",
        "관련 고시에 따르면 기준은 다음과 같습니다.",
    ],
    "기능개선": [
        "요청사항으로 접수해서 개선 검토하도록 하겠습니다.",
        "현재는 지원하지 않는 기능이라 개선 요청으로 등록해 드리겠습니다.",
    ],
    "기타": [
        "저희는 사회서비스 전자바우처 상담센터입니다. 해당 기관으로 안내드리겠습니다.",
        "평일 오전 9시부터 오후 6시까지 운영하고 있습니다.",
    ],
}

FILLER_PAIRS = [
    ("네, 알겠습니다.", "네 고객님, 더 궁금하신 점 있으실까요?"),
    ("아 그렇군요.", "네 맞습니다."),
    ("잠시만요, 확인해 볼게요.", "천천히 확인해 주세요."),
]


def make_record(rng: random.Random, idx: int) -> dict:
    primary = rng.choice(list(LABELS))
    labels = [primary]
    # 25%는 복합 케이스로 만든다
    if rng.random() < 0.25 and primary != "기타":
        second = rng.choice([l for l in LABELS if l not in (primary, "기타")])
        labels.append(second)

    utts = [
        {"speaker": "상담사", "text": "네, 사회서비스 전자바우처 상담센터입니다."},
        {"speaker": "고객", "text": rng.choice(OPENERS)},
    ]
    for lab in labels:
        utts.append({"speaker": "고객", "text": rng.choice(CUSTOMER_LINES[lab])})
        utts.append({"speaker": "상담사", "text": rng.choice(AGENT_LINES[lab])})
        a, b = rng.choice(FILLER_PAIRS)
        utts.append({"speaker": "고객", "text": a})
        utts.append({"speaker": "상담사", "text": b})
    utts.append({"speaker": "고객", "text": "네, 감사합니다."})
    utts.append({"speaker": "상담사", "text": "이용해 주셔서 감사합니다."})

    mid, sub = rng.choice(SUBCATEGORIES)
    return {
        "id": f"dummy-{idx:05d}",
        "sr_labels": labels,
        "sr_final": collapse(labels),
        "target_label": primary,
        "utterances": utts,
        "evidence_indices": [2],
        "rationale": "더미 데이터 (템플릿 조합)",
        "meta": {"mid_category": mid, "sub_category": sub,
                 "difficulty": "복합" if len(labels) > 1 else "일반",
                 "caller_role": rng.choice(CALLER_ROLES), "source": "dummy"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/dummy.jsonl"))
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = [make_record(rng, i) for i in range(args.n)]
    save_jsonl(records, args.out)

    from collections import Counter
    print(f"{len(records)}건 -> {args.out}")
    print("최종 라벨 분포:", Counter(r["sr_final"] for r in records).most_common())
    print("\n※ 배선 확인용 더미입니다. 학습 데이터로 쓰지 마십시오.")


if __name__ == "__main__":
    main()
