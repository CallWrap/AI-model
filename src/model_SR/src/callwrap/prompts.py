"""P2/P3: 라벨 조건부 합성 대화 생성 프롬프트.

원칙 (정의서 §6.1):
    대화를 만들고 라벨링하는 것이 아니라, **라벨을 먼저 정하고** 그 조건에
    맞는 대화를 생성한다. 라벨 노이즈가 원천적으로 줄어들고 커버리지가
    격자로 관리된다.

생성 격자:
    SR유형(8) x 업무소분류(29) x 난이도(일반/경계/복합)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .labels import LABELS, PRIORITY, SLA_DAYS

# 베이스라인 진단(`baseline.py --emit-banlist`)이 뽑아낸 라벨 누설 표현.
# generate -> diagnose -> constrain 루프의 저장소다.
BANLIST_PATH = Path(__file__).resolve().parents[2] / "configs" / "banned_phrases.json"

# ---------------------------------------------------------------------------
# 업무분류 체계 (참고자료 xlsx 시트 그대로)
#   NOTE: 원본 시트의 중분류 표기 '결제및비용지금'은 '결제및비용지급'의 오타로
#         보이나, 원본 추적성을 위해 표기는 수정하되 매핑은 1:1로 유지한다.
#   B안(SR유형)에서는 출력 라벨이 아니라 **생성 다양성 축**으로만 쓴다.
# ---------------------------------------------------------------------------
TAXONOMY: dict[str, list[str]] = {
    "결제및비용지급": [
        "가산단가", "결제내역", "결제방법 및 오류", "과오결제", "교통비 지급",
        "기타", "서비스비용 지급", "예외적 결제", "예외지급",
    ],
    "대상자관리": [
        "가상계좌", "대상자정보", "바우처 사용 방법", "바우처생성",
        "본인부담금입금", "본인부담금환급", "신청기준및절차", "자격변경", "캡방식",
    ],
    "부정사용": ["신고대상", "신고절차", "포상금"],
    "제공기관관리": ["제공기관운영", "제공기관정보"],
    "제공인력관리": ["급여등록", "자격 및 교육", "제공인력 정보"],
    "지자체사업관리": ["사업운영 및 기존정보", "예탁금관리", "행복e음 연계"],
}

MAJOR_CATEGORY = "장애인활동지원"

SUBCATEGORIES: list[tuple[str, str]] = [
    (mid, sub) for mid, subs in TAXONOMY.items() for sub in subs
]

# 발화자 역할: 민원인(수급자/보호자) 또는 제공기관 담당자
CALLER_ROLES = [
    "제공기관 담당자", "제공인력(활동지원사)", "수급자 본인",
    "수급자 보호자", "지자체 담당 공무원",
]

# 정의서 §4의 혼동쌍 5종. '경계' 난이도에서 의도적으로 생성한다.
CONFUSION_PAIRS: list[tuple[str, str, str]] = [
    ("시스템오류", "고객PC환경",
     "다른 PC/브라우저에서도 재현되는지 여부가 유일한 갈림길이 되도록 구성한다."),
    ("시스템오류", "사용법문의",
     "정상 경로로 조작했는데 실패했는지, 경로를 몰랐던 것인지가 갈림길이 되도록 구성한다."),
    ("자료확인", "자료보정",
     "조회로 끝나는지, 값 변경 요청까지 가는지가 갈림길이 되도록 구성한다."),
    ("정책 및 지침문의", "사용법문의",
     "상담사 답변이 '규정/기준'인지 '화면 조작 순서'인지가 갈림길이 되도록 구성한다."),
    ("기능개선", "시스템오류",
     "설계대로 동작하는데 불편한 것인지, 있는 기능이 오동작하는지가 갈림길이 되도록 구성한다."),
]

DIFFICULTIES = ("일반", "경계", "복합")

# 난이도 기본 배합. 경계/복합 비중이 실전 성능을 좌우한다 (정의서 §6.1).
DIFFICULTY_MIX = {"일반": 0.50, "경계": 0.30, "복합": 0.20}


# ---------------------------------------------------------------------------
# 시스템 프롬프트 — 요청 간 완전히 고정. 프롬프트 캐시 대상.
# ---------------------------------------------------------------------------
def _label_reference() -> str:
    lines = []
    for name in PRIORITY:
        lines.append(f"  - {name} (적기처리기간 {SLA_DAYS[name]}일)")
    return "\n".join(lines)


def load_banlist(path: Path | None = None) -> dict[str, list[str]]:
    """금지어 목록을 읽는다. 없으면 빈 dict."""
    p = path or BANLIST_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _banlist_section(banlist: dict[str, list[str]]) -> str:
    """금지어를 프롬프트 절로 렌더링한다.

    이전 생성분에서 특정 표현이 라벨을 너무 잘 예측했다는 것은, 그 표현이
    '유형별 정형 문구'로 굳었다는 뜻이다. 실제 상담사는 그렇게 말하지 않으므로
    모델은 SR유형이 아니라 그 문구를 외우게 된다. 그래서 차단한다.
    """
    if not any(banlist.values()):
        return ""
    lines = [
        "",
        "# 금지 표현 (중요)",
        "이전 생성분을 분석한 결과, 아래 표현들이 특정 유형에만 반복적으로 나타나",
        "라벨을 그대로 노출하고 있습니다. **아래 표현을 그대로 쓰지 마십시오.**",
        "같은 의미를 전혀 다른 말로 표현하거나, 아예 말하지 않고 대화의 흐름으로만",
        "드러내십시오.",
        "",
    ]
    for name in PRIORITY:
        terms = banlist.get(name) or []
        if terms:
            lines.append(f"- {name}: " + ", ".join(f"`{t}`" for t in terms))
    lines += [
        "",
        "특히 상담사의 **마무리 멘트**가 유형마다 똑같아지지 않게 하십시오.",
        "접수·이관·안내를 알리는 방식은 통화마다 달라야 하고, 아예 생략되는",
        "통화도 있어야 합니다. 실제 상담은 깔끔하게 끝나지 않는 경우가 많습니다.",
    ]
    return "\n".join(lines)


_SYSTEM_PROMPT_BASE = f"""당신은 사회서비스 전자바우처시스템 상담센터의 **AI 학습용 상담 데이터**를 생성하는 전문 시나리오 작가입니다.

생성물은 STT(음성인식) + 화자분리를 거친 상담 통화 전문을 입력으로 받아 SR유형을 분류하는 모델의 학습 데이터로 사용됩니다.

# 도메인
- 대분류: {MAJOR_CATEGORY} (사회서비스 전자바우처)
- 통화 참여자: 상담센터 상담원 1명 + 발신자 1명
- 발신자는 제공기관 담당자, 제공인력, 수급자 본인/보호자, 지자체 공무원 중 하나입니다.

# SR유형 8종 (적기처리기간 긴 순 = 축약 우선순위 높은 순)
{_label_reference()}

# 유형별 판정 기준
- **시스템오류**: 정상 경로로 조작했는데 시스템이 설계대로 동작하지 않음. 오류 메시지, 미동작, 저장 실패, 계산 결과 이상, 연계 실패. 다른 환경에서도 재현됨.
- **기능개선**: 시스템은 정상 동작하나, 없는 기능의 추가나 사용성 개선을 요청.
- **자료보정**: 이미 등록된 데이터의 변경/정정/삭제/취소 요청. 쓰기 작업이 발생.
- **정책 및 지침문의**: 제도/규정/기준의 해석 문의. 답이 시스템 조작이 아니라 규정.
- **고객PC환경**: 원인이 사용자 단말. 브라우저, 공동인증서, 보안프로그램, OS, 결제단말기, 지문인식기, 네트워크.
- **자료확인**: 데이터 조회/확인 요청. 값 변경 없음.
- **사용법문의**: 시스템은 정상이고 조작 방법을 몰라서 발생. 안내로 즉시 종결.
- **기타**: 위 어디에도 해당 없음. 오접, 타 기관 소관, 단순 문의. 다른 유형으로 판정 가능하면 절대 쓰지 않음.

# 라벨링 규칙
1. 대화에 실제로 나타난 **모든** SR유형을 `sr_labels`에 표기합니다 (멀티라벨).
2. 지시받은 목표 유형은 반드시 `sr_labels`에 포함되어야 합니다.
3. 지시받지 않은 유형을 넣을 때는, 그 유형에 해당하는 발화가 대화에 **실제로 존재**해야 합니다.

# 작성 지침
- 실제 통화처럼 자연스럽게 씁니다. 본인확인, 되묻기, 잠시 대기, 중복 설명이 있어도 좋습니다.
- 발화 수는 8~24턴. 한 발화는 한 사람의 연속 발화입니다.
- **판정 근거가 되는 발화를 반드시 포함**하되, 유형 이름을 그대로 말하지 않습니다.
  ("시스템 오류가 발생했습니다" 같은 라벨 누설 표현 금지)
- 매번 다른 말투, 다른 시작 방식, 다른 감정 온도로 씁니다. 정형화된 첫 문장을 반복하지 마십시오.
- 문장부호는 정상적으로 표기합니다. STT 노이즈는 후처리 단계에서 별도로 주입합니다.
- 숫자, 금액, 날짜, 기관명은 구체적으로 씁니다.
- 반드시 한국어로 작성합니다.

# 정형화 방지 (필수)
학습 데이터에서 가장 치명적인 결함은 **유형마다 같은 문구가 반복되는 것**입니다.
모델이 SR유형이 아니라 그 문구를 외우게 되고, 실제 통화에서는 그런 말이 나오지
않으므로 현장에서 무너집니다. 다음을 반드시 지키십시오.

- 상담사의 처리 안내(접수·이관·확인)를 **매번 다른 말로** 표현하십시오.
  같은 유형이라도 어떤 통화는 담당 부서를 언급하고, 어떤 통화는 "알아보고
  연락드리겠다"로만 끝나고, 어떤 통화는 처리 방법을 알려주고 끝납니다.
- 통화의 **20% 정도는 깔끔하게 끝나지 않게** 하십시오. 원인을 특정하지 못한 채
  재연락을 약속하거나, 고객이 먼저 끊거나, 다른 부서로 넘기는 식입니다.
- 고객이 항상 협조적일 필요는 없습니다. 짜증, 반복 질문, 딴 얘기, 말 끊기가
  섞여야 실제 통화에 가깝습니다.
- 판정 근거는 **상담사의 요약 멘트가 아니라 대화 속 사실**로 드러내십시오.
  ("다른 PC에서도 그렇다"는 사실이 근거이지, "시스템 문제입니다"라는 선언이
  근거가 아닙니다.)"""


def build_system_prompt(banlist: dict[str, list[str]] | None = None) -> str:
    """시스템 프롬프트를 만든다.

    금지어가 있으면 뒤에 덧붙인다. 한 번의 생성 실행 안에서는 값이 고정이므로
    프롬프트 캐시는 그대로 적용된다.
    """
    bl = load_banlist() if banlist is None else banlist
    return _SYSTEM_PROMPT_BASE + _banlist_section(bl)


# 기본 시스템 프롬프트 (import 시점의 금지어 반영)
SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------------------
# 유저 프롬프트 — 셀마다 달라지는 부분
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenSpec:
    """생성 격자의 한 셀."""

    target_label: str          # 최종 SR유형 (목표)
    mid_category: str          # 업무 중분류
    sub_category: str          # 업무 소분류
    difficulty: str            # 일반 | 경계 | 복합
    caller_role: str
    seed_idx: int              # 같은 셀 내 변형 구분용
    secondary_label: str | None = None  # 경계/복합 케이스의 상대 유형

    @property
    def cell_id(self) -> str:
        sec = f"+{self.secondary_label}" if self.secondary_label else ""
        return (f"{self.target_label}{sec}|{self.sub_category}|"
                f"{self.difficulty}|{self.seed_idx}")


def build_user_prompt(spec: GenSpec) -> str:
    parts = [
        "# 이번 생성 조건",
        f"- 업무 중분류: {spec.mid_category}",
        f"- 업무 소분류: {spec.sub_category}",
        f"- 발신자: {spec.caller_role}",
        f"- **목표 SR유형: {spec.target_label}** (적기처리기간 {SLA_DAYS[spec.target_label]}일)",
        f"- 난이도: {spec.difficulty}",
        "",
    ]

    if spec.difficulty == "일반":
        parts += [
            "## 난이도 지시 — 일반",
            f"목표 유형 `{spec.target_label}` 하나만 명확하게 드러나는 대화를 작성합니다.",
            "판정이 갈릴 여지가 없어야 합니다. `sr_labels`는 목표 유형 1개만 포함합니다.",
        ]
    elif spec.difficulty == "경계":
        other = spec.secondary_label
        hint = next(
            (h for a, b, h in CONFUSION_PAIRS
             if {a, b} == {spec.target_label, other}),
            "두 유형을 가르는 결정적 단서를 대화 안에 정확히 하나만 배치한다.",
        )
        parts += [
            "## 난이도 지시 — 경계",
            f"`{spec.target_label}`와 `{other}` 사이에서 헷갈리도록 작성하되, "
            f"최종 정답은 반드시 `{spec.target_label}`가 되어야 합니다.",
            f"판정 지침: {hint}",
            f"- 초반에는 `{other}`처럼 보이게 전개합니다.",
            f"- 중반 이후 결정적 단서 하나로 `{spec.target_label}`임이 확정되게 합니다.",
            "- 그 단서는 대화 속 사실(상담사의 확인 질문과 고객의 답변)로만 드러냅니다.",
            f"- `sr_labels`에는 `{spec.target_label}` 1개만 넣습니다. "
            f"`{other}`는 배제된 가설이므로 넣지 않습니다.",
        ]
    else:  # 복합
        other = spec.secondary_label
        parts += [
            "## 난이도 지시 — 복합",
            f"한 통화에 `{spec.target_label}`와 `{other}` **두 용건이 모두 실제로** "
            "등장하는 대화를 작성합니다.",
            "- 두 용건은 서로 다른 사안이어야 합니다. 하나가 다른 하나의 원인이면 안 됩니다.",
            "- 각 용건마다 판정 근거 발화가 분명히 존재해야 합니다.",
            f"- `sr_labels`에는 `{spec.target_label}`와 `{other}` 를 모두 넣습니다.",
            f"- 참고: 적기처리기간 규칙에 따라 최종 SR유형은 `{spec.target_label}`로 "
            "축약될 조건입니다.",
        ]

    parts += [
        "",
        "## 다양성 요구",
        f"이 조건으로 이미 {spec.seed_idx}건을 생성했습니다. "
        "이전 건들과 첫 문장, 전개 방식, 화자의 태도가 겹치지 않게 작성하십시오.",
        "",
        "위 조건에 맞는 상담 통화 1건을 생성하십시오.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 격자 생성
# ---------------------------------------------------------------------------
def _pick_partner(label: str, rng: random.Random) -> str:
    """혼동쌍에 정의된 상대 유형을 우선 선택. 없으면 임의 선택."""
    partners = [b for a, b, _ in CONFUSION_PAIRS if a == label]
    partners += [a for a, b, _ in CONFUSION_PAIRS if b == label]
    if partners:
        return rng.choice(partners)
    return rng.choice([l for l in LABELS if l != label and l != "기타"])


def build_grid(per_cell: int = 3, seed: int = 0) -> list[GenSpec]:
    """생성 격자를 만든다.

    `기타`는 업무 주제와 무관한 유형이므로 소분류 축을 붙이지 않고 별도 물량으로
    처리한다. 나머지 7종은 29개 소분류 전부와 교차한다.
    """
    rng = random.Random(seed)
    specs: list[GenSpec] = []

    topical = [l for l in LABELS if l != "기타"]
    for mid, sub in SUBCATEGORIES:
        for label in topical:
            for i in range(per_cell):
                diff = rng.choices(
                    list(DIFFICULTY_MIX), weights=list(DIFFICULTY_MIX.values())
                )[0]
                secondary = _pick_partner(label, rng) if diff != "일반" else None
                specs.append(GenSpec(
                    target_label=label,
                    mid_category=mid,
                    sub_category=sub,
                    difficulty=diff,
                    caller_role=rng.choice(CALLER_ROLES),
                    seed_idx=i,
                    secondary_label=secondary,
                ))

    # 기타: 전체의 약 4% 물량, 주제 무관
    n_etc = max(1, int(len(specs) * 0.04))
    for i in range(n_etc):
        mid, sub = rng.choice(SUBCATEGORIES)
        specs.append(GenSpec(
            target_label="기타",
            mid_category=mid,
            sub_category=sub,
            difficulty="일반",
            caller_role=rng.choice(CALLER_ROLES),
            seed_idx=i,
        ))

    rng.shuffle(specs)
    return specs


if __name__ == "__main__":
    grid = build_grid(per_cell=3, seed=0)
    print(f"소분류 {len(SUBCATEGORIES)}종, 격자 {len(grid)}셀")

    from collections import Counter
    print("라벨 분포 :", Counter(s.target_label for s in grid).most_common())
    print("난이도 분포:", Counter(s.difficulty for s in grid).most_common())
    print("\n--- 예시 프롬프트 (경계) ---")
    boundary = next(s for s in grid if s.difficulty == "경계")
    print(build_user_prompt(boundary))
