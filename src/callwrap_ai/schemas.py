"""입출력 계약 — 4명 공용. 변경 시 전원 리뷰. (명세 4.3, 4.4)

- `Transcript`: stt/ → (품질 게이트 ②) → ai/ 로 들어오는 유일한 입력
- `AnalysisResult`: ai/ → api/ → web/ 으로 나가는 유일한 출력
"""

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

# ── 입력 ──────────────────────────────────────────────


class Speaker(str, Enum):
    """발화자 역할. SPEAKER_00 같은 익명 라벨이 아니라 stt/가 역할로 해석한 값."""

    COUNSELOR = "상담사"
    CUSTOMER = "고객"


class Utterance(BaseModel):
    index: int = Field(ge=0, description="발화 순번. evidence_indices가 가리키는 대상")
    speaker: Speaker
    text: str
    start: float = Field(ge=0, description="발화 시작 시각(초)")
    end: float = Field(ge=0, description="발화 종료 시각(초)")
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="STT 신뢰도. 품질 게이트(②) 판정용"
    )


class Transcript(BaseModel):
    """ai/의 입력. 전사가 어디서 왔는지(Whisper/이노하이)는 모른다."""

    utterances: list[Utterance]


# ── 출력 ──────────────────────────────────────────────


class ConsultationType(str, Enum):
    """상담유형(SR유형) 8종 — 기관 참고자료(2026-07-21 수령)로 확정.

    유형 혼재 콜은 적기처리기간이 긴 항목을 우선 부여한다 (docs/label_spec.md).
    """

    USAGE = "사용법문의"
    SYSTEM_ERROR = "시스템오류"
    CUSTOMER_PC = "고객PC환경"
    DATA_CHECK = "자료확인"
    DATA_FIX = "자료보정"
    POLICY = "정책 및 지침문의"
    IMPROVEMENT = "기능개선"
    ETC = "기타"


class ConsultationCategory(str, Enum):
    """상담(업무)중분류 6종 — 기관 참고자료(2026-07-21) 기준.

    대분류는 바우처 사업명이며 본 과제 스코프는 '장애인활동지원'으로 고정이라
    필드로 두지 않는다. 참고자료 원문의 '결제및비용지금'은 명백한 오기이므로
    '결제및비용지급'으로 바로잡았다 (docs/decisions.md).
    """

    PAYMENT = "결제및비용지급"
    RECIPIENT = "대상자관리"
    FRAUD = "부정사용"
    PROVIDER_ORG = "제공기관관리"
    PROVIDER_STAFF = "제공인력관리"
    MUNICIPAL = "지자체사업관리"


class ConsultationSubcategory(str, Enum):
    """상담(업무)소분류 29종 — 기관 참고자료(2026-07-21) 기준.

    소속 중분류는 SUBCATEGORIES_BY_CATEGORY 참고. 성능 목표 지표는 중분류
    기준이며 소분류는 보조 출력이다 (docs/label_spec.md).
    """

    # 결제및비용지급
    SURCHARGE_RATE = "가산단가"
    PAYMENT_HISTORY = "결제내역"
    PAYMENT_METHOD_ERROR = "결제방법 및 오류"
    MISPAYMENT = "과오결제"
    TRANSPORT_FEE = "교통비 지급"
    PAYMENT_ETC = "기타"
    SERVICE_COST_PAYMENT = "서비스비용 지급"
    EXCEPTIONAL_PAYMENT = "예외적 결제"
    EXCEPTIONAL_DISBURSEMENT = "예외지급"
    # 대상자관리
    VIRTUAL_ACCOUNT = "가상계좌"
    RECIPIENT_INFO = "대상자정보"
    VOUCHER_USAGE = "바우처 사용 방법"
    VOUCHER_CREATION = "바우처생성"
    COPAY_DEPOSIT = "본인부담금입금"
    COPAY_REFUND = "본인부담금환급"
    APPLICATION_CRITERIA = "신청기준및절차"
    ELIGIBILITY_CHANGE = "자격변경"
    CAP_METHOD = "캡방식"
    # 부정사용
    REPORT_TARGET = "신고대상"
    REPORT_PROCEDURE = "신고절차"
    REWARD = "포상금"
    # 제공기관관리
    PROVIDER_OPERATION = "제공기관운영"
    PROVIDER_INFO = "제공기관정보"
    # 제공인력관리
    BENEFIT_REGISTRATION = "급여등록"
    WORKER_QUALIFICATION = "자격 및 교육"
    WORKER_INFO = "제공인력 정보"
    # 지자체사업관리
    MUNICIPAL_OPERATION = "사업운영 및 기존정보"
    DEPOSIT_MANAGEMENT = "예탁금관리"
    HAENGBOK_EUM_LINK = "행복e음 연계"


SUBCATEGORIES_BY_CATEGORY: dict[ConsultationCategory, frozenset[ConsultationSubcategory]] = {
    ConsultationCategory.PAYMENT: frozenset(
        {
            ConsultationSubcategory.SURCHARGE_RATE,
            ConsultationSubcategory.PAYMENT_HISTORY,
            ConsultationSubcategory.PAYMENT_METHOD_ERROR,
            ConsultationSubcategory.MISPAYMENT,
            ConsultationSubcategory.TRANSPORT_FEE,
            ConsultationSubcategory.PAYMENT_ETC,
            ConsultationSubcategory.SERVICE_COST_PAYMENT,
            ConsultationSubcategory.EXCEPTIONAL_PAYMENT,
            ConsultationSubcategory.EXCEPTIONAL_DISBURSEMENT,
        }
    ),
    ConsultationCategory.RECIPIENT: frozenset(
        {
            ConsultationSubcategory.VIRTUAL_ACCOUNT,
            ConsultationSubcategory.RECIPIENT_INFO,
            ConsultationSubcategory.VOUCHER_USAGE,
            ConsultationSubcategory.VOUCHER_CREATION,
            ConsultationSubcategory.COPAY_DEPOSIT,
            ConsultationSubcategory.COPAY_REFUND,
            ConsultationSubcategory.APPLICATION_CRITERIA,
            ConsultationSubcategory.ELIGIBILITY_CHANGE,
            ConsultationSubcategory.CAP_METHOD,
        }
    ),
    ConsultationCategory.FRAUD: frozenset(
        {
            ConsultationSubcategory.REPORT_TARGET,
            ConsultationSubcategory.REPORT_PROCEDURE,
            ConsultationSubcategory.REWARD,
        }
    ),
    ConsultationCategory.PROVIDER_ORG: frozenset(
        {
            ConsultationSubcategory.PROVIDER_OPERATION,
            ConsultationSubcategory.PROVIDER_INFO,
        }
    ),
    ConsultationCategory.PROVIDER_STAFF: frozenset(
        {
            ConsultationSubcategory.BENEFIT_REGISTRATION,
            ConsultationSubcategory.WORKER_QUALIFICATION,
            ConsultationSubcategory.WORKER_INFO,
        }
    ),
    ConsultationCategory.MUNICIPAL: frozenset(
        {
            ConsultationSubcategory.MUNICIPAL_OPERATION,
            ConsultationSubcategory.DEPOSIT_MANAGEMENT,
            ConsultationSubcategory.HAENGBOK_EUM_LINK,
        }
    ),
}


class ConsultationGrade(str, Enum):
    """상담등급 3종 (F3)."""

    PRAISE = "칭찬콜"
    ABUSIVE = "악성콜"
    NORMAL = "일반"


T = TypeVar("T")


class Judgment(BaseModel, Generic[T]):
    """값 + 신뢰도 + 근거 묶음. web/의 3-tier UI가 confidence로 렌더링을 가른다.

    3-tier 임계값(90/60)은 가정치이므로 스키마에 넣지 않고 설정으로 뺀다. (명세 4.5)
    """

    value: T
    confidence: float = Field(
        ge=0, le=1, description="캘리브레이션된 확률. LLM self-report 값 사용 금지 (명세 9.3)"
    )
    evidence_indices: list[int] = Field(
        default_factory=list, description="근거 발화의 Utterance.index. web/이 하이라이트"
    )


class Summary(BaseModel):
    title: str = Field(description="요청제목 — 상담시스템 폼의 제목 입력란용 한 줄")
    inquiry: str = Field(description="문의내용(요청내용) 요약 2~3문장 (고객 발화 기준)")
    resolution: str = Field(description="처리내용 요약 2~3문장 (상담사 발화 기준)")


class UtteranceSentiment(BaseModel):
    """발화 하나의 감정 점수. 상담등급 계산에는 고객 점수만 사용한다."""

    utterance_index: int = Field(ge=0, description="Transcript의 Utterance.index")
    speaker: Speaker
    score: float = Field(ge=-1, le=1, description="-1(매우 부정)~1(매우 긍정)")


class AnalysisResult(BaseModel):
    """analyze()의 반환값.

    부분 실패 허용 (명세 4.1): 실패한 모듈의 필드는 None으로 두고
    failures에 모듈명을 넣는다. 전체를 예외로 날리지 않는다.
    """

    summary: Judgment[Summary] | None
    consultation_type: Judgment[ConsultationType] | None
    consultation_category: Judgment[ConsultationCategory] | None
    consultation_subcategory: Judgment[ConsultationSubcategory] | None = Field(
        description="소분류 29종 — 보조 출력. 중분류와의 계층 정합은 SUBCATEGORIES_BY_CATEGORY로 검증"
    )
    grade: Judgment[ConsultationGrade] | None
    utterance_sentiments: list[UtteranceSentiment] = Field(
        default_factory=list, description="고객·상담사 발화별 감정 점수"
    )
    needs_followup: Judgment[bool] | None
    needs_faq: Judgment[bool] | None = Field(
        description="반복 문의로 판단되어 FAQ 신규 생성이 필요한지 (7/14 회의 확정 스코프)"
    )
    failures: list[str] = Field(
        default_factory=list, description='부분 실패한 모듈명. 예: ["summarize"]'
    )
