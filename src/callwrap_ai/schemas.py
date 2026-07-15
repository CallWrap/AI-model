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
    """상담유형 9종.

    기관 메타정보 표(7종)와 '25년 유형별 현황 표(8컬럼: 고객PC 환경·기타 포함)의
    합집합. 근거는 docs/label_spec.md 참고.
    """

    USAGE = "사용법 문의"
    DATA_CHECK = "자료 확인"
    DATA_FIX = "자료 보정"
    SYSTEM_ERROR = "시스템 오류"
    POLICY = "정책 및 지침 문의"
    IMPROVEMENT = "기능 개선"
    EXTERNAL_DATA_CHECK = "타 기관 자료 확인"
    CUSTOMER_PC = "고객PC 환경"
    ETC = "기타"


class ConsultationCategory(str, Enum):
    """상담분류 6종."""

    INDIVIDUAL = "개별업무"
    INTEGRATED = "통합업무"
    COMMON = "공통업무"
    TAILORED_BENEFIT = "맞춤형 급여"
    PAN_GOVERNMENT = "범정부"
    DW = "DW"


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
    inquiry: str = Field(description="문의내용 요약 2~3문장 (고객 발화 기준)")
    resolution: str = Field(description="처리내용 요약 2~3문장 (상담사 발화 기준)")


class AnalysisResult(BaseModel):
    """analyze()의 반환값.

    부분 실패 허용 (명세 4.1): 실패한 모듈의 필드는 None으로 두고
    failures에 모듈명을 넣는다. 전체를 예외로 날리지 않는다.
    """

    summary: Judgment[Summary] | None
    consultation_type: Judgment[ConsultationType] | None
    consultation_category: Judgment[ConsultationCategory] | None
    grade: Judgment[ConsultationGrade] | None
    needs_followup: Judgment[bool] | None
    needs_faq: Judgment[bool] | None = Field(
        description="반복 문의로 판단되어 FAQ 신규 생성이 필요한지 (7/14 회의 확정 스코프)"
    )
    failures: list[str] = Field(
        default_factory=list, description='부분 실패한 모듈명. 예: ["summarize"]'
    )
