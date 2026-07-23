"""analyze() — 모듈 조립. 소유: 김성진.

Step 0 상태: 계약 검증용 하드코딩 더미를 반환한다.
api/는 Celery 태스크에서, web/은 화면에서 이 시그니처 그대로 사용을 시작하면 되고,
실제 분석 로직(Step 2~5)은 이 파일 내부에서만 교체된다.
"""

from callwrap_ai.schemas import (
    AnalysisResult,
    ConsultationCategory,
    ConsultationGrade,
    ConsultationSubcategory,
    ConsultationType,
    Judgment,
    Summary,
    Transcript,
)


def analyze(transcript: Transcript) -> AnalysisResult:
    # 더미도 실존하는 발화 index만 가리킨다 — web/이 하이라이트를 바로 테스트할 수 있게.
    # confidence는 3-tier(High/Medium/Low)가 전부 등장하도록 일부러 흩어놓았다.
    indices = [u.index for u in transcript.utterances]
    head, tail = indices[:2], indices[-2:]

    return AnalysisResult(
        summary=Judgment(
            value=Summary(
                title="(더미) 바우처 결제 오류 확인 요청",
                inquiry="(더미) 제공기관 담당자가 바우처 결제 처리 중 오류가 반복된다고 문의함.",
                resolution="(더미) 상담원이 결제 메뉴 경로와 재시도 방법을 안내하고 처리 완료를 확인함.",
            ),
            confidence=0.92,
            evidence_indices=head,
        ),
        consultation_type=Judgment(
            value=ConsultationType.USAGE, confidence=0.95, evidence_indices=head
        ),
        consultation_category=Judgment(
            value=ConsultationCategory.PAYMENT, confidence=0.72, evidence_indices=head
        ),
        consultation_subcategory=Judgment(
            value=ConsultationSubcategory.PAYMENT_METHOD_ERROR,
            confidence=0.58,
            evidence_indices=head,
        ),
        grade=Judgment(value=ConsultationGrade.NORMAL, confidence=0.55),
        needs_followup=Judgment(value=False, confidence=0.88, evidence_indices=tail),
        needs_faq=Judgment(value=True, confidence=0.66, evidence_indices=head),
        failures=[],
    )
