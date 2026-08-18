"""계약 검증 — analyze()의 입출력이 schemas.py 계약대로 동작하는지.

**모델이 없는 환경에서도 통과해야 한다.** 서버 파트는 `uv sync`만 하므로 torch도
checkpoint도 없다. 그 환경에서 analyze()는 해당 필드를 None으로 두고 failures에
모듈명을 넣는다 (명세 4.1). 테스트는 값이 아니라 **계약을 지키는지**를 본다.
"""

from callwrap_ai import (
    SUBCATEGORIES_BY_CATEGORY,
    AnalysisResult,
    ConsultationCategory,
    ConsultationSubcategory,
    Speaker,
    Transcript,
    Utterance,
    analyze,
)


def make_transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(
                index=0,
                speaker=Speaker.COUNSELOR,
                text="안녕하세요, 무엇을 도와드릴까요?",
                start=0.0,
                end=2.1,
                confidence=0.98,
            ),
            Utterance(
                index=1,
                speaker=Speaker.CUSTOMER,
                text="소득 자료 보정은 어떻게 신청하나요?",
                start=2.4,
                end=5.0,
                confidence=0.95,
            ),
            Utterance(
                index=2,
                speaker=Speaker.COUNSELOR,
                text="자료 보정 요청 메뉴에서 신청하시면 됩니다.",
                start=5.2,
                end=8.0,
                confidence=None,
            ),
        ]
    )


# analyze()가 낼 수 있는 실패 모듈명. 오타로 엉뚱한 값이 들어가는 것을 막는다.
KNOWN_FAILURES = {
    "classify_category",
    "classify_subcategory",
    "subcategory_hierarchy",
    "grade",
    "summarize",
    "needs_followup",
    "needs_faq",
}

# 실패 모듈명 -> 그 실패가 비우는 필드
FAILURE_FIELD = {
    "classify_category": "consultation_category",
    "classify_subcategory": "consultation_subcategory",
    "subcategory_hierarchy": "consultation_subcategory",
    "grade": "grade",
    "needs_followup": "needs_followup",
    "needs_faq": "needs_faq",
}


def test_analyze_returns_analysis_result():
    result = analyze(make_transcript())
    assert isinstance(result, AnalysisResult)
    assert set(result.failures) <= KNOWN_FAILURES, f"모르는 실패 모듈: {result.failures}"


def test_failed_modules_do_not_pretend_to_have_answers():
    """failures에 오른 모듈은 필드가 비어 있거나 확신도가 0이어야 한다.

    실패했는데 그럴듯한 값이 남아 있으면 서버·화면이 그걸 진짜로 쓴다.
    summarize는 web/이 깨지지 않도록 자리표시 값을 두되 confidence=0으로 표시한다.
    """
    result = analyze(make_transcript())
    for module in result.failures:
        field = FAILURE_FIELD.get(module)
        if field:
            assert getattr(result, field) is None, f"{module} 실패인데 {field}에 값이 있다"
    if "summarize" in result.failures:
        assert result.summary is None or result.summary.confidence == 0.0


def test_evidence_indices_point_to_real_utterances():
    """근거 인덱스는 실존하는 발화를 가리켜야 한다. web/이 이걸로 원문을 하이라이트한다."""
    transcript = make_transcript()
    result = analyze(transcript)
    valid = {u.index for u in transcript.utterances}
    for judgment in (result.summary, result.consultation_type, result.consultation_category,
                     result.consultation_subcategory, result.grade,
                     result.needs_followup, result.needs_faq):
        if judgment is not None:   # 미구현·실패 모듈은 None이다
            assert set(judgment.evidence_indices) <= valid


def test_subcategory_mapping_covers_all_and_is_disjoint():
    seen: set[ConsultationSubcategory] = set()
    for cat in ConsultationCategory:
        subs = SUBCATEGORIES_BY_CATEGORY[cat]
        assert not (seen & subs), f"{cat}: 소분류가 여러 중분류에 중복 소속"
        seen |= subs
    assert seen == set(ConsultationSubcategory)


def test_subcategory_consistent_with_category():
    """소분류는 중분류에 속해야 한다. 추론 시 마스킹으로 보장되지만 안전망으로 확인한다.

    모델이 없는 환경에서는 둘 다 None이므로 검사를 건너뛴다.
    """
    result = analyze(make_transcript())
    if result.consultation_category is None or result.consultation_subcategory is None:
        return
    assert (
        result.consultation_subcategory.value
        in SUBCATEGORIES_BY_CATEGORY[result.consultation_category.value]
    )


def test_contract_roundtrips_as_json():
    transcript = make_transcript()
    assert Transcript.model_validate_json(transcript.model_dump_json()) == transcript

    result = analyze(transcript)
    assert AnalysisResult.model_validate_json(result.model_dump_json()) == result
