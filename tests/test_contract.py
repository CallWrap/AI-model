"""계약 검증 — analyze()의 입출력이 schemas.py 계약대로 동작하는지."""

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


def test_analyze_returns_analysis_result():
    result = analyze(make_transcript())
    assert isinstance(result, AnalysisResult)
    assert result.failures == []


def test_evidence_indices_point_to_real_utterances():
    transcript = make_transcript()
    result = analyze(transcript)
    valid = {u.index for u in transcript.utterances}
    judgments = [
        result.summary,
        result.consultation_type,
        result.consultation_category,
        result.consultation_subcategory,
        result.grade,
        result.needs_followup,
        result.needs_faq,
    ]
    for judgment in judgments:
        assert judgment is not None
        assert set(judgment.evidence_indices) <= valid


def test_subcategory_mapping_covers_all_and_is_disjoint():
    seen: set[ConsultationSubcategory] = set()
    for cat in ConsultationCategory:
        subs = SUBCATEGORIES_BY_CATEGORY[cat]
        assert not (seen & subs), f"{cat}: 소분류가 여러 중분류에 중복 소속"
        seen |= subs
    assert seen == set(ConsultationSubcategory)


def test_dummy_subcategory_consistent_with_category():
    result = analyze(make_transcript())
    assert result.consultation_category is not None
    assert result.consultation_subcategory is not None
    assert (
        result.consultation_subcategory.value
        in SUBCATEGORIES_BY_CATEGORY[result.consultation_category.value]
    )


def test_contract_roundtrips_as_json():
    transcript = make_transcript()
    assert Transcript.model_validate_json(transcript.model_dump_json()) == transcript

    result = analyze(transcript)
    assert AnalysisResult.model_validate_json(result.model_dump_json()) == result
