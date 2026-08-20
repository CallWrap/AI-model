"""analyze() — 모듈 조립. 소유: 김성진.

현재 상태: **F1(요약)·F2(상담유형·중분류·소분류)·F3(상담등급)·F4(후속처리)·FAQ 전부 실제 판정한다.**
요약·후속처리·FAQ필요는 EXAONE에 통화당 한 번 요청해 함께 받는다.
바깥에서 부르는 방법은 바뀌지 않으므로 api/·web/은 지금 형태 그대로 쓰면 된다.

**부분 실패를 허용한다** (명세 4.1). 모델이 없거나 ollama가 꺼져 있으면 해당 필드만
None으로 두고 failures에 모듈명을 넣는다. 전체를 예외로 날리지 않는다.

**소분류 정합은 구조로 보장된다.** 중분류가 정해지면 그 안의 소분류만 후보로 남기므로
어긋난 조합이 나올 수 없다. DB가 검증하지 않기로 했으므로(PR #11, B안) 이 책임이
AI 파트에 있는데, 마스킹이 그 요구를 자동으로 만족시킨다. 안전망으로 출력 직전에
한 번 더 검사한다.
"""

from callwrap_ai import classify, grade as grade_mod, sr_type, summary as summary_mod
from callwrap_ai.preprocess import utterances_to_text
from callwrap_ai.schemas import (
    SUBCATEGORIES_BY_CATEGORY,
    AnalysisResult,
    ConsultationCategory,
    ConsultationSubcategory,
    ConsultationType,
    Judgment,
    Speaker,
    Transcript,
)


def _customer_indices(t: Transcript) -> list[int]:
    return [u.index for u in t.utterances if u.speaker is Speaker.CUSTOMER]


def _check_hierarchy(category: str | None, subcategory: str | None) -> bool:
    """중분류에 속하지 않는 소분류면 False. 마스킹이 있어 정상적으로는 안 걸린다."""
    if category is None or subcategory is None:
        return False
    for cat, subs in SUBCATEGORIES_BY_CATEGORY.items():
        if cat.value == category:
            return subcategory in {s.value for s in subs}
    return False


def analyze(transcript: Transcript) -> AnalysisResult:
    failures: list[str] = []
    text = utterances_to_text(transcript.utterances)

    # --- 상담유형(SR유형 8종) ---
    type_judgment = None
    got = sr_type.predict_type(transcript)
    if got is None:
        failures.append("classify_type")
    else:
        type_value, conf = got
        type_judgment = Judgment(
            value=ConsultationType(type_value),
            confidence=conf,
            evidence_indices=_customer_indices(transcript)[:2],
        )

    # --- F2 중분류 ---
    cat_judgment = None
    cat_value = None
    got = classify.predict_category(text)
    if got is None:
        failures.append("classify_category")
    else:
        cat_value, conf = got
        cat_judgment = Judgment(
            value=ConsultationCategory(cat_value),
            confidence=round(conf, 4),
            evidence_indices=_customer_indices(transcript)[:2],
        )

    # --- F2 소분류 (중분류로 후보를 좁힌다) ---
    sub_judgment = None
    got = classify.predict_subcategory(text, cat_value)
    if got is None:
        failures.append("classify_subcategory")
    else:
        sub_value, conf = got
        if _check_hierarchy(cat_value, sub_value):
            sub_judgment = Judgment(
                value=ConsultationSubcategory(sub_value),
                confidence=round(conf, 4),
                evidence_indices=_customer_indices(transcript)[:2],
            )
        else:
            # 마스킹이 있어 정상적으로는 도달하지 않는다. 도달했다면 버그이므로
            # 어긋난 값을 내보내느니 비운다 (docs/decisions.md 2026-07-26).
            failures.append("subcategory_hierarchy")

    # --- F3 상담등급 (온디바이스 LLM 감정 점수 → 2축 규칙) ---
    grade_judgment = grade_mod.judge(transcript)
    if grade_judgment is None:
        failures.append("grade")

    # --- F1 요약·F4 후속처리·FAQ (EXAONE 1회 호출) ---
    summary_judgment = followup_judgment = faq_judgment = None
    summary_result = summary_mod.summarize(transcript)
    if summary_result is None:
        failures += ["summarize", "needs_followup", "needs_faq"]
    else:
        summary_judgment, followup_judgment, faq_judgment = summary_result

    return AnalysisResult(
        summary=summary_judgment,
        consultation_type=type_judgment,
        consultation_category=cat_judgment,
        consultation_subcategory=sub_judgment,
        grade=grade_judgment,
        needs_followup=followup_judgment,
        needs_faq=faq_judgment,
        failures=failures,
    )
