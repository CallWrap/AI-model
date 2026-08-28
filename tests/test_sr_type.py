"""상담유형(SR유형) 추론 모듈 — 모델 없이 계약과 실패 모드만 검증.

실제 예측값 검증은 여기서 하지 않는다. checkpoint(442MB)와 torch가 있어야 하고,
값의 품질은 model_SR 쪽 책임이다. 여기서 지키는 것은 두 가지다 —
checkpoint가 없을 때 조용히 부분 실패로 넘어가는가, 그리고 라벨이 계약과 맞는가.
"""

import callwrap_ai.pipeline as pipeline
import callwrap_ai.sr_type as sr_type
from callwrap_ai.schemas import ConsultationType, Speaker, Transcript, Utterance


def make_transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(index=0, speaker=Speaker.COUNSELOR, text="무엇을 도와드릴까요?",
                      start=0.0, end=1.0),
            Utterance(index=1, speaker=Speaker.CUSTOMER, text="저장 버튼이 안 눌립니다.",
                      start=1.0, end=2.0),
        ]
    )


def test_missing_checkpoint_is_partial_failure(monkeypatch, tmp_path):
    """checkpoint가 없으면 None — 예외를 던져 분석 전체를 죽이지 않는다."""
    monkeypatch.setenv("CALLWRAP_MODEL_DIR", str(tmp_path))
    sr_type._load.cache_clear()

    assert sr_type.predict_type(make_transcript()) is None
    assert sr_type.is_ready() is False
    sr_type._load.cache_clear()


def test_pipeline_records_failure_when_model_missing(monkeypatch):
    monkeypatch.setattr(sr_type, "predict_type", lambda _: None)
    monkeypatch.setattr(pipeline.classify, "predict_category", lambda _: None)
    monkeypatch.setattr(pipeline.classify, "predict_subcategory", lambda *_: None)
    monkeypatch.setattr(pipeline.grade_mod, "analyze_sentiments", lambda _: (None, [], False))
    monkeypatch.setattr(pipeline.summary_mod, "summarize", lambda _: None)

    result = pipeline.analyze(make_transcript())

    assert result.consultation_type is None
    assert "classify_type" in result.failures


def test_pipeline_uses_predicted_type(monkeypatch):
    monkeypatch.setattr(sr_type, "predict_type", lambda _: ("시스템오류", 0.83))
    monkeypatch.setattr(pipeline.classify, "predict_category", lambda _: None)
    monkeypatch.setattr(pipeline.classify, "predict_subcategory", lambda *_: None)
    monkeypatch.setattr(pipeline.grade_mod, "analyze_sentiments", lambda _: (None, [], False))
    monkeypatch.setattr(pipeline.summary_mod, "summarize", lambda _: None)

    result = pipeline.analyze(make_transcript())

    assert result.consultation_type.value is ConsultationType.SYSTEM_ERROR
    assert result.consultation_type.confidence == 0.83
    assert "classify_type" not in result.failures


def test_model_labels_match_the_contract():
    """model_SR의 라벨 8종이 schemas.ConsultationType과 문자열까지 같아야 한다.

    어긋나면 ConsultationType(label)이 ValueError로 터진다 — 배포 후가 아니라 여기서 걸린다.
    """
    from callwrap.labels import LABELS

    assert set(LABELS) == {t.value for t in ConsultationType}
