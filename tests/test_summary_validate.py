import json

import callwrap_ai.summary_validate as validator
from callwrap_ai.schemas import Judgment, Speaker, Summary, Transcript, Utterance


def transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(index=0, speaker=Speaker.CUSTOMER, text="[비속어] 결제가 왜 안 돼요?", start=0, end=1),
            Utterance(index=1, speaker=Speaker.COUNSELOR, text="가상계좌를 확인해 주세요.", start=1, end=2),
        ]
    )


def candidate():
    return (
        Judgment(
            value=Summary(title="결제 문의", inquiry="결제 오류를 문의함.", resolution="가상계좌 확인을 안내함."),
            confidence=0.8,
            evidence_indices=[0, 1],
        ),
        Judgment(value=False, confidence=0.7, evidence_indices=[0, 1]),
    )


def test_validation_pass_keeps_exaone_candidate(monkeypatch):
    seen = {}

    def fake_ask(prompt, model):
        seen.update(prompt=prompt, model=model)
        return '{"summary_valid": true, "needs_faq_agrees": true}'

    monkeypatch.setattr(validator, "_ask", fake_ask)

    result = validator.validate(transcript(), candidate())

    assert result == candidate()
    assert seen["model"] == "gemma3:12b"
    assert "[비속어]" in seen["prompt"]
    assert "복원" in seen["prompt"]


def test_validation_correction_returns_grounded_replacement(monkeypatch):
    corrected = {
        "summary_valid": False,
        "needs_faq_agrees": False,
        "title": "결제 확인 문의",
        "inquiry": "결제가 되지 않아 원인을 문의함.",
        "resolution": "가상계좌 상태를 확인하도록 안내함.",
        "evidence_indices": [0, 1, 99],
    }
    monkeypatch.setattr(validator, "_ask", lambda *_: json.dumps(corrected, ensure_ascii=False))

    summary, faq = validator.validate(transcript(), candidate())

    assert summary.value.title == "결제 확인 문의"
    assert summary.confidence == 0.0
    assert summary.evidence_indices == [0, 1]
    assert faq == candidate()[1]


def test_validation_warns_on_faq_disagreement_without_overriding_exaone(monkeypatch, caplog):
    monkeypatch.setattr(
        validator,
        "_ask",
        lambda *_: '{"summary_valid": true, "needs_faq_agrees": false}',
    )

    result = validator.validate(transcript(), candidate())

    assert result == candidate()
    assert "needs_faq" in caplog.text


def test_validation_rejects_incomplete_correction(monkeypatch):
    monkeypatch.setattr(
        validator,
        "_ask",
        lambda *_: '{"summary_valid": false, "needs_faq_agrees": true, "title": "부족"}',
    )
    assert validator.validate(transcript(), candidate()) is None
