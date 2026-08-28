import json

import callwrap_ai.faq_validate as validator


CASES = [
    {"inquiry": "[비속어] 본인부담금 납부 방법을 문의함.", "resolution": "가상계좌 납부를 안내함."},
    {"inquiry": "본인부담금 입금 계좌를 문의함.", "resolution": "지정 가상계좌를 안내함."},
]
CANDIDATE = [
    {"question": "본인부담금은 어떻게 내나요?", "answer": "지정된 가상계좌로 납부합니다.", "source_indices": [0, 1]}
]


def test_faq_validation_returns_grounded_items(monkeypatch):
    seen = {}

    def fake_ask(prompt, model):
        seen.update(prompt=prompt, model=model)
        return json.dumps({"items": CANDIDATE}, ensure_ascii=False)

    monkeypatch.setattr(validator, "_ask", fake_ask)

    result = validator.validate_faq("본인부담금입금", CASES, CANDIDATE)

    assert result == CANDIDATE
    assert seen["model"] == "gemma3:12b"
    assert "복원" in seen["prompt"]


def test_faq_validation_can_reject_all_candidates(monkeypatch):
    monkeypatch.setattr(validator, "_ask", lambda *_: '{"items": []}')
    assert validator.validate_faq("본인부담금입금", CASES, CANDIDATE) == []


def test_faq_validation_rejects_malformed_response(monkeypatch):
    monkeypatch.setattr(validator, "_ask", lambda *_: '{"question": "items 없음"}')
    assert validator.validate_faq("본인부담금입금", CASES, CANDIDATE) is None
