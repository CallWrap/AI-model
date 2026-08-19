"""F1 요약·F4 후속처리·FAQ필요 production 모듈 검증."""

import json

import callwrap_ai.pipeline as pipeline
import callwrap_ai.summary as summary_mod
from callwrap_ai.schemas import Judgment, Speaker, Summary, Transcript, Utterance


def make_transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(index=0, speaker=Speaker.COUNSELOR, text="무엇을 도와드릴까요?",
                      start=0.0, end=1.0),
            Utterance(index=1, speaker=Speaker.CUSTOMER, text="본인부담금은 어떻게 내나요?",
                      start=1.0, end=2.0),
            Utterance(index=2, speaker=Speaker.COUNSELOR, text="가상계좌로 납부하시면 됩니다.",
                      start=2.0, end=3.0),
        ]
    )


def valid_response() -> str:
    return json.dumps(
        {
            "title": "본인부담금 납부 문의",
            "inquiry": "수급자가 본인부담금 납부 방법을 문의함.",
            "resolution": "가상계좌로 납부하도록 안내함.",
            "needs_followup": False,
            "needs_faq": True,
            "evidence_indices": [1, 2, 99],
        },
        ensure_ascii=False,
    )


def test_summarize_returns_three_judgments(monkeypatch):
    seen: dict[str, str] = {}

    def fake_ask(prompt: str, model: str) -> str:
        seen["prompt"] = prompt
        seen["model"] = model
        return valid_response()

    monkeypatch.setattr(summary_mod, "_ask", fake_ask)
    result = summary_mod.summarize(make_transcript())

    assert result is not None
    summary, followup, faq = result
    assert summary.value.title == "본인부담금 납부 문의"
    assert followup.value is False
    assert faq.value is True
    assert summary.confidence == followup.confidence == faq.confidence == 0.0
    assert summary.evidence_indices == followup.evidence_indices == faq.evidence_indices == [1, 2]
    assert "[1] 고객: 본인부담금은 어떻게 내나요?" in seen["prompt"]
    assert seen["model"] == "exaone3.5:7.8b"


def test_summarize_rejects_invalid_response(monkeypatch):
    monkeypatch.setattr(summary_mod, "_ask", lambda *_: '{"title": "필드 부족"}')
    assert summary_mod.summarize(make_transcript()) is None


def test_summarize_ignores_invalid_evidence_indices(monkeypatch):
    response = json.loads(valid_response())
    response["evidence_indices"] = None
    monkeypatch.setattr(summary_mod, "_ask", lambda *_: json.dumps(response, ensure_ascii=False))

    result = summary_mod.summarize(make_transcript())

    assert result is not None
    assert result[0].evidence_indices == []


def test_summarize_returns_none_when_ollama_is_unavailable(monkeypatch):
    monkeypatch.setattr(summary_mod, "_ask", lambda *_: None)
    assert summary_mod.summarize(make_transcript()) is None


def test_summarize_does_not_retry_empty_response(monkeypatch):
    calls = 0

    def empty_response(*_):
        nonlocal calls
        calls += 1
        return ""

    monkeypatch.setattr(summary_mod, "_ask", empty_response)

    assert summary_mod.summarize(make_transcript()) is None
    assert calls == 1


def test_ask_rejects_non_string_response(monkeypatch):
    class FakeResponse:
        def read(self) -> bytes:
            return b'{"response": []}'

    monkeypatch.setattr(summary_mod.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    assert summary_mod._ask("prompt", "model") is None


def test_summarize_returns_none_when_prompt_cannot_be_loaded(monkeypatch):
    def broken_prompt() -> str:
        raise FileNotFoundError("prompt missing")

    monkeypatch.setattr(summary_mod, "_prompt", broken_prompt)
    assert summary_mod.summarize(make_transcript()) is None


def test_prompt_path_change_is_applied_without_restart(monkeypatch, tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("설명\n## 프롬프트\n첫 번째 {dialog}", encoding="utf-8")
    second.write_text("설명\n## 프롬프트\n두 번째 {dialog}", encoding="utf-8")
    if hasattr(summary_mod._prompt, "cache_clear"):
        summary_mod._prompt.cache_clear()

    monkeypatch.setenv("CALLWRAP_SUMMARY_PROMPT", str(first))
    assert summary_mod._prompt() == "첫 번째 {dialog}"

    monkeypatch.setenv("CALLWRAP_SUMMARY_PROMPT", str(second))
    assert summary_mod._prompt() == "두 번째 {dialog}"


def test_pipeline_uses_summary_result(monkeypatch):
    transcript = make_transcript()
    summary = Judgment(
        value=Summary(title="제목", inquiry="문의", resolution="처리"),
        confidence=0.0,
        evidence_indices=[1, 2],
    )
    followup = Judgment(value=False, confidence=0.0, evidence_indices=[1, 2])
    faq = Judgment(value=True, confidence=0.0, evidence_indices=[1, 2])

    monkeypatch.setattr(summary_mod, "summarize", lambda _: (summary, followup, faq))
    monkeypatch.setattr(pipeline.classify, "predict_category", lambda _: None)
    monkeypatch.setattr(pipeline.classify, "predict_subcategory", lambda *_: None)
    monkeypatch.setattr(pipeline.grade_mod, "judge", lambda _: None)

    result = pipeline.analyze(transcript)

    assert result.summary == summary
    assert result.needs_followup == followup
    assert result.needs_faq == faq
    assert "summarize" not in result.failures
    assert "needs_followup" not in result.failures
    assert "needs_faq" not in result.failures
