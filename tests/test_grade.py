from callwrap_ai import grade
from callwrap_ai.schemas import ConsultationGrade, Speaker, Transcript, Utterance


def make_transcript() -> Transcript:
    return Transcript(
        utterances=[
            Utterance(index=3, speaker=Speaker.COUNSELOR, text="확인하겠습니다.", start=0, end=1),
            Utterance(index=4, speaker=Speaker.CUSTOMER, text="왜 안 됩니까?", start=1, end=2),
            Utterance(index=8, speaker=Speaker.COUNSELOR, text="죄송합니다.", start=2, end=3),
            Utterance(index=9, speaker=Speaker.CUSTOMER, text="정말 화가 납니다.", start=3, end=4),
        ]
    )


def test_analyze_sentiments_maps_both_speakers_and_grades_from_customer_only(monkeypatch):
    monkeypatch.setattr(grade, "score_utterances", lambda texts, model=None: [-0.1, -0.8])
    monkeypatch.setattr(grade, "score_counselor_utterances", lambda texts, model=None: [0.9, 0.8])

    judgment, sentiments, counselor_ok = grade.analyze_sentiments(make_transcript())

    assert judgment.value is ConsultationGrade.ABUSIVE
    assert counselor_ok is True
    assert [(s.utterance_index, s.speaker, s.score) for s in sentiments] == [
        (3, Speaker.COUNSELOR, 0.9),
        (4, Speaker.CUSTOMER, -0.1),
        (8, Speaker.COUNSELOR, 0.8),
        (9, Speaker.CUSTOMER, -0.8),
    ]


def test_counselor_sentiment_failure_does_not_remove_customer_grade(monkeypatch):
    monkeypatch.setattr(grade, "score_utterances", lambda texts, model=None: [0.0, 0.1])
    monkeypatch.setattr(grade, "score_counselor_utterances", lambda texts, model=None: None)

    judgment, sentiments, counselor_ok = grade.analyze_sentiments(make_transcript())

    assert judgment.value is ConsultationGrade.NORMAL
    assert counselor_ok is False
    assert [(s.utterance_index, s.speaker) for s in sentiments] == [
        (4, Speaker.CUSTOMER),
        (9, Speaker.CUSTOMER),
    ]
