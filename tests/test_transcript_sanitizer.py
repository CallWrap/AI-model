from callwrap_ai.schemas import Speaker, Transcript, Utterance
from callwrap_ai.transcript_sanitizer import sanitize_text, sanitize_transcript


def test_sanitize_text_masks_profanity_variants_without_masking_benign_words():
    text = "시발점은 정상 단어지만 씨발, 씨*발, ㅆㅂ과 (욕설)은 가려야 함"

    assert sanitize_text(text) == (
        "시발점은 정상 단어지만 [비속어], [비속어], [비속어]과 [비속어]은 가려야 함"
    )


def test_sanitize_transcript_returns_copy_with_utterance_metadata_preserved():
    original = Transcript(
        utterances=[
            Utterance(
                index=7,
                speaker=Speaker.CUSTOMER,
                text="이런 씨발 같은 시스템",
                start=1.25,
                end=2.75,
                confidence=0.91,
            )
        ]
    )

    sanitized = sanitize_transcript(original)

    assert sanitized is not original
    assert sanitized.utterances[0] is not original.utterances[0]
    assert sanitized.utterances[0].text == "이런 [비속어] 같은 시스템"
    assert sanitized.utterances[0].model_dump(exclude={"text"}) == original.utterances[0].model_dump(
        exclude={"text"}
    )
    assert original.utterances[0].text == "이런 씨발 같은 시스템"
