"""callwrap_ai — Transcript를 받아 후처리 폼을 채운다. 외부 API는 analyze() 하나. (명세 4.1)

    from callwrap_ai import analyze
    result = analyze(transcript)   # Transcript -> AnalysisResult
"""

from callwrap_ai.pipeline import analyze
from callwrap_ai.schemas import (
    AnalysisResult,
    ConsultationCategory,
    ConsultationGrade,
    ConsultationType,
    Judgment,
    Speaker,
    Summary,
    Transcript,
    Utterance,
)

__all__ = [
    "analyze",
    "AnalysisResult",
    "ConsultationCategory",
    "ConsultationGrade",
    "ConsultationType",
    "Judgment",
    "Speaker",
    "Summary",
    "Transcript",
    "Utterance",
]
