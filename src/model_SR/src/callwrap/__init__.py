"""CallWrap — 상담 SR유형 분류 모델 (B안).

STT + 화자분리된 상담 통화 전문을 입력받아 SR유형 8종 중 1종을 예측한다.
라벨 체계와 판정 기준은 docs/label_definition.md 를 따른다.
"""

from .labels import (LABELS, NUM_LABELS, PRIORITY, SLA_DAYS, collapse,
                     collapse_probs, is_sla_violation)

__version__ = "0.1.0"

__all__ = [
    "LABELS",
    "NUM_LABELS",
    "PRIORITY",
    "SLA_DAYS",
    "collapse",
    "collapse_probs",
    "is_sla_violation",
]
