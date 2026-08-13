"""SR유형 라벨 체계와 우선순위 축약 로직.

정의서: docs/label_definition.md
출처:   99_archive/...참고자료.xlsx (SR유형 / 적기처리기간 시트)

핵심 규칙 (참고자료 명시):
    "SR유형 설정은 적기처리기간이 긴 항목을 우선하여 설정"

동점 tie-break는 정의서 §1.1에서 확정했다.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# 라벨 순서 = 모델 출력 헤드의 인덱스 순서. 절대 바꾸지 말 것.
LABELS: tuple[str, ...] = (
    "사용법문의",
    "시스템오류",
    "고객PC환경",
    "자료확인",
    "자료보정",
    "정책 및 지침문의",
    "기능개선",
    "기타",
)

NUM_LABELS = len(LABELS)
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}

# 적기처리기간(일). 참고자료 원본 값.
SLA_DAYS: dict[str, int] = {
    "사용법문의": 1,
    "시스템오류": 30,
    "고객PC환경": 3,
    "자료확인": 2,
    "자료보정": 5,
    "정책 및 지침문의": 5,
    "기능개선": 30,
    "기타": 3,
}

# 축약 우선순위 (앞이 높음). SLA 내림차순 + 정의서 §1.1 tie-break.
#   시스템오류 > 기능개선      (30일 동점: 업무 중단이 요구사항보다 우선)
#   자료보정   > 정책 및 지침문의 (5일 동점: 쓰기 작업 발생 건이 우선)
#   고객PC환경 > 기타          (3일 동점: 기타는 항상 최하위 fallback)
PRIORITY: tuple[str, ...] = (
    "시스템오류",
    "기능개선",
    "자료보정",
    "정책 및 지침문의",
    "고객PC환경",
    "자료확인",
    "사용법문의",
    "기타",
)

PRIORITY_RANK = {name: rank for rank, name in enumerate(PRIORITY)}

FALLBACK_LABEL = "기타"

assert set(PRIORITY) == set(LABELS) == set(SLA_DAYS)


# --------------------------------------------------------------------------
# 우선순위 축약
# --------------------------------------------------------------------------
def collapse(labels: Iterable[str]) -> str:
    """멀티라벨 집합 -> 최종 SR유형 1개.

    적기처리기간이 긴 쪽을 우선한다. 비어 있으면 '기타'.
    """
    candidates = [l for l in labels if l in PRIORITY_RANK]
    if not candidates:
        return FALLBACK_LABEL
    return min(candidates, key=lambda l: PRIORITY_RANK[l])


def collapse_probs(
    probs: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
    *,
    return_detail: bool = False,
):
    """모델 확률 -> 최종 SR유형.

    1) 클래스별 임계값을 넘은 라벨을 후보로 모은다.
    2) 후보 중 우선순위 최상위를 선택한다.
    3) 후보가 없으면 확률 최댓값 라벨을 쓰되, '기타'가 아닌 라벨만 인정.
       모두 미달이면 '기타'.

    return_detail=True면 (최종라벨, 부가정보 dict)를 반환한다.
    부가정보의 needs_review는 상담사 확인 큐로 보낼지 판단하는 플래그다.
    """
    probs = np.asarray(probs, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if probs.shape != thresholds.shape:
        raise ValueError(f"shape mismatch: probs {probs.shape} vs thr {thresholds.shape}")

    hits = [LABELS[i] for i in range(NUM_LABELS) if probs[i] >= thresholds[i]]

    if hits:
        # 동일 우선순위는 없으므로(PRIORITY는 전순서) min으로 유일하게 결정된다.
        final = min(hits, key=lambda l: PRIORITY_RANK[l])
        source = "threshold"
    else:
        # 아무 임계값도 못 넘음 = 모델이 자신 없음. 확률 최대 라벨로 대체.
        top = int(np.argmax(probs))
        final = LABELS[top]
        source = "argmax_fallback"

    if not return_detail:
        return final

    order = np.argsort(-probs)
    top3 = [(LABELS[i], float(probs[i])) for i in order[:3]]
    margin = float(probs[order[0]] - probs[order[1]]) if NUM_LABELS > 1 else 1.0

    detail = {
        "final": final,
        "source": source,
        "hits": hits,
        "top3": top3,
        "margin": margin,
        # 상담사 확인 필요 조건:
        #   - 임계값을 아무도 못 넘었거나
        #   - 1~2위 확률 차가 작거나
        #   - 상위 SLA 라벨과 하위 SLA 라벨이 동시에 후보에 올라 규칙으로만 갈렸을 때
        "needs_review": (
            source == "argmax_fallback"
            or margin < 0.15
            or (len(hits) >= 2 and SLA_DAYS[hits[0]] != SLA_DAYS[final])
        ),
        "sla_days": SLA_DAYS[final],
    }
    return final, detail


def collapse_batch(
    probs: np.ndarray, thresholds: Sequence[float] | np.ndarray
) -> list[str]:
    """(N, NUM_LABELS) 확률 행렬 -> 최종 라벨 리스트."""
    return [collapse_probs(row, thresholds) for row in np.asarray(probs)]


# --------------------------------------------------------------------------
# 인코딩
# --------------------------------------------------------------------------
def to_multihot(labels: Iterable[str]) -> np.ndarray:
    """라벨 이름 집합 -> (NUM_LABELS,) float32 멀티핫."""
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for name in labels:
        idx = LABEL2ID.get(name)
        if idx is None:
            raise KeyError(f"알 수 없는 SR유형: {name!r} (정의서 §1 참조)")
        vec[idx] = 1.0
    return vec


def from_multihot(vec: Sequence[float] | np.ndarray) -> list[str]:
    return [LABELS[i] for i, v in enumerate(np.asarray(vec)) if v > 0.5]


# --------------------------------------------------------------------------
# 학습용 가중치
# --------------------------------------------------------------------------
def priority_pos_weight(base: float = 1.0, span: float = 0.5) -> np.ndarray:
    """우선순위 상위 클래스에 더 큰 pos_weight를 부여한다.

    정의서 §1.2의 비대칭 오류비용을 손실함수에 반영하기 위한 것.
    상위 클래스(시스템오류)를 놓치면 SLA 위반이므로 재현율을 우선한다.

    주의: class_balance_pos_weight와 곱해서 쓸 경우 span을 크게 잡지 말 것.
    class_balance_pos_weight는 neg/pos 비율로 손실의 균형점을 정확히 중립으로
    맞추는데, 여기에 곱하면 그 중립점이 다시 깨져 여러 클래스가 동시에 "일단
    positive" 방향으로 편향된다. 실측(span=1.5, clip=10)으로 결합 가중치가
    최대 16배까지 벌어지면, 학습 초반 다섯 개 클래스가 동시에 임계값 0.5를
    넘어 collapse_probs의 우선순위 로직이 매번 최상위 클래스 하나로 붕괴시켰다.
    span 기본값을 0.5로 낮춘 것은 그 재현 결과를 반영한 것이다.

    반환: (NUM_LABELS,) float32. LABELS 인덱스 순서.
    """
    w = np.empty(NUM_LABELS, dtype=np.float32)
    last = NUM_LABELS - 1
    for name in LABELS:
        rank = PRIORITY_RANK[name]  # 0 = 최상위
        w[LABEL2ID[name]] = base + span * (last - rank) / last
    return w


def class_balance_pos_weight(y: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """클래스 불균형 보정용 pos_weight = neg/pos. priority 가중과 곱해서 쓴다.

    clip 기본값을 4.0으로 낮췄다 (이전 10.0). 표본이 적은 클래스(예: positive
    7~8건)는 neg/pos가 쉽게 8~10배까지 뛰는데, priority_pos_weight와 곱하면
    16배를 넘어가 pos_weight가 사실상 "무조건 positive로 예측하라"는 신호가
    되어버린다 (붕괴 재현 결과는 priority_pos_weight 문서 참조).
    """
    y = np.asarray(y, dtype=np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    w = np.divide(neg, np.maximum(pos, 1.0), dtype=np.float32)
    return np.clip(w, 1.0 / clip, clip).astype(np.float32)


# --------------------------------------------------------------------------
# SLA 위반 판정
# --------------------------------------------------------------------------
def is_sla_violation(true_label: str, pred_label: str) -> bool:
    """실제보다 짧은 적기처리기간으로 예측했는가?

    업무적으로 실제 손해가 발생하는 방향의 오분류만 센다.
    (긴 쪽으로 잘못 예측하는 것은 안전한 방향이므로 위반이 아니다.)
    """
    return SLA_DAYS[pred_label] < SLA_DAYS[true_label]


if __name__ == "__main__":
    # 정의서 §5 예시 검증
    assert collapse(["고객PC환경", "자료확인"]) == "고객PC환경"
    assert collapse(["사용법문의", "시스템오류"]) == "시스템오류"
    assert collapse(["자료확인", "자료보정"]) == "자료보정"
    assert collapse(["기타", "사용법문의"]) == "사용법문의"
    assert collapse([]) == "기타"
    assert is_sla_violation("시스템오류", "사용법문의") is True
    assert is_sla_violation("사용법문의", "시스템오류") is False

    print("PRIORITY :", " > ".join(PRIORITY))
    print("pos_weight:", dict(zip(LABELS, priority_pos_weight().round(3))))
    print("labels.py self-test OK")
