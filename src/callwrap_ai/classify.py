"""F2 분류 — 중분류·소분류 checkpoint 로딩과 예측. 소유: 김성진.

**모델은 프로세스당 한 번만 올린다** (명세 4.1). 요청마다 400MB씩 읽으면 서버가 죽는다.
모듈 레벨 지연 싱글턴으로 구현했다.

**torch·transformers를 파일 맨 위에서 import하지 않는다.** 기본 설치(`uv sync`)에는
이 둘이 없다 — 학습용이라 `train` extra로 빠져 있다. 위에서 import하면 서버 파트가
`from callwrap_ai import analyze` 만 해도 깨진다. 함수 안에서 늦게 불러오고, 실패하면
None을 돌려 부분 실패로 넘긴다.

checkpoint 경로는 `CALLWRAP_MODEL_DIR` 환경변수로 바꿀 수 있다. 기본값은 저장소의
`data/models/` 인데 이건 개발용이고, 서버 배포 시에는 환경변수로 지정한다.
"""

import os
from functools import lru_cache
from pathlib import Path

from callwrap_ai.preprocess import strip_boilerplate
from callwrap_ai.schemas import SUBCATEGORIES_BY_CATEGORY

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data/models"
CATEGORY_CKPT = "category-klue-roberta-base-seed42"
SUBCATEGORY_CKPT = "subcategory-klue-roberta-base-seed42"
MAX_LENGTH = 512  # 학습과 같은 값. 바꾸면 결과가 달라진다


def model_dir() -> Path:
    return Path(os.environ.get("CALLWRAP_MODEL_DIR", DEFAULT_MODEL_DIR))


class _Classifier:
    """checkpoint 하나를 감싼다. 라벨 순서는 config.json(id2label)에서 읽는다 —
    계약(schemas.py)의 enum 순서가 바뀌어도 학습된 모델은 자기 매핑을 그대로 쓴다."""

    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    def probs(self, text: str):
        """대본 -> 클래스별 확률. 전처리는 학습과 동일하게 여기서 건다."""
        enc = self.tok(strip_boilerplate(text), truncation=True, max_length=MAX_LENGTH,
                       return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device)).logits
        return logits.softmax(-1)[0].cpu()


@lru_cache(maxsize=None)
def _load(ckpt: str) -> _Classifier | None:
    """싱글턴. torch가 없거나 checkpoint가 없으면 None — 호출부가 부분 실패로 처리한다."""
    path = model_dir() / ckpt
    if not (path / "config.json").exists():
        return None
    try:
        return _Classifier(path)
    except ImportError:
        return None  # torch/transformers 미설치 (기본 설치 환경)


def predict_category(text: str) -> tuple[str, float] | None:
    """대본 -> (중분류, 확률). 모델이 없으면 None."""
    clf = _load(CATEGORY_CKPT)
    if clf is None:
        return None
    p = clf.probs(text)
    i = int(p.argmax())
    return clf.labels[i], float(p[i])


def predict_subcategory(text: str, category: str | None) -> tuple[str, float] | None:
    """대본 + 중분류 -> (소분류, 확률).

    **중분류가 주어지면 그 안의 소분류만 후보로 남긴다.** 29지선다가 2~9지선다가 되고,
    계층 위반이 원천적으로 나올 수 없다 — PR #11(B안)에서 AI로 넘어온 정합 검증 책임을
    구조가 자동으로 만족시킨다.

    중분류를 모르면 마스킹 없이 29종에서 고른다.
    """
    clf = _load(SUBCATEGORY_CKPT)
    if clf is None:
        return None
    p = clf.probs(text)

    allowed = None
    if category:
        for cat, subs in SUBCATEGORIES_BY_CATEGORY.items():
            if cat.value == category:
                allowed = {s.value for s in subs}
                break
    if allowed:
        masked = [(v, i) for i, v in enumerate(p.tolist()) if clf.labels[i] in allowed]
        if masked:
            score, i = max(masked)
            return clf.labels[i], float(score)
    i = int(p.argmax())
    return clf.labels[i], float(p[i])


def is_ready() -> bool:
    """두 모델이 다 올라오는지. 서버 기동 시 확인용."""
    return _load(CATEGORY_CKPT) is not None and _load(SUBCATEGORY_CKPT) is not None
