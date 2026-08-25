"""F2' 상담유형(SR유형) 8종 분류 — checkpoint 로딩과 예측. 모델 소유: 이재혁.

`classify.py`(중분류·소분류)와 같은 구조다 — 모듈 레벨 지연 싱글턴, torch/transformers는
함수 안에서 늦게 import, 실패하면 None을 돌려 부분 실패로 넘긴다.

**전처리는 학습 코드를 그대로 재사용한다.** 이 모델은 평문을 먹지 않는다:

- 발화마다 `[상담사]`/`[고객]` 스페셜 토큰이 앞에 붙는다 (vocab 32002 = 32000 + 2)
- 토큰마다 화자 id가 따로 들어간다 (`speaker_ids` — 모델에 화자 임베딩이 있다)
- 512토큰을 넘으면 앞뒤를 잘라 붙인다 (head+tail, head_ratio)

셋 중 하나만 어긋나도 성능이 조용히 무너진다. 그래서 재구현하지 않고 `callwrap.dataset`의
`encode_dialogue`·`strip_boilerplate`를 그대로 부른다 (pyproject가 그 패키지를 함께 담는다).

**멀티라벨이다.** 한 통화에 여러 유형이 섞이는 것이 정상이라 8-way sigmoid로 학습했고,
추론은 `collapse_probs()`가 "적기처리기간이 긴 항목 우선" 규칙으로 하나를 고른다.
그 규칙은 기관 참고자료에 명시된 것이라 코드로 100% 재현된다 — 적기처리기간이 개정되면
재학습 없이 후처리만 고치면 된다.

checkpoint 경로는 `classify.py`와 같은 `CALLWRAP_MODEL_DIR`을 쓴다.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from callwrap_ai.classify import DEVICE, model_dir

SR_CKPT = os.environ.get("CALLWRAP_SR_CKPT", "sr-type-klue-roberta-base-v2")


class _SRClassifier:
    """checkpoint 하나를 감싼다. 라벨 순서·임계값·백본 설정은 전부 config.json에서 읽는다 —
    학습 때 쓴 값과 어긋나면 로드가 실패하거나 결과가 조용히 틀어진다."""

    def __init__(self, path: Path):
        import torch
        from transformers import AutoTokenizer

        from callwrap.model import SRClassifier

        self._torch = torch
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.labels = config["labels"]
        self.thresholds = config["thresholds"]
        self.max_length = config.get("max_length", 512)
        self.head_ratio = config.get("head_ratio", 0.5)

        self.tok = AutoTokenizer.from_pretrained(path / "tokenizer")
        self.device = torch.device(DEVICE)
        model = SRClassifier(
            backbone_name=config["backbone"],
            num_labels=len(self.labels),
            dropout=config.get("dropout", 0.1),
            pooling=config.get("pooling", "cls_mean"),
            use_speaker_embedding=config.get("use_speaker_embedding", True),
            vocab_size=config.get("vocab_size"),
            pretrained=False,  # 추론만 함 — 아래서 state_dict로 덮어쓸 가중치를 미리 받을 필요 없다
        )
        model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))
        self.model = model.to(self.device).eval()

    def probs(self, transcript):
        """Transcript -> 라벨별 확률(sigmoid). 학습과 동일한 전처리를 태운다."""
        from callwrap.dataset import encode_dialogue, strip_boilerplate
        from callwrap.stt_noise import Utterance as SRUtterance

        utts = [SRUtterance(speaker=u.speaker.value, text=u.text) for u in transcript.utterances]
        stripped = strip_boilerplate(utts)
        enc = encode_dialogue(stripped or utts, self.tok, self.max_length, self.head_ratio)

        t = self._torch
        with t.no_grad():
            logits = self.model(
                input_ids=t.tensor([enc.input_ids], device=self.device),
                attention_mask=t.tensor([enc.attention_mask], device=self.device),
                speaker_ids=t.tensor([enc.speaker_ids], device=self.device),
            )
        return t.sigmoid(logits)[0].cpu().numpy()


@lru_cache(maxsize=None)
def _load(ckpt: str) -> _SRClassifier | None:
    """싱글턴. torch가 없거나 checkpoint가 없으면 None — 호출부가 부분 실패로 처리한다."""
    path = model_dir() / ckpt
    if not (path / "config.json").exists() or not (path / "model.pt").exists():
        return None
    try:
        return _SRClassifier(path)
    except (ImportError, ModuleNotFoundError):
        return None  # torch/transformers 미설치 (기본 설치 환경)
    except OSError:
        return None  # 백본 config를 HF 허브/캐시에서 못 찾음(오프라인) — 부분 실패로 넘긴다


def predict_type(transcript) -> tuple[str, float] | None:
    """Transcript -> (상담유형, 확신도). 모델이 없으면 None.

    확신도는 **선택된 라벨의 sigmoid 확률**이다. 다만 우선순위 규칙으로 고른 값이라
    "1등 확률"이 아닐 수 있다 — 적기처리기간이 긴 라벨이 임계값만 넘으면 확률이 더 높은
    라벨을 제치고 선택된다(그게 기관 규칙이다). 그래서 `needs_review`가 서면 절반으로 깎는다:
    임계값을 아무도 못 넘었거나, 1~2위 확률 차가 작거나, 우선순위 규칙만으로 갈린 경우다.

    ⚠️ 캘리브레이션된 확률이 아니다. 받은 checkpoint의 임계값은 아직 전부 0.5(초기값)이고,
    검증셋도 14건이다. 3-tier UI 임계값에 물리기 전 보정이 필요하다 (명세 9.3).
    """
    clf = _load(SR_CKPT)
    if clf is None:
        return None

    from callwrap.labels import collapse_probs

    probs = clf.probs(transcript)
    label, detail = collapse_probs(probs, clf.thresholds, return_detail=True)
    confidence = float(probs[clf.labels.index(label)])
    if detail["needs_review"]:
        confidence *= 0.5
    return label, round(min(1.0, max(0.0, confidence)), 4)


def is_ready() -> bool:
    """모델이 올라오는지. 서버 기동 시 확인용."""
    return _load(SR_CKPT) is not None
