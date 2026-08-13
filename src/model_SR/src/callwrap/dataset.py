"""전처리 · 데이터셋.

핵심 설계 3가지:

1. **head+tail 절단** — SR유형 신호는 고객의 최초 용건(앞)과 상담사의 처리
   결론(뒤)에 몰려 있고, 중간은 본인확인·대기·잡담이다. 단순 앞자르기보다
   앞 H + 뒤 T 를 남기는 편이 확실히 낫다.

2. **화자 정보 보존** — `[고객]`/`[상담사]` 스페셜 토큰 + 토큰별 speaker_id.
   diarization 결과를 버리면 안 된다. 고객 발화가 SR유형의 주 신호다.

3. **STT 노이즈 증강** — 학습 시 확률적으로 적용. train-serving skew 방지.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import NUM_LABELS, to_multihot
from .stt_noise import NoiseConfig, STTNoiseAugmenter, Utterance

CUSTOMER_TOKEN = "[고객]"
AGENT_TOKEN = "[상담사]"
SPEAKER_TOKENS = [CUSTOMER_TOKEN, AGENT_TOKEN]

SPK_AGENT, SPK_CUSTOMER = 0, 1

# 정보량이 거의 없는 상투 구간. 유효 토큰 밀도를 높이기 위해 제거한다.
_BOILERPLATE = re.compile(
    r"(안녕하세[요십니까]+|무엇을 도와드릴까요|반갑습니다|"
    r"성함이 어떻게 되[시세]|생년월일[을은]? (?:말씀|알려)|"
    r"본인 ?확인[을이]? ?(?:위해|하겠)|잠시만 기다려 ?주[시세]|"
    r"확인하는 동안 잠시|더 궁금하신 (?:점|사항)|"
    r"이용해 ?주셔서 감사합니다|좋은 하루 되[세십])"
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_utterances(record: dict) -> list[Utterance]:
    return [Utterance(u["speaker"], u["text"]) for u in record["utterances"]]


# ---------------------------------------------------------------------------
# 정제
# ---------------------------------------------------------------------------
def strip_boilerplate(utts: list[Utterance], keep_min: int = 4) -> list[Utterance]:
    """인사·본인확인 등 상투 발화를 제거한다.

    전부 지워지면 원본을 유지한다 (짧은 통화 보호).
    """
    kept = [u for u in utts if not (_BOILERPLATE.search(u.text) and len(u.text) < 60)]
    return kept if len(kept) >= keep_min else utts


# ---------------------------------------------------------------------------
# 인코딩
# ---------------------------------------------------------------------------
@dataclass
class Encoded:
    input_ids: list[int]
    attention_mask: list[int]
    speaker_ids: list[int]


def encode_dialogue(
    utts: Sequence[Utterance],
    tokenizer,
    max_length: int = 512,
    head_ratio: float = 0.5,
) -> Encoded:
    """대화 -> 토큰. 길면 head+tail 절단.

    speaker_ids는 input_ids와 같은 길이이며, 스페셜 토큰 위치는 SPK_AGENT(0).
    """
    ids: list[int] = []
    spk: list[int] = []

    for u in utts:
        marker = CUSTOMER_TOKEN if u.speaker == "고객" else AGENT_TOKEN
        sid = SPK_CUSTOMER if u.speaker == "고객" else SPK_AGENT
        piece = tokenizer.encode(f"{marker} {u.text}", add_special_tokens=False)
        ids.extend(piece)
        spk.extend([sid] * len(piece))

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if len(ids) <= max_length - 2:
        out_ids = [cls_id] + ids + [sep_id]
        out_spk = [SPK_AGENT] + spk + [SPK_AGENT]
    else:
        # [CLS] head [SEP] tail [SEP]
        budget = max_length - 3
        n_head = int(budget * head_ratio)
        n_tail = budget - n_head
        out_ids = ([cls_id] + ids[:n_head] + [sep_id] + ids[-n_tail:] + [sep_id])
        out_spk = ([SPK_AGENT] + spk[:n_head] + [SPK_AGENT] + spk[-n_tail:]
                   + [SPK_AGENT])

    return Encoded(out_ids, [1] * len(out_ids), out_spk)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SRDataset(Dataset):
    """SR유형 멀티라벨 분류 데이터셋.

    noise_prob > 0 이면 __getitem__ 시점마다 STT 노이즈를 새로 샘플링한다.
    (에폭마다 다른 오염이 걸리므로 정적 증강보다 다양성이 크다.)
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        *,
        max_length: int = 512,
        head_ratio: float = 0.5,
        noise_prob: float = 0.0,
        noise_config: NoiseConfig | None = None,
        strip_bp: bool = True,
        seed: int = 0,
        label_key: str = "sr_labels",
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_ratio = head_ratio
        self.noise_prob = noise_prob
        self.noise_config = noise_config or NoiseConfig()
        self.strip_bp = strip_bp
        self.label_key = label_key
        self.seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        utts = to_utterances(rec)

        if self.noise_prob and self._rng.random() < self.noise_prob:
            aug = STTNoiseAugmenter(self.noise_config,
                                    seed=self._rng.randrange(1 << 30))
            utts = aug.augment_dialogue(utts)
        if self.strip_bp:
            utts = strip_boilerplate(utts)
        if not utts:
            utts = to_utterances(rec)

        enc = encode_dialogue(utts, self.tokenizer, self.max_length, self.head_ratio)
        labels = to_multihot(rec.get(self.label_key, []))

        return {
            "input_ids": torch.tensor(enc.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(enc.attention_mask, dtype=torch.long),
            "speaker_ids": torch.tensor(enc.speaker_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


@dataclass
class Collator:
    """동적 패딩 collator."""

    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        maxlen = max(len(b["input_ids"]) for b in batch)
        out: dict[str, list] = {k: [] for k in
                                ("input_ids", "attention_mask", "speaker_ids")}
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            out["input_ids"].append(
                torch.cat([b["input_ids"],
                           torch.full((pad,), self.pad_token_id, dtype=torch.long)])
            )
            out["attention_mask"].append(
                torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            out["speaker_ids"].append(
                torch.cat([b["speaker_ids"], torch.zeros(pad, dtype=torch.long)])
            )
        result = {k: torch.stack(v) for k, v in out.items()}
        result["labels"] = torch.stack([b["labels"] for b in batch])
        return result


# ---------------------------------------------------------------------------
# 분할
# ---------------------------------------------------------------------------
def stratified_split(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 0,
    key: str = "sr_final",
) -> tuple[list[dict], list[dict]]:
    """최종 SR유형 기준 층화 분할."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in records:
        buckets.setdefault(r.get(key, "기타"), []).append(r)

    train, val = [], []
    for group in buckets.values():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def label_matrix(records: list[dict], key: str = "sr_labels") -> np.ndarray:
    return np.stack([to_multihot(r.get(key, [])) for r in records]) \
        if records else np.zeros((0, NUM_LABELS), dtype=np.float32)


def register_speaker_tokens(tokenizer, model=None) -> int:
    """화자 스페셜 토큰을 추가하고 필요 시 임베딩을 리사이즈한다."""
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPEAKER_TOKENS}
    )
    if added and model is not None:
        model.resize_token_embeddings(len(tokenizer))
    return added
