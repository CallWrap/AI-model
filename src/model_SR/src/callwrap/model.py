"""SR유형 분류 모델 (PyTorch).

구조:
    BERT계열 백본  ->  풀링  ->  드롭아웃  ->  Linear(8)  ->  sigmoid

멀티라벨 8-way sigmoid 헤드다 (softmax 아님). 한 통화에 여러 유형이 섞이는
것이 정상이고, 최종 SR유형은 추론 후 우선순위 규칙으로 축약한다.
정의서 §2 원칙 1 참조.

화자 임베딩:
    diarization 결과(고객/상담사)를 토큰 임베딩에 더한다. 0으로 초기화하므로
    학습 초기에는 항등이고, 필요한 만큼만 학습된다.
    RoBERTa 계열은 token_type_ids의 type_vocab_size가 1이라 세그먼트 임베딩을
    쓸 수 없으므로 이 방식을 택했다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .labels import NUM_LABELS


class SRClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "klue/roberta-large",
        num_labels: int = NUM_LABELS,
        *,
        dropout: float = 0.1,
        pooling: str = "cls_mean",   # cls | mean | cls_mean
        use_speaker_embedding: bool = True,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=self.config)
        if vocab_size is not None and vocab_size != self.backbone.config.vocab_size:
            self.backbone.resize_token_embeddings(vocab_size)

        hidden = self.config.hidden_size
        self.pooling = pooling
        self.use_speaker_embedding = use_speaker_embedding

        if use_speaker_embedding:
            self.speaker_embedding = nn.Embedding(2, hidden)
            nn.init.zeros_(self.speaker_embedding.weight)  # 항등으로 시작

        pooled_dim = hidden * (2 if pooling == "cls_mean" else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(pooled_dim, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    # -- 내부 -------------------------------------------------------------
    def _word_embeddings(self) -> nn.Embedding:
        emb = self.backbone.get_input_embeddings()
        if emb is None:
            raise RuntimeError("백본에서 입력 임베딩을 찾을 수 없습니다.")
        return emb

    def _pool(self, hidden_states: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        mean = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        if self.pooling == "mean":
            return mean
        return torch.cat([hidden_states[:, 0], mean], dim=-1)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_speaker_embedding and speaker_ids is not None:
            inputs_embeds = self._word_embeddings()(input_ids) \
                + self.speaker_embedding(speaker_ids)
            out = self.backbone(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))  # logits

    # -- 옵티마이저 파라미터 그룹 -------------------------------------------
    def optimizer_groups(
        self, lr: float, head_lr_mult: float = 10.0,
        weight_decay: float = 0.01, llrd: float = 0.9,
    ) -> list[dict]:
        """Layer-wise LR decay.

        상위 레이어일수록 큰 학습률을 준다. 데이터가 적을 때 하위 레이어의
        일반 언어 표현이 망가지는 것을 막아준다.
        """
        no_decay = ("bias", "LayerNorm.weight", "layer_norm")
        n_layers = self.config.num_hidden_layers
        groups: list[dict] = []

        def decayable(name: str) -> bool:
            return not any(nd in name for nd in no_decay)

        def layer_index(name: str) -> int:
            """임베딩=0, encoder.layer.i = i+1, 나머지 백본=n_layers+1"""
            if "embeddings" in name:
                return 0
            if ".layer." in name:
                try:
                    return int(name.split(".layer.")[1].split(".")[0]) + 1
                except (IndexError, ValueError):
                    pass
            return n_layers + 1

        buckets: dict[tuple[int, bool], list[nn.Parameter]] = {}
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            buckets.setdefault((layer_index(name), decayable(name)), []).append(param)

        for (idx, wd), params in sorted(buckets.items()):
            groups.append({
                "params": params,
                "lr": lr * (llrd ** (n_layers + 1 - idx)),
                "weight_decay": weight_decay if wd else 0.0,
            })

        head_params = list(self.classifier.parameters())
        if self.use_speaker_embedding:
            head_params += list(self.speaker_embedding.parameters())
        groups.append({
            "params": head_params,
            "lr": lr * head_lr_mult,
            "weight_decay": weight_decay,
        })
        return groups
