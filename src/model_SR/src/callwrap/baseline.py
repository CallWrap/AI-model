"""P3-b: TF-IDF 베이스라인.

**용도를 오해하지 말 것.** 이것은 성능 하한 측정용이 아니라 **데이터 품질
진단용**이다.

char n-gram + LogisticRegression 같은 단순 모델이 합성 검증셋에서 macro-F1
0.90을 넘으면 좋은 신호가 아니라 **나쁜 신호**다. LLM이 유형별로 정형화된
표현("시스템 오류가 발생하여")을 반복 생성했다는 뜻이고, 모델은 그 키워드만
외운다. 실제 통화에서는 그런 말을 안 하므로 현장에서 무너진다.

이 경우 P4로 가지 말고 P2(생성 프롬프트)로 돌아가 경계 케이스와 표현
다양성을 늘려야 한다.

사용:
    python -m callwrap.baseline --data data/synth_stt.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .dataset import (load_jsonl, label_matrix, stratified_split,
                      strip_boilerplate, to_utterances)
from .evaluate import report
from .labels import LABELS, collapse_batch

QUALITY_GATE = 0.90  # 정의서 §6.3


def to_text(record: dict, strip_bp: bool = True) -> str:
    utts = to_utterances(record)
    if strip_bp:
        utts = strip_boilerplate(utts)
    return " ".join(f"[{u.speaker}] {u.text}" for u in utts)


def build_pipeline() -> Pipeline:
    # char n-gram은 STT 오인식과 띄어쓰기 붕괴에 의외로 강하다.
    # word n-gram과 결합하면 더 좋지만, 진단용이므로 단순하게 유지한다.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            min_df=2, max_features=200_000, sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
            n_jobs=-1,
        )),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-IDF 베이스라인 (데이터 품질 진단)")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--val-data", type=Path, default=None,
                    help="별도 검증셋(실데이터 권장). 없으면 --data에서 층화 분할")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-banlist", type=Path, default=None,
                    help="누설 표현을 금지어 목록으로 저장 (생성 프롬프트가 읽는다)")
    ap.add_argument("--banlist-top-k", type=int, default=12)
    args = ap.parse_args()

    records = load_jsonl(args.data)
    if args.val_data:
        train_recs = records
        val_recs = load_jsonl(args.val_data)
        val_kind = f"별도 검증셋 ({args.val_data.name})"
    else:
        train_recs, val_recs = stratified_split(records, args.val_ratio, args.seed)
        val_kind = "합성 검증셋 (층화 분할)"

    print(f"학습 {len(train_recs)}건 / 검증 {len(val_recs)}건 — {val_kind}")

    X_tr = [to_text(r) for r in train_recs]
    X_va = [to_text(r) for r in val_recs]
    Y_tr = label_matrix(train_recs)

    pipe = build_pipeline()
    pipe.fit(X_tr, Y_tr)

    probs = np.asarray(pipe.predict_proba(X_va), dtype=np.float64)
    thresholds = np.full(len(LABELS), 0.5)

    y_pred = collapse_batch(probs, thresholds)
    y_true = [r["sr_final"] for r in val_recs]

    m = report(y_true, y_pred, probs, thresholds,
               title=f"TF-IDF 베이스라인 — {val_kind}")

    # ---- 데이터 품질 게이트 ------------------------------------------------
    print("\n" + "=" * 68)
    if not args.val_data and m["macro_f1"] > QUALITY_GATE:
        print(f"⚠ 경고: 합성 검증셋 macro-F1 = {m['macro_f1']:.3f} "
              f"> 게이트 {QUALITY_GATE}")
        print("  단순 모델이 이 정도로 잘 맞힌다는 것은 표현 다양성 부족을 뜻합니다.")
        print("  (정의서 §6.3) P4로 진행하지 말고 P2 생성 프롬프트로 회귀하십시오.")
        print("  점검: 유형별 첫 문장 반복, 라벨 누설 표현, 경계 케이스 부족")
        _diagnose_leakage(pipe)
    elif not args.val_data:
        print(f"✔ 합성 검증셋 macro-F1 = {m['macro_f1']:.3f} "
              f"(게이트 {QUALITY_GATE} 이하). 표현 다양성 양호.")
        print(f"  BERT 계열 목표치: macro-F1 {min(0.95, m['macro_f1'] + 0.15):.2f} 내외")
    else:
        print(f"실데이터 검증셋 macro-F1 = {m['macro_f1']:.3f} "
              f"— BERT 파인튜닝의 비교 기준선입니다.")
    print("=" * 68)

    if args.emit_banlist:
        banlist = build_banlist(records, top_k=args.banlist_top_k)
        args.emit_banlist.parent.mkdir(parents=True, exist_ok=True)
        args.emit_banlist.write_text(
            json.dumps(banlist, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total = sum(len(v) for v in banlist.values())
        print(f"\n금지어 {total}개를 {args.emit_banlist} 에 저장했습니다.")
        print("  다음 생성부터 이 표현들이 프롬프트에서 차단됩니다.")
        for name, terms in banlist.items():
            if terms:
                print(f"   {name:<16}: {', '.join(terms[:6])}")


def _diagnose_leakage(pipe: Pipeline, top_k: int = 8) -> None:
    """클래스별 최고 가중치 n-gram을 뽑아 라벨 누설 표현을 눈으로 찾는다."""
    try:
        vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
        print("\n  [클래스별 최고 가중치 n-gram — 정형 표현 점검]")
        for i, est in enumerate(pipe.named_steps["clf"].estimators_):
            top = np.argsort(-est.coef_[0])[:top_k]
            grams = ", ".join(repr(g) for g in vocab[top])
            print(f"   {LABELS[i]:<16}: {grams}")
    except Exception as exc:  # noqa: BLE001 — 진단 보조 기능
        print(f"  (n-gram 진단 생략: {exc})")


# 금지해도 의미가 없고 자연스러움만 해치는 일반 어휘.
# 도메인 용어(바우처, 결제 등)는 반드시 나와야 하므로 여기 넣지 않는다.
_GENERIC_TOKENS = {
    # 감탄사 · 대명사 · 지시어
    "네", "네.", "예", "아", "어", "음", "그", "저", "저는", "제가", "그건", "그게",
    "이거", "저희", "여기", "거기", "그거", "이건", "우리",
    # 접속 · 부사
    "한데", "근데", "그럼", "이번", "지금", "지금은", "좀", "조금", "다시", "먼저",
    "또", "잘", "안", "못", "따라", "대해", "때문에", "다만", "혹시", "아직",
    "많이", "바로", "계속", "그냥", "아예", "이제", "다른", "같은", "어떤",
    # 형식명사 · 의존명사
    "것이", "것을", "것", "수", "때", "경우", "경우가", "부분", "정도", "중에",
    # 흔한 서술어 종결
    "있어", "있어요", "없어요", "해요", "돼요", "합니다", "됩니다", "있습니다",
    "합니다.", "됩니다.", "있습니다.", "없습니다.", "않습니다.", "됐어요.",
    "감사합니다", "감사합니다.", "알겠습니다", "알겠습니다.", "말씀", "부탁",
    # 의문형 종결 — 유형과 무관하게 어디서나 나온다
    "어떻게", "어떻게요?", "되나요?", "있나요?", "하나요?", "뭔가요?", "될까요?",
    "은", "는", "이", "가", "을", "를", "만",
}

_SPEAKER_TAG_RE = re.compile(r"\[(고객|상담사)\]")

# 도메인 핵심 어휘 — **단독으로는 절대 금지하지 않는다.**
#
# 실제 상담 통화에는 이 단어들이 그대로 나온다. 금지하면 생성기가 "카드리더기",
# "인터넷 창", "바로잡기" 같은 우회 표현을 쓰게 되고, 그렇게 만든 학습 데이터는
# 실제 STT 전사와 어휘 분포가 어긋난다. 정형화를 막으려다 새로운 train-serving
# skew를 만드는 셈이다.
#
# 문제는 단어가 아니라 **정형구**다. `취소`는 보호하되 `취소 처리해 드리겠습니다`
# 같은 다어절 상투구는 계속 금지 대상으로 둔다.
_PROTECTED_TERMS = {
    # 시스템 · 단말
    "단말기", "결제단말기", "지문인식기", "브라우저", "크롬", "엣지", "인증서",
    "공동인증서", "보안", "설치", "화면", "메뉴", "버튼", "로그인", "포털",
    "시스템", "프로그램", "오류", "장애", "접속", "조회", "검색", "출력",
    # 바우처 업무
    "바우처", "전자바우처", "결제", "승인", "취소", "정정", "환급", "지급",
    "본인부담금", "가산단가", "예탁금", "가상계좌", "이용권", "잔량", "생성",
    "자격", "등록", "신청", "접수", "급여", "교통비", "과오결제",
    # 주체
    "제공기관", "제공인력", "활동지원사", "대상자", "수급자", "지자체",
    "주민센터", "상담센터", "행복e음", "기관", "인력", "담당자",
    "사회서비스", "복지관", "자립생활센터", "건강보험공단", "국민연금공단",
    # 기능 · 절차
    "기능", "개선", "신고", "포상금", "부정사용", "지침", "기준", "요건",
    "서류", "계좌", "명의", "변경", "이관", "확인",
}


_JOSA = ("으로", "에서", "에게", "이나", "라도", "까지", "부터", "이란", "이라",
         "은", "는", "이", "가", "을", "를", "의", "에", "도", "만", "과", "와", "로")


def _strip_josa(term: str) -> str:
    """어절에서 조사를 떼어 보호어 대조에 쓴다. (`단말기가` -> `단말기`)"""
    for j in _JOSA:  # 긴 것부터 정렬되어 있음
        if len(term) > len(j) + 1 and term.endswith(j):
            return term[: -len(j)]
    return term


def build_banlist(records: list[dict], top_k: int = 10, min_len: int = 2,
                  min_coef: float = 0.40, unigram_coef: float = 0.75) -> dict[str, list[str]]:
    """라벨을 지나치게 잘 예측하는 **어절 단위** 표현을 뽑아낸다.

    char n-gram은 사람이 읽기 어려운 조각(' 개발', '팀에')을 내므로,
    금지어 목록에는 word n-gram을 따로 학습해 쓴다. 결과는 생성 프롬프트에
    "쓰지 말 것" 목록으로 주입된다 (generate -> diagnose -> constrain 루프).

    걸러내는 것:
      - 화자 태그가 섞인 조각 (`[고객] 아`) — 전처리 산물이지 표현이 아니다
      - 일반 기능어 (`저는`, `조금`) — 금지하면 자연스러움만 해친다
      - 계수가 낮은 항목 — 약한 신호까지 막으면 생성이 과도하게 제약된다

    **단어 1개짜리는 더 높은 임계(`unigram_coef`)를 요구한다.** 표본이 적으면
    로지스틱 회귀가 흔한 단어에 과적합해 `다른`, `경우` 같은 것을 상위로 올린다.
    진짜 누설은 대개 `개발팀에`, `요청으로 접수` 같은 다어절 정형구다.
    """
    # 화자 태그를 뺀 본문으로 계산한다. 태그는 모든 대화에 있으므로 신호가 아니다.
    X = [_SPEAKER_TAG_RE.sub(" ", to_text(r)) for r in records]
    Y = label_matrix(records)

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                          min_df=2, sublinear_tf=True, token_pattern=r"\S+")
    Xv = vec.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())

    banlist: dict[str, list[str]] = {}
    for i, name in enumerate(LABELS):
        if Y[:, i].sum() < 2:
            banlist[name] = []
            continue
        clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
        clf.fit(Xv, Y[:, i])
        coef = clf.coef_[0]
        picked: list[str] = []
        for idx in np.argsort(-coef):
            if coef[idx] < min_coef:
                break  # 정렬되어 있으므로 이후는 모두 임계 미만
            term = vocab[idx].strip()
            if len(term) < min_len or term.isdigit():
                continue
            tokens = term.split()
            if len(tokens) == 1:
                # 도메인 핵심어는 단독으로 금지하지 않는다. 실제 통화에 나오는
                # 말을 막으면 학습 데이터가 실제 전사와 어긋난다.
                if _strip_josa(term) in _PROTECTED_TERMS:
                    continue
                # 단어 1개는 과적합되기 쉬우므로 더 강한 증거를 요구한다
                if coef[idx] < unigram_coef:
                    continue
            if any(tok in _GENERIC_TOKENS for tok in tokens):
                continue
            if any(term in p or p in term for p in picked):
                continue
            picked.append(term)
            if len(picked) >= top_k:
                break
        banlist[name] = picked
    return banlist


if __name__ == "__main__":
    main()
