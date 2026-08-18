"""TF-IDF 누출 진단 — 생성 데이터가 라벨을 문구로 흘리는지 검사한다.

    uv run --extra train python eval/diagnose_leakage.py eval/datasets/golden_subcategory.jsonl
    uv run --extra train python eval/diagnose_leakage.py --label category eval/datasets/golden_category.jsonl
    uv run --extra train python eval/diagnose_leakage.py A.jsonl B.jsonl        # 각각 + 합친 것

왜 하나:
  라벨을 먼저 정하고 대본을 생성하면 라벨마다 정해진 문구가 반복되기 쉽다. 그러면
  모델이 내용이 아니라 문구를 외운다. 단어 빈도만 세는 TF-IDF로도 점수가 높게 나오면
  그 신호다. 재혁 파트에서 1차 생성 60건이 macro-F1 1.000이 나와 폐기한 전례가 있다.

읽는 법:
  절대값보다 **파인튜닝 모델 점수와의 격차**가 중요하다.
    TF-IDF가 0.95 이상        누출 의심. 아래 n-gram을 보고 범인을 찾는다
    TF-IDF ≈ 파인튜닝 점수     파인튜닝이 단어 세기 이상을 못 하고 있다
    TF-IDF < 파인튜닝 점수     정상. 모델이 문맥을 쓰고 있다

  클래스당 표본이 적으면 점수가 낮게 나온다. 낮다고 무조건 안전한 것은 아니다.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI / "train"))  # 학습과 같은 전처리를 쓴다
from preprocess import strip_boilerplate  # noqa: E402


def load(paths: list[Path], label_key: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for p in paths:
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            text = r.get("text") or "\n".join(f"{u['speaker']}: {u['text']}" for u in r.get("utterances", []))
            lab = r.get(label_key) or (r.get("labels") or {}).get(f"consultation_{label_key}")
            if text and lab:
                texts.append(strip_boilerplate(text))
                labels.append(lab)
    return texts, labels


def diagnose(texts: list[str], labels: list[str], name: str, top_n: int = 6) -> float:
    cnt = Counter(labels)
    # 층화 교차검증은 클래스당 최소 2건이 필요하다. 1건짜리는 빼고 리포트에 남긴다.
    singles = [c for c, n in cnt.items() if n < 2]
    keep = [(t, l) for t, l in zip(texts, labels) if cnt[l] >= 2]
    if not keep:
        print(f"  [{name}] 표본 부족 — 진단 불가")
        return float("nan")
    X, y = zip(*keep)

    folds = min(5, min(Counter(y).values()))
    # char n-gram: 한국어는 형태소 분석기 없이도 이 방식이 안정적이다.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    preds = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(X, y):
        Xtr = vec.fit_transform([X[i] for i in tr])
        clf.fit(Xtr, [y[i] for i in tr])
        preds[te] = clf.predict(vec.transform([X[i] for i in te]))

    macro = f1_score(y, preds, average="macro", zero_division=0)
    verdict = "누출 의심" if macro >= 0.95 else "경계" if macro >= 0.90 else "정상 범위"
    print(f"  [{name}] {len(y)}건 / {len(set(y))}클래스 / {folds}겹 → TF-IDF macro-F1 {macro:.4f}  ({verdict})")
    if singles:
        print(f"      표본 1건이라 제외한 클래스 {len(singles)}종: {', '.join(sorted(singles))}")

    if macro >= 0.90:
        vec.fit(X)
        clf.fit(vec.transform(X), y)
        names = np.array(vec.get_feature_names_out())
        print("      클래스별 가중치 상위 n-gram (누출 문구 후보):")
        for i, c in enumerate(clf.classes_):
            w = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_[0]
            top = names[np.argsort(w)[-top_n:][::-1]]
            print(f"        {c:18s} {' · '.join(repr(t) for t in top)}")
    return macro


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-IDF 누출 진단")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--label", default="subcategory", choices=("subcategory", "category", "type", "grade"))
    args = ap.parse_args()

    missing = [p for p in args.files if not p.exists()]
    if missing:
        sys.exit("파일이 없습니다: " + ", ".join(map(str, missing)))

    print(f"라벨: {args.label}\n")
    for p in args.files:
        t, l = load([p], args.label)
        diagnose(t, l, p.name)

    if len(args.files) > 1:
        print()
        t, l = load(args.files, args.label)
        diagnose(t, l, "합친 것")


if __name__ == "__main__":
    main()
