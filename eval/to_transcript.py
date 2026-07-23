"""대본 → Transcript 변환기.

생성기가 만든 상담 대본을 계약(callwrap_ai.Transcript) 형식으로 바꾼다.
실서비스에서 stt/가 하는 "화자분리 결과 → Transcript" 변환을, 음성이 없는
1단계에서 대신하는 대역이다. AI를 쓰지 않는 규칙 기반 변환:

  1. 화자 매핑   : "상담원" → 상담사, 그 외 모든 라벨 → 고객
  2. 타임스탬프  : 음성이 없으므로 글자 수 기반으로 start/end 추정
                   (F1/F2/F4는 안 쓰고, F3 발화 패턴용 근사치. 음성 확보 시 실측 대체)
  3. 메타 제거   : tone/speed/volume은 TTS 지시용이라 Transcript에 넣지 않는다

라벨·요약 정답은 Transcript에 섞지 않고 gold 파일로 따로 뽑는다 (평가 CLI가 대조용으로 사용).

실행:
  uv run python eval/to_transcript.py            # 유지분 + v2 전체 변환
  uv run python eval/to_transcript.py --check     # 변환만 하고 계약 검증(파일 미출력)

출력:
  ai/data/scenarios/transcripts.jsonl   {scenario_id, transcript}
  ai/data/scenarios/gold.jsonl          {scenario_id, labels, summary, meta}
"""

import json
import sys
from pathlib import Path

from callwrap_ai import Speaker, Transcript, Utterance

AI_DIR = Path(__file__).resolve().parents[1]
SCN_DIR = AI_DIR / "data/scenarios"
SOURCES = ["scenarios_v2.jsonl", "scenarios_v1_kept.jsonl"]

# 화자 라벨 → 역할. "상담원"만 상담사, 나머지(문의자·제공기관 담당자·공무원·보호자 등)는 고객.
COUNSELOR_LABELS = {"상담원", "상담사"}

# 타임스탬프 추정 파라미터 (음성 부재 시 근사치)
CHARS_PER_SEC = 5.0  # 한국어 상담 발화 대략치
PAUSE_SEC = 0.6  # 발화 간 간격


def map_speaker(raw: str) -> Speaker:
    return Speaker.COUNSELOR if raw.strip() in COUNSELOR_LABELS else Speaker.CUSTOMER


def to_transcript(scn: dict) -> Transcript:
    utterances, clock = [], 0.0
    for i, u in enumerate(scn["utterances"]):
        text = u["text"]
        dur = max(len(text) / CHARS_PER_SEC, 0.5)
        start = clock
        end = start + dur
        clock = end + PAUSE_SEC
        utterances.append(
            Utterance(
                index=i,  # 원본 index를 신뢰하지 않고 0..n 재부여 (evidence 기준 통일)
                speaker=map_speaker(u["speaker"]),
                text=text,
                start=round(start, 2),
                end=round(end, 2),
                confidence=None,  # 합성 데이터엔 STT 신뢰도 없음
            )
        )
    return Transcript(utterances=utterances)


def extract_gold(scn: dict) -> dict:
    """평가 정답. v1 유지분은 legacy_labels(구 라벨, 사람 재라벨 대상)를 참고로 담는다."""
    if scn.get("origin") == "v1":
        labels = scn.get("legacy_labels", {})
        summary = scn.get("summary_draft", {})
        label_status = "legacy_needs_relabel"  # 구 체계 — 라벨링 세션에서 확정
    else:
        labels = scn.get("labels", {})
        summary = scn.get("summary_gold", {})
        label_status = "generated"  # 신 체계 생성 라벨 — 사람 검수 대상
    return {
        "scenario_id": scn.get("spec_id") or scn.get("id"),
        "label_status": label_status,
        "labels": labels,
        "summary": summary,
        "scenario_type": scn.get("scenario_type"),
        "seed_faq_ids": scn.get("seed_faq_ids", []),
    }


def load_all() -> list[dict]:
    scns = []
    for name in SOURCES:
        path = SCN_DIR / name
        if not path.exists():
            print(f"  경고: {name} 없음 — 건너뜀")
            continue
        scns += [json.loads(line) for line in path.open(encoding="utf-8")]
    return scns


def main() -> None:
    check_only = "--check" in sys.argv
    scns = load_all()

    transcripts, golds = [], []
    role_counter = {Speaker.COUNSELOR: 0, Speaker.CUSTOMER: 0}
    for scn in scns:
        sid = scn.get("spec_id") or scn.get("id")
        tr = to_transcript(scn)
        for u in tr.utterances:
            role_counter[u.speaker] += 1
        transcripts.append({"scenario_id": sid, "transcript": tr.model_dump()})
        golds.append(extract_gold(scn))

    # 계약 왕복 검증: 직렬화 → 재파싱이 무손실인지
    for t in transcripts:
        Transcript.model_validate(t["transcript"])

    print(f"변환 {len(transcripts)}건 (계약 검증 통과)")
    print(f"  발화 역할: 상담사 {role_counter[Speaker.COUNSELOR]} / 고객 {role_counter[Speaker.CUSTOMER]}")
    relabel = sum(1 for g in golds if g["label_status"] == "legacy_needs_relabel")
    print(f"  gold: 신규 생성 라벨 {len(golds) - relabel} / 재라벨 필요(v1) {relabel}")

    if check_only:
        print("(--check: 파일 미출력)")
        return

    with (SCN_DIR / "transcripts.jsonl").open("w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with (SCN_DIR / "gold.jsonl").open("w", encoding="utf-8") as f:
        for g in golds:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"저장: transcripts.jsonl · gold.jsonl -> {SCN_DIR}")


if __name__ == "__main__":
    main()
