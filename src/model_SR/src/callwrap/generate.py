"""P2/P3-a: 합성 상담 대화 생성 러너.

Anthropic Messages API + 구조화 출력(Pydantic)으로 라벨 조건부 생성을 수행한다.
스키마 검증이 API 레이어에서 이뤄지므로 파싱 실패 시 모델이 자동 재시도한다.

사용:
    # P2 파일럿 200건
    python -m callwrap.generate --out data/pilot.jsonl --limit 200 --seed 0

    # P3-a 본 생성
    python -m callwrap.generate --out data/synth_raw.jsonl --per-cell 3 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .labels import LABELS, collapse
from .prompts import (GenSpec, build_grid, build_system_prompt,
                      build_user_prompt, load_banlist)

MAX_TOKENS = 8000


def load_env() -> None:
    """프로젝트 루트의 .env 를 환경변수로 읽어들인다.

    이미 설정된 환경변수는 덮어쓰지 않는다 (셸에서 준 값이 우선).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # src/callwrap/generate.py -> 프로젝트 루트는 2단계 위
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


def get_model() -> str:
    return os.getenv("CALLWRAP_GEN_MODEL", "claude-opus-5")


def check_auth(verbose: bool = True) -> bool:
    """자격증명이 실제로 동작하는지 최소 비용으로 확인한다."""
    load_env()
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()

    if not key and not token:
        if verbose:
            print("✘ 자격증명 없음.")
            print("  .env 파일의 ANTHROPIC_API_KEY= 뒤에 키를 붙여넣으십시오.")
            print(f"  경로: {Path(__file__).resolve().parents[2] / '.env'}")
        return False

    if key and not key.startswith("sk-ant-"):
        if verbose:
            print(f"✘ ANTHROPIC_API_KEY 형식이 이상합니다 (앞 8자: {key[:8]!r}).")
            print("  'sk-ant-' 로 시작해야 합니다. 따옴표나 공백이 섞이지 않았는지 확인하십시오.")
        return False

    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=get_model(), max_tokens=16,
            messages=[{"role": "user", "content": "OK 라고만 답하세요."}],
        )
        if verbose:
            src = "ANTHROPIC_API_KEY" if key else "ANTHROPIC_AUTH_TOKEN"
            print(f"✔ 인증 성공 ({src}, 모델 {resp.model})")
            print(f"  토큰 사용: in {resp.usage.input_tokens} / "
                  f"out {resp.usage.output_tokens}")
        return True
    except Exception as exc:  # noqa: BLE001 — 원인을 그대로 보여준다
        if verbose:
            print(f"✘ 인증 실패: {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# 구조화 출력 스키마
# ---------------------------------------------------------------------------
class Utterance(BaseModel):
    speaker: Literal["고객", "상담사"]
    text: str = Field(description="해당 화자의 연속 발화 한 덩어리")


class GeneratedCall(BaseModel):
    """생성된 상담 통화 1건."""

    sr_labels: list[str] = Field(
        description=(
            "이 통화에 실제로 나타난 모든 SR유형. "
            "허용값: 사용법문의, 시스템오류, 고객PC환경, 자료확인, 자료보정, "
            "정책 및 지침문의, 기능개선, 기타"
        )
    )
    utterances: list[Utterance] = Field(description="통화 전문. 8~24턴.")
    evidence_indices: list[int] = Field(
        description="SR유형 판정 근거가 된 발화의 0-기반 인덱스"
    )
    rationale: str = Field(description="라벨 판정 근거 1~2문장")


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------
def _make_client():
    import anthropic

    return anthropic.Anthropic()


def generate_one(client, spec: GenSpec, system_prompt: str,
                 *, max_retries: int = 3) -> dict | None:
    """셀 하나에 대해 통화 1건을 생성한다. 실패 시 None."""
    for attempt in range(max_retries):
        try:
            response = client.messages.parse(
                model=get_model(),
                max_tokens=MAX_TOKENS,
                output_config={"effort": "medium"},
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    # 시스템 프롬프트는 모든 요청에서 바이트 단위로 동일하다.
                    # 격자 수백~수천 셀을 도는 동안 캐시 히트가 누적된다.
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": build_user_prompt(spec)}],
                output_format=GeneratedCall,
            )
            call = response.parsed_output
            if call is None:
                continue

            valid = [l for l in call.sr_labels if l in LABELS]
            if spec.target_label not in valid:
                # 목표 유형이 빠졌으면 라벨 조건부 생성이 실패한 것. 재시도.
                continue

            return {
                "id": f"{spec.cell_id}#{attempt}",
                "sr_labels": valid,
                "sr_final": collapse(valid),
                "target_label": spec.target_label,
                "utterances": [u.model_dump() for u in call.utterances],
                "evidence_indices": call.evidence_indices,
                "rationale": call.rationale,
                "meta": {
                    "mid_category": spec.mid_category,
                    "sub_category": spec.sub_category,
                    "difficulty": spec.difficulty,
                    "caller_role": spec.caller_role,
                    "secondary_label": spec.secondary_label,
                    "source": "synthetic",
                },
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_input_tokens": getattr(
                        response.usage, "cache_read_input_tokens", 0
                    ),
                },
            }
        except Exception as exc:  # noqa: BLE001 — 러너는 개별 실패를 흡수한다
            if attempt == max_retries - 1:
                print(f"  [FAIL] {spec.cell_id}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def run(specs: list[GenSpec], out_path: Path, workers: int = 8) -> None:
    client = _make_client()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 금지어는 실행 시작 시점에 한 번 읽어 고정한다 (프롬프트 캐시 유지).
    banlist = load_banlist()
    system_prompt = build_system_prompt(banlist)
    n_banned = sum(len(v) for v in banlist.values())
    print(f"금지어 {n_banned}개 적용" if n_banned else
          "금지어 없음 (baseline --emit-banlist 로 생성하면 다음 실행부터 적용됩니다)")

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "in_tok": 0, "out_tok": 0, "cached": 0}
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, client, s, system_prompt): s
                   for s in specs}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                if rec is None:
                    stats["fail"] += 1
                else:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    stats["ok"] += 1
                    stats["in_tok"] += rec["usage"]["input_tokens"]
                    stats["out_tok"] += rec["usage"]["output_tokens"]
                    stats["cached"] += rec["usage"]["cache_read_input_tokens"]

                if i % 25 == 0 or i == len(specs):
                    el = time.time() - t0
                    print(
                        f"[{i}/{len(specs)}] ok={stats['ok']} fail={stats['fail']} "
                        f"in={stats['in_tok']:,} out={stats['out_tok']:,} "
                        f"cache_read={stats['cached']:,} ({el:.0f}s)",
                        flush=True,
                    )

    print(f"\n완료: {stats['ok']}건 -> {out_path}")
    if stats["fail"]:
        print(f"실패 {stats['fail']}건 (재시도 소진)")
    if stats["in_tok"]:
        hit = stats["cached"] / (stats["cached"] + stats["in_tok"]) * 100
        print(f"프롬프트 캐시 히트율: {hit:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="라벨 조건부 합성 상담 데이터 생성")
    ap.add_argument("--out", type=Path, default=Path("data/synth_raw.jsonl"))
    ap.add_argument("--per-cell", type=int, default=3,
                    help="소분류 x SR유형 조합당 생성 건수")
    ap.add_argument("--limit", type=int, default=None,
                    help="격자에서 앞 N셀만 (파일럿용)")
    ap.add_argument("--workers", type=int,
                    default=int(os.getenv("CALLWRAP_GEN_WORKERS", "8")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-auth", action="store_true",
                    help="자격증명만 확인하고 종료 (생성하지 않음)")
    ap.add_argument("--dry-run", action="store_true",
                    help="격자만 계산하고 종료 (API 호출 없음)")
    args = ap.parse_args()

    load_env()

    if args.check_auth:
        raise SystemExit(0 if check_auth() else 1)

    if not args.dry_run and not check_auth(verbose=True):
        raise SystemExit(1)

    specs = build_grid(per_cell=args.per_cell, seed=args.seed)
    if args.limit:
        # 파일럿은 라벨/난이도 분포를 유지한 채 층화 샘플링한다.
        rng = random.Random(args.seed)
        by_label: dict[str, list[GenSpec]] = {}
        for s in specs:
            by_label.setdefault(s.target_label, []).append(s)
        picked: list[GenSpec] = []
        per_label = max(1, args.limit // len(by_label))
        for group in by_label.values():
            picked.extend(rng.sample(group, min(per_label, len(group))))
        rng.shuffle(picked)
        specs = picked[: args.limit]

    print(f"생성 대상 {len(specs)}셀, 워커 {args.workers}, 모델 {get_model()}")
    if args.dry_run:
        from collections import Counter
        print("라벨 분포 :", Counter(s.target_label for s in specs).most_common())
        print("난이도    :", Counter(s.difficulty for s in specs).most_common())
        print("(--dry-run: API 호출 없이 종료)")
        return
    run(specs, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
