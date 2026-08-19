#!/usr/bin/env bash
# 망분리(사보원) 반입용 오프라인 번들을 만든다. 온라인 환경에서 1회 실행.
#
#   ./scripts/build_offline_bundle.sh
#
# 산출물: bundle/ 한 폴더 — 코드·의존성 wheel·실행 안내.
# 반입 후 밖에서 아무것도 받지 않는다(PyPI·GitHub·ollama.com 전부 불필요).
#
# ※ 이 번들은 callwrap_ai(F1~F4·FAQ 추론)만 다룬다. F2/SR checkpoint는 별도로 받아
#   bundle/models/checkpoints/ 에 넣는다(gitignore된 data/models와 동일 레이아웃).
# ※ model_SR(GPU 파인튜닝, klue/roberta-large + CUDA)과 eval/의 합성데이터 생성
#   스크립트(anthropic API 사용)는 이 번들 범위 밖이다 — 둘 다 이미 학습된 모델을
#   "실데이터로 테스트"하는 사보원 방문 목적과 무관하고, 전자는 특정 GPU(CUDA) 전용,
#   후자는 애초에 외부 API 호출이 필수라 오프라인으로 옮길 수 없다.
# ※ F3(감정판정)·F1/F4(요약·후속처리)는 온디바이스 LLM(ollama, exaone3.5:7.8b)이 필요하다.
#   ollama가 설치되고 모델을 pull해둔 머신에서 돌려야 5단계가 같이 담긴다.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bundle"
rm -rf "$OUT"
mkdir -p "$OUT"/{code,wheels,models/checkpoints,ollama}

echo "[1/5] 코드 아카이브"
git -C "$ROOT" archive --format=tar HEAD | (mkdir -p "$OUT/code/AI-model" && tar -x -C "$OUT/code/AI-model")

echo "[2/5] 의존성 wheel (base + train extra, transitive 포함)"
uv export --no-hashes --extra train --format requirements-txt > "$OUT/wheels/requirements.txt" 2>/dev/null
grep -vE "^-e |^callwrap-ai" "$OUT/wheels/requirements.txt" > "$OUT/wheels/requirements.pypi.txt"
# uv 가상환경엔 pip이 없다 — pip이 있는 python(3.11)이 필요하다. PIP_PYTHON으로 지정하거나,
# 기본으로 이 레포의 .venv에서 pip이 있는 걸 찾는다(없으면 만든다).
PIPPY="${PIP_PYTHON:-$ROOT/.venv/bin/python}"
if ! "$PIPPY" -m pip --version >/dev/null 2>&1; then
  python3 -m venv "$ROOT/.venv" && PIPPY="$ROOT/.venv/bin/python"
fi
"$PIPPY" -m pip download -q -r "$OUT/wheels/requirements.pypi.txt" -d "$OUT/wheels"
# callwrap-ai 자신도 wheel로 미리 빌드해 담는다 — editable(-e) 설치는 hatchling의 editable
# 백엔드가 추가 패키지(editables 등)를 더 요구해 오프라인에서 깨진다.
"$PIPPY" -m pip wheel -q --no-deps -w "$OUT/wheels" "$ROOT"

echo "[3/5] F2/SR checkpoint 자리 표시"
echo "F2(중분류·소분류)·SR유형 checkpoint를 여기 넣으세요 (data/models/ 와 같은 레이아웃)" \
  > "$OUT/models/checkpoints/README.txt"

echo "[4/5] ollama (F1·F3·F4·FAQ용 온디바이스 LLM)"
if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | grep -q "exaone3.5:7.8b"; then
  echo "     ollama 모델 발견 — 파일 복사(용량 큼, 수 GB)"
  cp -R "$HOME/.ollama/models" "$OUT/ollama/" 2>/dev/null && echo "     + ~/.ollama/models"
else
  cat > "$OUT/ollama/README.txt" <<'EOF'
ollama(exaone3.5:7.8b)가 이 머신엔 없어 자동으로 못 담았다.
ollama가 설치되고 `ollama pull exaone3.5:7.8b`를 실행해둔 머신에서
이 스크립트를 다시 돌리거나, 그 머신의 ~/.ollama/models 를 통째로 이 폴더에 복사해서 가져가라.
EOF
  echo "     ! ollama 없음 — README만 남김 (수동 반입 필요)"
fi

echo "[5/5] 실행 안내"
cp "$ROOT/scripts/OFFLINE_README.md" "$OUT/README.md"

du -sh "$OUT" 2>/dev/null || true
echo "완료: $OUT"
