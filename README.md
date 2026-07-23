# ai/

`Transcript`를 받아 후처리 폼(요약·유형·분류·등급·후속처리)을 채우는 AI 분석 파트. 파이프라인의 ③.

**담당:** 김성진 (F1 요약, F2 유형·분류, F4 후속판정, 캘리브레이션, PII 재탐지), 이재혁 (F3 상담등급·악성콜)

## 할 일

- `analyze(transcript) -> AnalysisResult` 함수 하나만 외부에 노출 (무상태 — DB/HTTP/Celery 모름)
- `schemas.py` — `Transcript`, `AnalysisResult` 입출력 계약 (4명 공용, 변경 시 전원 리뷰)
- F1+F4: LLM 요약 + 후속처리 판정 (1회 호출로 통합)
- F2: 상담유형 8종·상담분류(중분류 6종 + 소분류 29종) 분류 (LLM few-shot vs 파인튜닝 비교)
- F3: 상담등급·악성콜 감지
- FAQ 생성 필요 판단 (7/14 회의 확정 스코프, 담당 미정) — F1·F4와 같은 LLM 호출에 통합 검토
- 캘리브레이션(temperature scaling, ECE)
- PII 재탐지(NER)·k-익명성 — 가명처리는 이노하이가 하고, 여기서는 수령 데이터에
  PII가 남았는지 검증(재탐지)하고 축적·산출 데이터의 개인 특정 가능성(k-익명성)을 측정
- LLM 백엔드 추상화 (Anthropic API ↔ Ollama 온프렘 스위칭)
- `eval/` — 골든셋 기반 평가 CLI

`data/`는 로컬 전용(.gitignore). 음성·기관 데이터는 절대 커밋 금지.
