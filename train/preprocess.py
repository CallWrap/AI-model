"""대본 전처리 — 학습·추론이 공유한다.

512토큰 제약 대응. 대본 최대 길이가 663토큰이라 정형 인사말만 걷어내도
512 초과가 51건 → 23건으로 줄고, 초과분도 평균 64 → 42토큰으로 준다.
버리는 것은 "○○상담센터입니다", "감사합니다" 같은 분류 단서가 없는 줄뿐이다.

주의: 여기를 고치면 학습 때와 추론 때의 입력이 달라진다. 서비스(predict)를
만들 때 이 모듈을 그대로 import 해서 쓴다. 복사해 쓰면 조용히 틀린다.
"""

import re

# 도입 인사 — 소속 안내만 있는 줄
OPENING = re.compile(r"(상담센터입니다|무엇을 도와드릴|안녕하세요.{0,20}상담)")
# 마무리 인사 — 실질 내용 없는 종료 발화
CLOSING = re.compile(r"(감사합니다|좋은 하루|연락 주세요|안녕히|도움이 되셨|수고하세요|고맙습니다)")

# 대본에만 있는 감정·비언어 표지. 실제 STT 전사에는 나오지 않으므로 감정분석 전에 지운다.
# 내용을 담은 괄호((보호자)·(잠시 후)·전화번호 등)는 남긴다.
EMOTION_MARK = re.compile(
    r"\(\s*(욕설|한숨|침묵|정적|울먹|흐느낌|짜증|격앙|고성|버럭|화남|비웃음|웃음|헛웃음|목소리 높임|언성)\s*\)"
)


def strip_boilerplate(text: str) -> str:
    """앞뒤의 정형 인사말 줄을 제거한다. 실질 내용이 섞인 긴 줄은 남긴다."""
    lines = [ln for ln in text.split("\n") if ln.strip()]

    while lines and OPENING.search(lines[0]) and len(lines[0]) < 80:
        lines.pop(0)
    while lines and CLOSING.search(lines[-1]) and len(lines[-1]) < 90:
        lines.pop()

    return "\n".join(lines)


def utterances_to_text(utterances) -> str:
    """STT 출력(Utterance 목록) → 학습 때 쓴 '화자: 발화' 평문.

    whisper는 대략 문장 단위로 자르므로 한 사람이 이어 말한 내용이 여러 발화로
    쪼개진다. 학습 대본은 한 턴이 한 줄이었으므로 **연속된 같은 화자 발화를 합친다.**
    형태를 맞추는 목적도 있지만, 합치지 않으면 "상담사: "가 줄마다 붙어 512토큰
    한도를 본문 대신 화자 표기가 잡아먹는다.

    utterances는 Utterance 객체 목록이거나 STT가 준 dict 목록이면 된다.
    """
    lines: list[str] = []
    prev: str | None = None

    for u in utterances:
        speaker = u["speaker"] if isinstance(u, dict) else u.speaker
        text = (u["text"] if isinstance(u, dict) else u.text).strip()
        speaker = getattr(speaker, "value", speaker)  # Enum이면 문자열로
        if not text:
            continue
        if speaker == prev:
            lines[-1] += " " + text
        else:
            lines.append(f"{speaker}: {text}")
            prev = speaker

    return "\n".join(lines)


def strip_emotion_markers(text: str) -> str:
    """대본의 감정 표지를 지운다 — `(욕설)`, `(한숨)` 등.

    생성 대본은 감정 상태를 괄호로 적어 두지만 실제 상담 전사에는 그런 표지가 없다.
    이걸 남긴 채 학습하면 모델이 표지에 의존해 실데이터에서 무너진다.

    포함/제거 두 버전의 점수 차이가 곧 "재현 데이터가 얼마나 친절했는지"이므로,
    비교를 위해 함수로 분리해 둔다.
    """
    return re.sub(r"\s{2,}", " ", EMOTION_MARK.sub("", text)).strip()
