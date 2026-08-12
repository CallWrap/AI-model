"""P1-b: STT 노이즈 증강기.

문제:
    학습 데이터는 LLM이 생성한 깨끗한 대본, 추론 입력은 STT + 화자분리 출력.
    이 train-serving skew를 방치하면 실서비스에서 성능이 무너진다.

해법:
    생성본을 규칙 기반으로 오염시켜 STT 출력처럼 보이게 만든 버전을 함께 학습한다.

전부 결정적(seed 고정 가능)이며 외부 의존성은 표준 라이브러리 + numpy뿐이다.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 도메인 용어 오인식 사전
#   전자바우처 도메인 고유명사는 STT가 자주 틀린다. 실제 STT 로그를 확보하면
#   이 표를 교체할 것. 그전까지는 음성학적으로 그럴듯한 후보를 수동 작성.
# ---------------------------------------------------------------------------
ASR_CONFUSIONS: dict[str, list[str]] = {
    "바우처": ["바우쳐", "바오처", "바우저", "바우처가"],
    "본인부담금": ["본인 부담큼", "본인부담 금", "보닌부담금", "본인 부담금"],
    "행복e음": ["행복이음", "행복 이음", "행복 e 음", "행복의음"],
    "가산단가": ["가산 단과", "가산단과", "가산 당가"],
    "제공기관": ["제공 기간", "제공기간", "재공기관"],
    "제공인력": ["제공 인력", "재공인력", "제공 임력"],
    "예탁금": ["에탁금", "예탁 금", "예탁근"],
    "활동지원": ["활동 지원", "활똥지원", "활동지언"],
    "결제단말기": ["결재단말기", "결제 단말기", "결제단마리"],
    "지문인식기": ["지문 인식기", "지문인식 기", "지문인식가"],
    "공동인증서": ["공인인증서", "공동 인증서", "공동인증 서"],
    "부정사용": ["부정 사용", "부정싸용", "부장사용"],
    "자격변경": ["자격 변경", "자겨변경"],
    "과오결제": ["과오 결제", "과오결재", "가오결제"],
    "환급": ["환금", "환굽", "환 급"],
    "지자체": ["지자채", "지 자체", "지자체가"],
    "이용권": ["이용 권", "이용꿘"],
    "급여등록": ["급여 등록", "그뎌등록"],
    "생성": ["샹성", "생 성"],
    "결제": ["결재", "겨제"],
    "승인": ["승인이", "성인", "승 인"],
    "조회": ["조회가", "저회", "조 회"],
    "신청": ["신청이", "심청", "신 청"],
}

# 발화 시작에 붙는 filler
LEADING_FILLERS = ["어", "음", "아", "그", "저", "예", "네", "아 예", "어 그",
                   "음 그", "네 그", "아니 그", "그 뭐냐", "잠시만요"]
# 문장 중간에 삽입되는 filler
MID_FILLERS = ["어", "음", "그", "이제", "좀", "뭐", "인제", "그니까"]
# 맞장구 / 확인 응답
BACKCHANNELS = ["네", "네네", "예", "아 네", "음", "어 네", "그렇죠", "맞아요"]

_PUNCT_RE = re.compile(r"[.,!?~…·\"'“”‘’\-—\(\)\[\]{}:;/]+")
_SPACE_RE = re.compile(r"\s{2,}")
_SPEAKER_RE = re.compile(r"^\s*\[(?P<spk>[^\]]+)\]\s*(?P<text>.*)$")

_NUM_KO = {
    "0": "영", "1": "일", "2": "이", "3": "삼", "4": "사",
    "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
}
_UNIT_SPACING = ["만원", "만 원", "원", "건", "명", "일", "개월", "시간", "회"]


@dataclass
class NoiseConfig:
    """각 값은 '해당 변형이 적용될 확률'. 0이면 비활성."""

    strip_punctuation: float = 1.0      # STT는 문장부호를 거의 못 낸다
    leading_filler: float = 0.30        # 발화 앞 filler 삽입
    mid_filler: float = 0.20            # 어절 사이 filler 삽입
    repeat_word: float = 0.12           # 자기수정/더듬기 "결제가 결제가"
    asr_confusion: float = 0.35         # 도메인 용어 오인식 치환
    number_corruption: float = 0.25     # 숫자 표기 흔들기
    drop_particle: float = 0.08         # 조사 탈락
    merge_split_space: float = 0.15     # 띄어쓰기 붕괴/과분리
    insert_backchannel: float = 0.10    # 맞장구 발화 삽입
    split_utterance: float = 0.12       # 한 발화를 두 발화로 분절
    speaker_flip: float = 0.04          # 화자분리(diarization) 오류 시뮬레이션
    drop_utterance: float = 0.02        # 발화 유실 (통화 끊김/무음)


@dataclass
class Utterance:
    speaker: str  # "고객" | "상담사"
    text: str


@dataclass
class STTNoiseAugmenter:
    config: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- 확률 헬퍼 -------------------------------------------------------
    def _hit(self, p: float) -> bool:
        return p > 0 and self._rng.random() < p

    # -- 토큰 수준 변형 ---------------------------------------------------
    def _strip_punct(self, text: str) -> str:
        text = _PUNCT_RE.sub(" ", text)
        return _SPACE_RE.sub(" ", text).strip()

    def _apply_confusions(self, text: str) -> str:
        for term, variants in ASR_CONFUSIONS.items():
            if term in text and self._hit(self.config.asr_confusion):
                text = text.replace(term, self._rng.choice(variants), 1)
        return text

    def _corrupt_numbers(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            if not self._hit(self.config.number_corruption):
                return m.group(0)
            digits = m.group(0)
            mode = self._rng.random()
            if mode < 0.4:  # 한글 음차: 30 -> 삼영
                return "".join(_NUM_KO.get(c, c) for c in digits)
            if mode < 0.7:  # 자릿수 분리: 30 -> 3 0
                return " ".join(digits)
            return digits  # 그대로 두되 뒤 단위 띄어쓰기는 별도 처리

        text = re.sub(r"\d+", repl, text)
        for unit in _UNIT_SPACING:
            if unit in text and self._hit(self.config.number_corruption * 0.5):
                text = text.replace(unit, f" {unit}", 1)
        return text

    def _insert_fillers(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        out: list[str] = []
        for w in words:
            if self._hit(self.config.mid_filler):
                out.append(self._rng.choice(MID_FILLERS))
            if self._hit(self.config.repeat_word) and len(w) > 1:
                out.append(w)  # 더듬기: 같은 어절 반복
            out.append(w)
        if self._hit(self.config.leading_filler):
            out.insert(0, self._rng.choice(LEADING_FILLERS))
        return " ".join(out)

    def _drop_particles(self, text: str) -> str:
        # 어말 조사 탈락. 과도하면 의미가 깨지므로 확률을 낮게 유지한다.
        particles = ("가", "이", "을", "를", "은", "는", "에", "의")
        words = text.split()
        for i, w in enumerate(words):
            if len(w) > 2 and w.endswith(particles) and self._hit(self.config.drop_particle):
                words[i] = w[:-1]
        return " ".join(words)

    def _mangle_spacing(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        out: list[str] = []
        i = 0
        while i < len(words):
            if i + 1 < len(words) and self._hit(self.config.merge_split_space):
                if self._rng.random() < 0.5:  # 붙여쓰기
                    out.append(words[i] + words[i + 1])
                    i += 2
                    continue
                w = words[i]  # 과분리
                if len(w) > 3:
                    cut = self._rng.randint(1, len(w) - 1)
                    out.extend([w[:cut], w[cut:]])
                    i += 1
                    continue
            out.append(words[i])
            i += 1
        return " ".join(out)

    # -- 발화 수준 --------------------------------------------------------
    def augment_text(self, text: str) -> str:
        """단일 발화 텍스트에 STT 노이즈를 적용한다."""
        if self.config.strip_punctuation >= 1.0 or self._hit(self.config.strip_punctuation):
            text = self._strip_punct(text)
        text = self._apply_confusions(text)
        text = self._corrupt_numbers(text)
        text = self._drop_particles(text)
        text = self._insert_fillers(text)
        text = self._mangle_spacing(text)
        return _SPACE_RE.sub(" ", text).strip()

    def augment_dialogue(self, utterances: list[Utterance]) -> list[Utterance]:
        """대화 전체에 발화 수준 변형까지 적용한다."""
        out: list[Utterance] = []
        for utt in utterances:
            if self._hit(self.config.drop_utterance):
                continue

            text = self.augment_text(utt.text)
            if not text:
                continue

            speaker = utt.speaker
            # diarization 오류: 인접 발화의 화자 라벨이 뒤집힌다.
            if self._hit(self.config.speaker_flip):
                speaker = "상담사" if speaker == "고객" else "고객"

            # 긴 발화가 두 발화로 쪼개지는 현상
            words = text.split()
            if self._hit(self.config.split_utterance) and len(words) > 6:
                cut = self._rng.randint(2, len(words) - 2)
                out.append(Utterance(speaker, " ".join(words[:cut])))
                out.append(Utterance(speaker, " ".join(words[cut:])))
            else:
                out.append(Utterance(speaker, text))

            # 상대방 맞장구 삽입
            if self._hit(self.config.insert_backchannel):
                other = "상담사" if speaker == "고객" else "고객"
                out.append(Utterance(other, self._rng.choice(BACKCHANNELS)))
        return out


# ---------------------------------------------------------------------------
# 직렬화 헬퍼: 대화 <-> 텍스트
# ---------------------------------------------------------------------------
def parse_dialogue(text: str) -> list[Utterance]:
    """`[고객] ...` / `[상담사] ...` 줄 단위 포맷을 파싱한다."""
    utterances: list[Utterance] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            utterances.append(Utterance(m.group("spk").strip(), m.group("text").strip()))
        elif utterances:  # 화자 태그 없는 줄은 직전 발화에 이어붙인다
            utterances[-1].text += " " + line
    return utterances


def render_dialogue(utterances: list[Utterance]) -> str:
    return "\n".join(f"[{u.speaker}] {u.text}" for u in utterances)


def augment_dialogue_text(text: str, config: NoiseConfig | None = None,
                          seed: int | None = None) -> str:
    """문자열 -> 문자열 편의 함수."""
    aug = STTNoiseAugmenter(config or NoiseConfig(), seed=seed)
    return render_dialogue(aug.augment_dialogue(parse_dialogue(text)))


if __name__ == "__main__":
    sample = """[고객] 안녕하세요, 바우처 결제가 자꾸 실패해서요.
[상담사] 네, 고객님. 혹시 지문인식기 인식은 정상적으로 되시나요?
[고객] 그게 빨간불이 계속 떠 있어요. 본인부담금 30,000원 결제를 못 하고 있습니다.
[상담사] 확인했습니다. 공동인증서 재설치 후 다시 시도해 보시겠어요?"""

    print("=== 원본 ===")
    print(sample)
    print("\n=== STT 노이즈 적용 ===")
    print(augment_dialogue_text(sample, seed=42))
