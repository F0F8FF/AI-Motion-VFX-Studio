"""
OSC 송신 모듈.
AI 분석 결과를 OSC 프로토콜로 TouchDesigner에 실시간 전송한다.

TouchDesigner에서 OSC In CHOP 또는 OSC In DAT으로 수신.
기본 포트: 9000
"""

from __future__ import annotations

from pythonosc import udp_client
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from ai.pose_detector import DetectionResult
from ai.gesture_classifier import GestureResult
from ai.emotion_detector import EmotionResult


class OSCSender:
    """
    OSC 메시지 구조:
        /pose/landmarks          - 33개 포즈 랜드마크 (x,y,z,vis 플랫)
        /pose/world              - 월드 좌표 포즈 랜드마크
        /hand/left/landmarks     - 왼손 21개 랜드마크
        /hand/right/landmarks    - 오른손 21개 랜드마크
        /gesture/left/name       - 왼손 제스처 이름
        /gesture/left/confidence - 왼손 제스처 신뢰도
        /gesture/right/name      - 오른손 제스처 이름
        /gesture/right/confidence- 오른손 제스처 신뢰도
        /emotion/name            - 감정 이름
        /emotion/confidence      - 감정 신뢰도
        /emotion/arousal         - 각성도 (0~1)
        /emotion/valence         - 긍정도 (-1~1)
        /face/landmarks          - 얼굴 주요 랜드마크 (선별 전송)
    """

    # 얼굴 랜드마크 중 TouchDesigner에서 유용한 것만 선별 (468개 전부 보내면 과부하)
    FACE_KEY_INDICES = [
        1, 4, 5, 6,           # 코
        33, 133, 362, 263,    # 눈 코너
        61, 291,              # 입 코너
        13, 14,               # 입 위아래
        70, 107, 336, 300,    # 눈썹
        10, 152,              # 이마, 턱
    ]

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self._client = udp_client.SimpleUDPClient(host, port)
        self._host = host
        self._port = port
        print(f"[OSC] 송신 준비 → {host}:{port}")

    def send_detection(self, detection: DetectionResult):
        bundle = OscBundleBuilder(IMMEDIATELY)

        if detection.pose_landmarks:
            flat = [v for lm in detection.pose_landmarks for v in lm]
            self._client.send_message("/pose/landmarks", flat)

        if detection.pose_world_landmarks:
            flat = [v for lm in detection.pose_world_landmarks for v in lm]
            self._client.send_message("/pose/world", flat)

        if detection.left_hand_landmarks:
            flat = [v for lm in detection.left_hand_landmarks for v in lm]
            self._client.send_message("/hand/left/landmarks", flat)

        if detection.right_hand_landmarks:
            flat = [v for lm in detection.right_hand_landmarks for v in lm]
            self._client.send_message("/hand/right/landmarks", flat)

        if detection.face_landmarks:
            key_pts = []
            for idx in self.FACE_KEY_INDICES:
                if idx < len(detection.face_landmarks):
                    key_pts.extend(detection.face_landmarks[idx])
            if key_pts:
                self._client.send_message("/face/landmarks", key_pts)

    def send_gesture(self, result: GestureResult, hand: str = "right"):
        prefix = f"/gesture/{hand}"
        self._client.send_message(f"{prefix}/name", result.gesture)
        self._client.send_message(f"{prefix}/confidence", result.confidence)

    def send_emotion(self, result: EmotionResult):
        self._client.send_message("/emotion/name", result.emotion)
        self._client.send_message("/emotion/confidence", result.confidence)
        self._client.send_message("/emotion/arousal", result.arousal)
        self._client.send_message("/emotion/valence", result.valence)

    def send_audio(self, level: float, beat: bool, spectrum: list[float] | None = None):
        self._client.send_message("/audio/level", level)
        self._client.send_message("/audio/beat", 1.0 if beat else 0.0)
        if spectrum:
            self._client.send_message("/audio/spectrum", spectrum[:16])

    def send_custom(self, address: str, value):
        self._client.send_message(address, value)

    @property
    def endpoint(self) -> str:
        return f"{self._host}:{self._port}"
