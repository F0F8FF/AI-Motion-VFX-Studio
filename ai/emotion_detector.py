"""
얼굴 랜드마크 기반 감정 인식 모듈.
MediaPipe Face Mesh의 468개 랜드마크에서 기하학적 특징을 추출하여 감정을 추정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    arousal: float  # 각성도 (0~1) — TouchDesigner VFX 강도에 매핑
    valence: float  # 긍정도 (-1~1) — TouchDesigner VFX 색조에 매핑


# Face Mesh 주요 인덱스
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_MOUTH_TOP = 13
_MOUTH_BOTTOM = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_LEFT_EYEBROW_INNER = 107
_RIGHT_EYEBROW_INNER = 336
_NOSE_TIP = 1
_LEFT_EYEBROW_OUTER = 70
_RIGHT_EYEBROW_OUTER = 300


class EmotionDetector:
    """
    기하학적 특징 기반 감정 추정기.
    눈 개폐도(EAR), 입 개폐도(MAR), 눈썹 높이 등을 분석하여
    6가지 기본 감정 + neutral을 추정한다.
    """

    EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]

    def __init__(self):
        self._prev_emotion = "neutral"
        self._smoothing = 0.3

    def detect(self, face_landmarks: list[list[float]]) -> EmotionResult:
        if len(face_landmarks) < 468:
            return EmotionResult("neutral", 0.0, 0.5, 0.0)

        pts = np.array(face_landmarks)

        ear = self._eye_aspect_ratio(pts)
        mar = self._mouth_aspect_ratio(pts)
        brow_height = self._eyebrow_height(pts)
        mouth_width = self._mouth_width(pts)

        scores = self._compute_emotion_scores(ear, mar, brow_height, mouth_width)

        best_emotion = max(scores, key=scores.get)
        best_conf = scores[best_emotion]

        arousal = np.clip(mar * 2 + (1 - ear) * 0.5 + abs(brow_height - 0.5), 0, 1)
        valence = scores.get("happy", 0) - scores.get("sad", 0) - scores.get("angry", 0) * 0.5

        return EmotionResult(
            emotion=best_emotion,
            confidence=float(best_conf),
            arousal=float(arousal),
            valence=float(np.clip(valence, -1, 1)),
        )

    def _eye_aspect_ratio(self, pts: np.ndarray) -> float:
        left_ear = np.linalg.norm(pts[_LEFT_EYE_TOP] - pts[_LEFT_EYE_BOTTOM])
        right_ear = np.linalg.norm(pts[_RIGHT_EYE_TOP] - pts[_RIGHT_EYE_BOTTOM])
        return float((left_ear + right_ear) / 2)

    def _mouth_aspect_ratio(self, pts: np.ndarray) -> float:
        vertical = np.linalg.norm(pts[_MOUTH_TOP] - pts[_MOUTH_BOTTOM])
        horizontal = np.linalg.norm(pts[_MOUTH_LEFT] - pts[_MOUTH_RIGHT])
        if horizontal < 1e-6:
            return 0.0
        return float(vertical / horizontal)

    def _mouth_width(self, pts: np.ndarray) -> float:
        return float(np.linalg.norm(pts[_MOUTH_LEFT] - pts[_MOUTH_RIGHT]))

    def _eyebrow_height(self, pts: np.ndarray) -> float:
        left = np.linalg.norm(pts[_LEFT_EYEBROW_INNER] - pts[_LEFT_EYE_TOP])
        right = np.linalg.norm(pts[_RIGHT_EYEBROW_INNER] - pts[_RIGHT_EYE_TOP])
        return float((left + right) / 2)

    def _compute_emotion_scores(
        self, ear: float, mar: float, brow_height: float, mouth_width: float
    ) -> dict[str, float]:
        scores = {e: 0.1 for e in self.EMOTIONS}

        # Happy: 입 넓게 벌림 + 눈 살짝 감김
        if mouth_width > 0.08 and mar > 0.05:
            scores["happy"] = 0.5 + min(mouth_width * 3, 0.4)

        # Surprise: 눈 크게 뜸 + 입 벌림 + 눈썹 올라감
        if ear > 0.035 and mar > 0.15 and brow_height > 0.04:
            scores["surprise"] = 0.5 + min(ear * 5, 0.4)

        # Sad: 입꼬리 내려감 + 눈썹 내려감
        if brow_height < 0.03 and mouth_width < 0.07:
            scores["sad"] = 0.5 + max(0.03 - brow_height, 0) * 10

        # Angry: 눈썹 찌푸림 + 입 꽉 다묾
        if brow_height < 0.025 and mar < 0.05:
            scores["angry"] = 0.5 + max(0.025 - brow_height, 0) * 15

        # Neutral
        if max(scores.values()) < 0.4:
            scores["neutral"] = 0.7

        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}
