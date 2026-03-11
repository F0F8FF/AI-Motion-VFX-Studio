"""감정 인식 모듈 테스트."""

import pytest

from ai.emotion_detector import EmotionDetector, EmotionResult


@pytest.fixture
def detector():
    return EmotionDetector()


class TestEmotionDetector:
    def test_insufficient_landmarks(self, detector):
        result = detector.detect([[0.0, 0.0, 0.0]] * 10)
        assert result.emotion == "neutral"
        assert result.confidence == 0.0

    def test_returns_valid_emotion(self, detector):
        landmarks = [[0.5, 0.5, 0.0]] * 478
        result = detector.detect(landmarks)
        assert result.emotion in EmotionDetector.EMOTIONS
        assert 0.0 <= result.confidence <= 1.0

    def test_arousal_range(self, detector):
        landmarks = [[0.5, 0.5, 0.0]] * 478
        result = detector.detect(landmarks)
        assert 0.0 <= result.arousal <= 1.0

    def test_valence_range(self, detector):
        landmarks = [[0.5, 0.5, 0.0]] * 478
        result = detector.detect(landmarks)
        assert -1.0 <= result.valence <= 1.0


class TestEmotionResult:
    def test_dataclass_fields(self):
        result = EmotionResult(emotion="happy", confidence=0.9, arousal=0.8, valence=0.5)
        assert result.emotion == "happy"
        assert result.confidence == 0.9
        assert result.arousal == 0.8
        assert result.valence == 0.5
