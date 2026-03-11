"""제스처 분류기 테스트."""

import numpy as np
import pytest

from ai.gesture_classifier import GestureClassifier


@pytest.fixture
def classifier():
    return GestureClassifier()


class TestNormalizeLandmarks:
    def test_output_shape(self):
        landmarks = [[float(i), float(i), float(i)] for i in range(21)]
        result = GestureClassifier.normalize_landmarks(landmarks)
        assert result.shape == (63,)

    def test_wrist_at_origin(self):
        """정규화 후 손목(index 0)은 원점이어야 한다."""
        landmarks = [[0.5, 0.5, 0.0]] + [[float(i), float(i), 0.0] for i in range(1, 21)]
        result = GestureClassifier.normalize_landmarks(landmarks)
        assert abs(result[0]) < 1e-6
        assert abs(result[1]) < 1e-6
        assert abs(result[2]) < 1e-6

    def test_scale_invariance(self):
        """크기가 달라도 정규화 결과는 동일해야 한다."""
        base = [[float(i) * 0.01, float(i) * 0.01, 0.0] for i in range(21)]
        scaled = [[x * 3, y * 3, z * 3] for x, y, z in base]

        norm_base = GestureClassifier.normalize_landmarks(base)
        norm_scaled = GestureClassifier.normalize_landmarks(scaled)
        np.testing.assert_allclose(norm_base, norm_scaled, atol=1e-5)

    def test_translation_invariance(self):
        """위치가 달라도 정규화 결과는 동일해야 한다."""
        base = [[float(i) * 0.01, float(i) * 0.01, 0.0] for i in range(21)]
        translated = [[x + 5.0, y + 3.0, z + 1.0] for x, y, z in base]

        norm_base = GestureClassifier.normalize_landmarks(base)
        norm_translated = GestureClassifier.normalize_landmarks(translated)
        np.testing.assert_allclose(norm_base, norm_translated, atol=1e-5)


class TestRuleBasedClassify:
    def _make_hand(self, fingers_up: list[bool]) -> list[list[float]]:
        """테스트용 손 랜드마크 생성. fingers_up = [thumb, index, middle, ring, pinky]"""
        landmarks = [[0.5, 0.8, 0.0]] * 21  # 기본값: 모든 관절 같은 위치

        # 손목
        landmarks[0] = [0.5, 0.8, 0.0]

        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]

        for i, (tip, pip_) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:  # 엄지는 x축 기반
                if fingers_up[i]:
                    landmarks[tip] = [0.2, 0.6, 0.0]
                    landmarks[pip_] = [0.4, 0.6, 0.0]
                else:
                    landmarks[tip] = [0.45, 0.6, 0.0]
                    landmarks[pip_] = [0.4, 0.6, 0.0]
            else:
                if fingers_up[i]:
                    landmarks[tip] = [0.3 + i * 0.1, 0.2, 0.0]   # 높이 올라감 (y 작음)
                    landmarks[pip_] = [0.3 + i * 0.1, 0.5, 0.0]
                else:
                    landmarks[tip] = [0.3 + i * 0.1, 0.7, 0.0]   # 접혀있음 (y 큼)
                    landmarks[pip_] = [0.3 + i * 0.1, 0.5, 0.0]

        return landmarks

    def test_open_palm(self, classifier):
        hand = self._make_hand([True, True, True, True, True])
        result = classifier._rule_based_classify(hand)
        assert result.gesture == "open_palm"

    def test_fist(self, classifier):
        hand = self._make_hand([False, False, False, False, False])
        result = classifier._rule_based_classify(hand)
        assert result.gesture == "fist"

    def test_peace(self, classifier):
        hand = self._make_hand([False, True, True, False, False])
        result = classifier._rule_based_classify(hand)
        assert result.gesture == "peace"

    def test_point(self, classifier):
        hand = self._make_hand([False, True, False, False, False])
        result = classifier._rule_based_classify(hand)
        assert result.gesture == "point"

    def test_invalid_input(self, classifier):
        result = classifier.classify([[0.0, 0.0, 0.0]] * 5)  # 21개가 아님
        assert result.gesture == "none"


class TestGestureResult:
    def test_classify_returns_all_fields(self, classifier):
        hand = [[float(i) * 0.01, float(i) * 0.02, 0.0] for i in range(21)]
        result = classifier.classify(hand)
        assert hasattr(result, "gesture")
        assert hasattr(result, "confidence")
        assert hasattr(result, "all_scores")
        assert isinstance(result.gesture, str)
        assert 0.0 <= result.confidence <= 1.0
