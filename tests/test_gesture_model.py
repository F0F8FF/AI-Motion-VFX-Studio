"""PyTorch 제스처 모델 아키텍처 테스트."""

import torch
import pytest

from models.gesture_model import GestureNet


class TestGestureNet:
    def test_output_shape(self):
        model = GestureNet(input_dim=63, num_classes=8)
        model.eval()
        x = torch.randn(1, 63)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8)

    def test_batch_inference(self):
        model = GestureNet()
        x = torch.randn(32, 63)
        out = model(x)
        assert out.shape == (32, 8)

    def test_num_classes_property(self):
        model = GestureNet(num_classes=10)
        assert model.num_classes == 10

    def test_gesture_classes_defined(self):
        assert len(GestureNet.GESTURE_CLASSES) == 8
        assert "open_palm" in GestureNet.GESTURE_CLASSES
        assert "fist" in GestureNet.GESTURE_CLASSES
        assert "none" in GestureNet.GESTURE_CLASSES

    def test_softmax_sums_to_one(self):
        model = GestureNet()
        model.eval()
        x = torch.randn(1, 63)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_custom_dimensions(self):
        model = GestureNet(input_dim=126, num_classes=12)
        x = torch.randn(4, 126)
        out = model(x)
        assert out.shape == (4, 12)
