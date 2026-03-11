"""설정 모듈 테스트."""

from config import AppConfig, CameraConfig, OSCConfig, AIConfig


class TestAppConfig:
    def test_default_values(self):
        config = AppConfig()
        assert config.camera.device_id == 0
        assert config.camera.width == 1280
        assert config.camera.height == 720
        assert config.osc.host == "127.0.0.1"
        assert config.osc.port == 9000
        assert config.osc.enabled is True
        assert config.debug is False

    def test_custom_values(self):
        config = AppConfig(
            camera=CameraConfig(device_id=1, width=1920, height=1080),
            osc=OSCConfig(host="192.168.1.100", port=7000),
        )
        assert config.camera.device_id == 1
        assert config.osc.port == 7000

    def test_ai_defaults(self):
        config = AppConfig()
        assert config.ai.pose_enabled is True
        assert config.ai.gesture_enabled is True
        assert config.ai.emotion_enabled is True
        assert config.ai.min_detection_confidence == 0.7
