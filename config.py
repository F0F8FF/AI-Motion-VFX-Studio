from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class OSCConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    enabled: bool = True


@dataclass
class AIConfig:
    pose_enabled: bool = True
    hand_enabled: bool = True
    face_enabled: bool = True
    gesture_enabled: bool = True
    emotion_enabled: bool = True
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AudioConfig:
    enabled: bool = False
    device_id: int | None = None
    sample_rate: int = 22050
    block_size: int = 1024


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    osc: OSCConfig = field(default_factory=OSCConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    debug: bool = False
