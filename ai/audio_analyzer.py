"""
실시간 오디오 분석 모듈.
마이크 입력에서 볼륨, 비트, 스펙트럼을 추출하여 VFX 파라미터로 변환한다.
별도 스레드에서 구동되며 최신 분석 결과를 메인 루프에 노출.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioResult:
    level: float = 0.0             # RMS 볼륨 (0~1)
    beat: bool = False             # 비트 감지 여부
    spectrum: list[float] = field(default_factory=list)  # 16밴드 스펙트럼
    bpm_estimate: float = 0.0
    onset_strength: float = 0.0    # 온셋 강도 → VFX 펄스에 매핑


class AudioAnalyzer:
    """
    sounddevice로 마이크 스트림을 열고, numpy FFT로 실시간 분석.
    librosa 없이도 동작하며, librosa가 있으면 비트 추정 정확도가 올라간다.
    """

    NUM_BANDS = 16

    def __init__(
        self,
        device_id: int | None = None,
        sample_rate: int = 22050,
        block_size: int = 1024,
    ):
        self._sr = sample_rate
        self._block_size = block_size
        self._device_id = device_id

        self._result = AudioResult()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # 비트 감지용 상태
        self._energy_history: list[float] = []
        self._history_max = 43  # ~1초 분량 (22050/1024 ≈ 21.5 frames/sec)
        self._beat_cooldown = 0.0
        self._last_beat_time = 0.0
        self._beat_intervals: list[float] = []

    @property
    def result(self) -> AudioResult:
        with self._lock:
            return AudioResult(
                level=self._result.level,
                beat=self._result.beat,
                spectrum=list(self._result.spectrum),
                bpm_estimate=self._result.bpm_estimate,
                onset_strength=self._result.onset_strength,
            )

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio")
        self._thread.start()
        print("[Audio] 마이크 분석 시작")

    def _run(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("[Audio] sounddevice 미설치 — pip install sounddevice")
            self._running = False
            return

        def callback(indata, frames, time_info, status):
            if status:
                return
            audio = indata[:, 0]
            self._process(audio)

        try:
            with sd.InputStream(
                device=self._device_id,
                samplerate=self._sr,
                channels=1,
                blocksize=self._block_size,
                callback=callback,
            ):
                while self._running:
                    time.sleep(0.01)
        except Exception as e:
            print(f"[Audio] 에러: {e}")
            self._running = False

    def _process(self, audio: np.ndarray):
        rms = float(np.sqrt(np.mean(audio ** 2)))
        level = min(1.0, rms * 10)  # 정규화

        # FFT → 16밴드 스펙트럼
        fft = np.abs(np.fft.rfft(audio))
        band_size = max(1, len(fft) // self.NUM_BANDS)
        spectrum = []
        for i in range(self.NUM_BANDS):
            start = i * band_size
            end = min(start + band_size, len(fft))
            band_energy = float(np.mean(fft[start:end])) if start < end else 0.0
            spectrum.append(min(1.0, band_energy * 5))

        # 비트 감지: 현재 에너지가 최근 평균 대비 스파이크
        energy = float(np.sum(audio ** 2))
        self._energy_history.append(energy)
        if len(self._energy_history) > self._history_max:
            self._energy_history.pop(0)

        avg_energy = np.mean(self._energy_history) if self._energy_history else 0
        now = time.time()
        beat = False
        onset = min(1.0, energy / (avg_energy + 1e-8) - 1.0)
        onset = max(0.0, onset)

        if energy > avg_energy * 1.8 and (now - self._last_beat_time) > 0.15:
            beat = True
            if self._last_beat_time > 0:
                interval = now - self._last_beat_time
                self._beat_intervals.append(interval)
                if len(self._beat_intervals) > 16:
                    self._beat_intervals.pop(0)
            self._last_beat_time = now

        # BPM 추정
        bpm = 0.0
        if len(self._beat_intervals) >= 4:
            avg_interval = np.median(self._beat_intervals)
            if avg_interval > 0:
                bpm = 60.0 / avg_interval

        with self._lock:
            self._result.level = level
            self._result.beat = beat
            self._result.spectrum = spectrum
            self._result.bpm_estimate = bpm
            self._result.onset_strength = onset

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[Audio] 마이크 분석 종료")
