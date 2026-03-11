"""오디오 분석기 테스트."""

import numpy as np
import pytest

from ai.audio_analyzer import AudioAnalyzer, AudioResult


class TestAudioResult:
    def test_defaults(self):
        r = AudioResult()
        assert r.level == 0.0
        assert r.beat is False
        assert r.spectrum == []
        assert r.bpm_estimate == 0.0

    def test_custom(self):
        r = AudioResult(level=0.8, beat=True, spectrum=[0.1] * 16, bpm_estimate=120.0)
        assert r.level == 0.8
        assert r.beat is True
        assert len(r.spectrum) == 16


class TestAudioAnalyzer:
    def test_initial_result(self):
        analyzer = AudioAnalyzer()
        r = analyzer.result
        assert r.level == 0.0
        assert r.beat is False

    def test_process_silence(self):
        analyzer = AudioAnalyzer()
        silence = np.zeros(1024, dtype=np.float32)
        analyzer._process(silence)
        r = analyzer.result
        assert r.level < 0.01
        assert r.beat is False
        assert len(r.spectrum) == 16

    def test_process_loud(self):
        analyzer = AudioAnalyzer()
        for _ in range(10):
            analyzer._process(np.zeros(1024, dtype=np.float32))

        loud = np.random.randn(1024).astype(np.float32) * 0.5
        analyzer._process(loud)
        r = analyzer.result
        assert r.level > 0.0
        assert len(r.spectrum) == 16

    def test_spectrum_bands(self):
        analyzer = AudioAnalyzer()
        tone = np.sin(2 * np.pi * 440 * np.arange(1024) / 22050).astype(np.float32)
        analyzer._process(tone)
        r = analyzer.result
        assert len(r.spectrum) == 16
        assert any(v > 0 for v in r.spectrum)
