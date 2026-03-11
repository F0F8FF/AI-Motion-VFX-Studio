"""
세션 녹화/재생 모듈.
AI 분석 결과를 JSON Lines으로 저장하고, 카메라 없이 재생할 수 있다.
데모 영상 제작, 디버깅, 오프라인 테스트에 활용.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class SessionRecorder:
    """AI 파이프라인 상태를 .jsonl 파일로 기록."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")
        self._start_time = time.time()
        self._frame_count = 0
        print(f"[Session] 녹화 시작 → {path}")

    def record(self, state: dict):
        entry = {
            "t": round(time.time() - self._start_time, 4),
            "frame": self._frame_count,
            **state,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._frame_count += 1

    def close(self):
        self._file.close()
        print(f"[Session] 녹화 종료 ({self._frame_count} 프레임)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SessionPlayer:
    """
    녹화된 .jsonl 파일을 원래 타이밍대로 재생.
    OSC 전송/대시보드와 연결하면 카메라 없이 TouchDesigner 테스트 가능.
    """

    def __init__(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"세션 파일 없음: {path}")
        with open(path, encoding="utf-8") as f:
            self._frames = [json.loads(line) for line in f if line.strip()]
        self._index = 0
        self._start_time: float | None = None
        self.loop = True
        print(f"[Session] 재생 로드: {path} ({len(self._frames)} 프레임)")

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def duration(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1].get("t", 0.0)

    def reset(self):
        self._index = 0
        self._start_time = None

    def next_frame(self) -> dict | None:
        """타이밍에 맞는 다음 프레임을 반환. 아직 시간이 안됐으면 None."""
        if not self._frames:
            return None

        if self._start_time is None:
            self._start_time = time.time()

        elapsed = time.time() - self._start_time
        frame = self._frames[self._index]
        frame_time = frame.get("t", 0.0)

        if elapsed < frame_time:
            return None

        self._index += 1
        if self._index >= len(self._frames):
            if self.loop:
                self._index = 0
                self._start_time = time.time()
            else:
                return None

        return frame

    def get_frame_immediate(self) -> dict | None:
        """타이밍 무시하고 다음 프레임 즉시 반환."""
        if not self._frames:
            return None

        frame = self._frames[self._index]
        self._index = (self._index + 1) % len(self._frames)
        return frame
