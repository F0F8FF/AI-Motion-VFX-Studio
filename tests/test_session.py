"""세션 녹화/재생 테스트."""

import json
import os
import tempfile

import pytest

from session import SessionRecorder, SessionPlayer


class TestSessionRecorder:
    def test_record_and_read(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name

        try:
            recorder = SessionRecorder(path)
            recorder.record({"fps": 30.0, "pose_detected": True})
            recorder.record({"fps": 29.5, "pose_detected": False})
            recorder.close()

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            first = json.loads(lines[0])
            assert first["fps"] == 30.0
            assert first["frame"] == 0
            assert "t" in first

            second = json.loads(lines[1])
            assert second["frame"] == 1
        finally:
            os.unlink(path)

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            with SessionRecorder(path) as rec:
                rec.record({"test": True})

            with open(path) as f:
                data = json.loads(f.readline())
            assert data["test"] is True
        finally:
            os.unlink(path)


class TestSessionPlayer:
    def _create_session(self, num_frames: int = 5) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            for i in range(num_frames):
                entry = {"t": i * 0.033, "frame": i, "fps": 30.0}
                f.write(json.dumps(entry) + "\n")
            return f.name

    def test_load(self):
        path = self._create_session(10)
        try:
            player = SessionPlayer(path)
            assert player.total_frames == 10
            assert player.duration > 0
        finally:
            os.unlink(path)

    def test_immediate_playback(self):
        path = self._create_session(3)
        try:
            player = SessionPlayer(path)
            frames = [player.get_frame_immediate() for _ in range(3)]
            assert all(f is not None for f in frames)
            assert frames[0]["frame"] == 0
            assert frames[2]["frame"] == 2
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SessionPlayer("/nonexistent/path.jsonl")

    def test_loop(self):
        path = self._create_session(2)
        try:
            player = SessionPlayer(path)
            player.loop = True
            for _ in range(5):
                f = player.get_frame_immediate()
                assert f is not None
        finally:
            os.unlink(path)
