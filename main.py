"""
AI Motion VFX Studio — 메인 파이프라인.

웹캠 → MediaPipe 포즈/손/얼굴 감지 → 제스처 분류 → 감정 인식 → OSC 전송 → TouchDesigner

사용법:
    python main.py                       # 기본 실행
    python main.py --no-osc              # OSC 비활성화
    python main.py --dashboard           # 웹 대시보드 함께 실행
    python main.py --audio               # 마이크 오디오 분석 활성화
    python main.py --record session1     # 세션 녹화
    python main.py --play session1       # 세션 재생 (카메라 불필요)
    python main.py --collect wave        # 제스처 데이터 수집
    python main.py --train               # 수집된 데이터로 모델 학습
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import cv2
import numpy as np

from config import AppConfig
from ai.pose_detector import PoseDetector, DetectionResult
from ai.gesture_classifier import GestureClassifier, GestureResult
from ai.emotion_detector import EmotionDetector, EmotionResult
from ai.audio_analyzer import AudioAnalyzer, AudioResult
from osc.sender import OSCSender
from session import SessionRecorder, SessionPlayer


class MotionVFXPipeline:
    """메인 파이프라인. 모든 AI 모듈, OSC 송신, 오디오, 세션을 통합 관리."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._running = False

        self.detector = PoseDetector(
            min_detection_confidence=config.ai.min_detection_confidence,
            min_tracking_confidence=config.ai.min_tracking_confidence,
        )
        self.gesture = GestureClassifier()
        self.emotion = EmotionDetector()

        self.audio: AudioAnalyzer | None = None
        if config.audio.enabled:
            self.audio = AudioAnalyzer(
                device_id=config.audio.device_id,
                sample_rate=config.audio.sample_rate,
                block_size=config.audio.block_size,
            )

        self.osc: OSCSender | None = None
        if config.osc.enabled:
            self.osc = OSCSender(config.osc.host, config.osc.port)

        self._latest_state: dict = {}
        self._latest_frame: np.ndarray | None = None
        self._state_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._fps = 0.0

    @property
    def latest_state(self) -> dict:
        with self._state_lock:
            return self._latest_state.copy()

    @property
    def latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def run(
        self,
        collect_gesture: str | None = None,
        record_path: str | None = None,
        play_path: str | None = None,
    ):
        player: SessionPlayer | None = None
        if play_path:
            player = SessionPlayer(f"data/sessions/{play_path}.jsonl")

        recorder: SessionRecorder | None = None
        if record_path:
            recorder = SessionRecorder(f"data/sessions/{record_path}.jsonl")

        cap = None
        if not player:
            cap = cv2.VideoCapture(self.config.camera.device_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            if not cap.isOpened():
                print("[ERROR] 카메라를 열 수 없습니다.")
                sys.exit(1)

        if collect_gesture:
            self.gesture.start_collecting()
            print(f"\n[수집 모드] '{collect_gesture}' 제스처를 보여주세요.")
            print("  → 's' 키: 현재 프레임 저장 | 'q' 키: 종료\n")

        if self.audio:
            self.audio.start()

        self._running = True
        prev_time = time.time()
        frame_count = 0

        print("\n╔══════════════════════════════════════╗")
        print("║     AI Motion VFX Studio 실행중      ║")
        print("║                                      ║")
        if self.osc:
            print(f"║  OSC → {self.osc.endpoint:<28s} ║")
        if self.audio:
            print("║  Audio → 마이크 분석 ON              ║")
        if player:
            print(f"║  Play → {play_path:<27s} ║")
        if recorder:
            print(f"║  Rec  → {record_path:<27s} ║")
        print("║  'q' = 종료 | 'd' = 디버그 토글     ║")
        print("╚══════════════════════════════════════╝\n")

        show_debug = True

        try:
            while self._running:
                # --- 재생 모드 ---
                if player:
                    state = player.next_frame()
                    if state is None:
                        time.sleep(0.001)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        continue
                    with self._state_lock:
                        self._latest_state = state
                    if self.osc:
                        self._send_state_osc(state)
                    time.sleep(0.03)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                # --- 카메라 모드 ---
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                detection = self.detector.detect(frame)

                left_gesture = GestureResult("none", 0.0, {})
                right_gesture = GestureResult("none", 0.0, {})
                if self.config.ai.gesture_enabled:
                    if detection.left_hand_landmarks:
                        left_gesture = self.gesture.classify(detection.left_hand_landmarks)
                    if detection.right_hand_landmarks:
                        right_gesture = self.gesture.classify(detection.right_hand_landmarks)

                emotion_result = EmotionResult("neutral", 0.0, 0.5, 0.0)
                if self.config.ai.emotion_enabled and detection.face_landmarks:
                    emotion_result = self.emotion.detect(detection.face_landmarks)

                audio_result = self.audio.result if self.audio else AudioResult()

                # OSC 전송
                if self.osc:
                    self.osc.send_detection(detection)
                    if detection.left_hand_landmarks:
                        self.osc.send_gesture(left_gesture, "left")
                    if detection.right_hand_landmarks:
                        self.osc.send_gesture(right_gesture, "right")
                    if detection.face_landmarks:
                        self.osc.send_emotion(emotion_result)
                    if self.audio:
                        self.osc.send_audio(audio_result.level, audio_result.beat, audio_result.spectrum)

                # 상태 업데이트
                state = {
                    "fps": self._fps,
                    "pose_detected": bool(detection.pose_landmarks),
                    "left_hand": bool(detection.left_hand_landmarks),
                    "right_hand": bool(detection.right_hand_landmarks),
                    "face_detected": bool(detection.face_landmarks),
                    "left_gesture": {"name": left_gesture.gesture, "confidence": left_gesture.confidence},
                    "right_gesture": {"name": right_gesture.gesture, "confidence": right_gesture.confidence},
                    "emotion": {
                        "name": emotion_result.emotion,
                        "confidence": emotion_result.confidence,
                        "arousal": emotion_result.arousal,
                        "valence": emotion_result.valence,
                    },
                    "audio": {
                        "level": audio_result.level,
                        "beat": audio_result.beat,
                        "spectrum": audio_result.spectrum,
                        "bpm": audio_result.bpm_estimate,
                    },
                }

                with self._state_lock:
                    self._latest_state = state

                if recorder:
                    recorder.record(state)

                # 프레임 렌더링 + 대시보드용 저장
                if show_debug:
                    display = self.detector.draw_landmarks(frame, detection)
                    self._draw_hud(display, detection, left_gesture, right_gesture, emotion_result, audio_result)
                else:
                    display = frame

                with self._frame_lock:
                    self._latest_frame = display

                cv2.imshow("AI Motion VFX Studio", display)

                if collect_gesture and detection.right_hand_landmarks:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        self.gesture.collect_sample(detection.right_hand_landmarks, collect_gesture)
                        print(f"  ✓ 샘플 저장됨 ({collect_gesture})")

                frame_count += 1
                now = time.time()
                if now - prev_time >= 1.0:
                    self._fps = frame_count / (now - prev_time)
                    frame_count = 0
                    prev_time = now

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    show_debug = not show_debug

        finally:
            self._running = False
            if collect_gesture:
                self.gesture.save_collected_data("data/gesture_data.json")
            if recorder:
                recorder.close()
            if self.audio:
                self.audio.stop()
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

    def _send_state_osc(self, state: dict):
        """재생 모드에서 저장된 상태를 OSC로 재전송."""
        if not self.osc:
            return
        if "emotion" in state:
            e = state["emotion"]
            self.osc.send_custom("/emotion/name", e.get("name", "neutral"))
            self.osc.send_custom("/emotion/arousal", e.get("arousal", 0.5))
            self.osc.send_custom("/emotion/valence", e.get("valence", 0.0))
        if "audio" in state:
            a = state["audio"]
            self.osc.send_audio(a.get("level", 0), a.get("beat", False), a.get("spectrum", []))

    def _draw_hud(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        left_gesture: GestureResult,
        right_gesture: GestureResult,
        emotion: EmotionResult,
        audio: AudioResult,
    ):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        hud_h = 200 if audio.level > 0 else 180
        cv2.rectangle(overlay, (10, 10), (320, hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 35
        color = (0, 255, 200)
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += 25

        status_items = [
            ("POSE", bool(detection.pose_landmarks)),
            ("L-HAND", bool(detection.left_hand_landmarks)),
            ("R-HAND", bool(detection.right_hand_landmarks)),
            ("FACE", bool(detection.face_landmarks)),
        ]
        x = 20
        for label, active in status_items:
            c = (0, 255, 0) if active else (0, 0, 150)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
            x += 70
        y += 30

        if right_gesture.gesture != "none":
            cv2.putText(frame, f"R: {right_gesture.gesture} ({right_gesture.confidence:.0%})",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            y += 25

        if left_gesture.gesture != "none":
            cv2.putText(frame, f"L: {left_gesture.gesture} ({left_gesture.confidence:.0%})",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 1)
            y += 25

        if emotion.emotion != "neutral":
            cv2.putText(frame, f"Emotion: {emotion.emotion} ({emotion.confidence:.0%})",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 255), 1)
            y += 25

        bar_w = int(emotion.arousal * 100)
        cv2.rectangle(frame, (20, y), (20 + bar_w, y + 8), (0, 100, 255), -1)
        cv2.putText(frame, "arousal", (125, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        y += 20

        # 오디오 스펙트럼 바
        if audio.spectrum:
            bar_x = 20
            for i, val in enumerate(audio.spectrum[:16]):
                bar_h = int(val * 30)
                c = (0, 200, 255) if audio.beat else (100, 100, 100)
                cv2.rectangle(frame, (bar_x, y + 30 - bar_h), (bar_x + 12, y + 30), c, -1)
                bar_x += 18
            if audio.bpm_estimate > 0:
                cv2.putText(frame, f"{audio.bpm_estimate:.0f}BPM", (bar_x + 5, y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="AI Motion VFX Studio")
    parser.add_argument("--no-osc", action="store_true", help="OSC 전송 비활성화")
    parser.add_argument("--osc-host", default="127.0.0.1", help="OSC 대상 호스트")
    parser.add_argument("--osc-port", type=int, default=9000, help="OSC 대상 포트")
    parser.add_argument("--camera", type=int, default=0, help="카메라 디바이스 ID")
    parser.add_argument("--collect", type=str, default=None, help="제스처 데이터 수집 모드")
    parser.add_argument("--train", action="store_true", help="수집된 데이터로 모델 학습")
    parser.add_argument("--dashboard", action="store_true", help="웹 대시보드 함께 실행")
    parser.add_argument("--audio", action="store_true", help="마이크 오디오 분석 활성화")
    parser.add_argument("--record", type=str, default=None, help="세션 녹화 (이름)")
    parser.add_argument("--play", type=str, default=None, help="세션 재생 (이름)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    args = parser.parse_args()

    if args.train:
        print("[학습 모드]")
        classifier = GestureClassifier()
        classifier.train(
            data_path="data/gesture_data.json",
            epochs=100,
            save_path="data/gesture_model.pt",
        )
        return

    config = AppConfig()
    config.camera.device_id = args.camera
    config.osc.enabled = not args.no_osc
    config.osc.host = args.osc_host
    config.osc.port = args.osc_port
    config.audio.enabled = args.audio
    config.debug = args.debug

    pipeline = MotionVFXPipeline(config)

    def signal_handler(sig, frame):
        print("\n[종료 중...]")
        pipeline.stop()

    signal.signal(signal.SIGINT, signal_handler)

    if args.dashboard:
        from dashboard.app import create_app
        import uvicorn

        app = create_app(pipeline)
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": config.dashboard.host, "port": config.dashboard.port, "log_level": "warning"},
            daemon=True,
        )
        server_thread.start()
        print(f"[Dashboard] http://localhost:{config.dashboard.port}")

    pipeline.run(
        collect_gesture=args.collect,
        record_path=args.record,
        play_path=args.play,
    )


if __name__ == "__main__":
    main()
