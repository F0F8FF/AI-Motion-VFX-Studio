"""
MediaPipe Tasks API 기반 실시간 포즈/손/얼굴 감지 모듈.
PoseLandmarker + HandLandmarker + FaceLandmarker를 병렬로 추론하여 FPS 극대화.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"


@dataclass
class DetectionResult:
    pose_landmarks: list[list[float]] = field(default_factory=list)
    left_hand_landmarks: list[list[float]] = field(default_factory=list)
    right_hand_landmarks: list[list[float]] = field(default_factory=list)
    face_landmarks: list[list[float]] = field(default_factory=list)
    pose_world_landmarks: list[list[float]] = field(default_factory=list)


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class PoseDetector:
    """
    MediaPipe Tasks 기반 통합 감지기.
    Pose(33) + Hand(21x2) + Face(478) 랜드마크를 ThreadPoolExecutor로 병렬 추출.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        pose_model = str(MODEL_DIR / "pose_landmarker_heavy.task")
        hand_model = str(MODEL_DIR / "hand_landmarker.task")
        face_model = str(MODEL_DIR / "face_landmarker.task")

        for path, name in [(pose_model, "Pose"), (hand_model, "Hand"), (face_model, "Face")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} 모델 파일 없음: {path}")

        self._pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ))

        self._hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ))

        self._face = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ))

        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="mp")
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000) + self._frame_count
        self._frame_count += 1

        # 3개 모델을 병렬 추론 — 순차 대비 ~2-3x 빠름
        pose_future = self._executor.submit(self._detect_pose, mp_image, timestamp_ms)
        hand_future = self._executor.submit(self._detect_hands, mp_image, timestamp_ms)
        face_future = self._executor.submit(self._detect_face, mp_image, timestamp_ms)

        detection = DetectionResult()

        pose_landmarks, pose_world = pose_future.result()
        detection.pose_landmarks = pose_landmarks
        detection.pose_world_landmarks = pose_world

        left_hand, right_hand = hand_future.result()
        detection.left_hand_landmarks = left_hand
        detection.right_hand_landmarks = right_hand

        detection.face_landmarks = face_future.result()

        return detection

    def _detect_pose(self, mp_image: mp.Image, ts: int):
        result = self._pose.detect_for_video(mp_image, ts)
        landmarks, world = [], []
        if result.pose_landmarks:
            landmarks = [
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, "visibility") else 1.0]
                for lm in result.pose_landmarks[0]
            ]
        if result.pose_world_landmarks:
            world = [
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, "visibility") else 1.0]
                for lm in result.pose_world_landmarks[0]
            ]
        return landmarks, world

    def _detect_hands(self, mp_image: mp.Image, ts: int):
        result = self._hands.detect_for_video(mp_image, ts)
        left, right = [], []
        if result.hand_landmarks:
            for i, handedness_list in enumerate(result.handedness):
                label = handedness_list[0].category_name
                lms = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[i]]
                if label == "Right":
                    left = lms
                else:
                    right = lms
        return left, right

    def _detect_face(self, mp_image: mp.Image, ts: int):
        result = self._face.detect_for_video(mp_image, ts)
        if result.face_landmarks:
            return [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]]
        return []

    def draw_landmarks(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        if result.pose_landmarks:
            self._draw_connections(annotated, result.pose_landmarks, POSE_CONNECTIONS, (0, 255, 0), w, h, has_visibility=True)
        if result.left_hand_landmarks:
            self._draw_connections(annotated, result.left_hand_landmarks, HAND_CONNECTIONS, (255, 100, 0), w, h)
        if result.right_hand_landmarks:
            self._draw_connections(annotated, result.right_hand_landmarks, HAND_CONNECTIONS, (0, 100, 255), w, h)

        return annotated

    @staticmethod
    def _draw_connections(
        frame: np.ndarray,
        landmarks: list[list[float]],
        connections: list[tuple[int, int]],
        color: tuple,
        w: int,
        h: int,
        has_visibility: bool = False,
    ):
        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            pt1, pt2 = landmarks[start_idx], landmarks[end_idx]
            x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
            x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
            if has_visibility and min(pt1[3], pt2[3]) < 0.5:
                continue
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        for lm in landmarks:
            cv2.circle(frame, (int(lm[0] * w), int(lm[1] * h)), 3, color, -1)

    def close(self):
        self._executor.shutdown(wait=False)
        self._pose.close()
        self._hands.close()
        self._face.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
