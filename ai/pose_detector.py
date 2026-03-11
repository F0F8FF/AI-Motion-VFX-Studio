"""
MediaPipe Tasks API 기반 실시간 포즈/손/얼굴 감지 모듈.
PoseLandmarker + HandLandmarker + FaceLandmarker를 VIDEO 모드로 통합 운영.
"""

from __future__ import annotations

import os
import time
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


# 포즈 연결 (시각화용)
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
    """MediaPipe Tasks 기반 통합 감지기. Pose(33) + Hand(21x2) + Face(478) 랜드마크 추출."""

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

        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000) + self._frame_count
        self._frame_count += 1

        detection = DetectionResult()

        pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)
        if pose_result.pose_landmarks:
            detection.pose_landmarks = [
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                for lm in pose_result.pose_landmarks[0]
            ]
        if pose_result.pose_world_landmarks:
            detection.pose_world_landmarks = [
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                for lm in pose_result.pose_world_landmarks[0]
            ]

        hand_result = self._hands.detect_for_video(mp_image, timestamp_ms)
        if hand_result.hand_landmarks:
            for i, handedness_list in enumerate(hand_result.handedness):
                label = handedness_list[0].category_name
                landmarks = [
                    [lm.x, lm.y, lm.z]
                    for lm in hand_result.hand_landmarks[i]
                ]
                # MediaPipe의 handedness는 미러링 기준 — 좌우 반전
                if label == "Right":
                    detection.left_hand_landmarks = landmarks
                else:
                    detection.right_hand_landmarks = landmarks

        face_result = self._face.detect_for_video(mp_image, timestamp_ms)
        if face_result.face_landmarks:
            detection.face_landmarks = [
                [lm.x, lm.y, lm.z]
                for lm in face_result.face_landmarks[0]
            ]

        return detection

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
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]
            x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
            x2, y2 = int(pt2[0] * w), int(pt2[1] * h)

            if has_visibility:
                vis = min(pt1[3], pt2[3])
                if vis < 0.5:
                    continue

            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (x, y), 3, color, -1)

    def close(self):
        self._pose.close()
        self._hands.close()
        self._face.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
