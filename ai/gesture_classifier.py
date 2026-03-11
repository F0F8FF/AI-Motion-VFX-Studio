"""
제스처 분류 모듈.
손 랜드마크를 정규화 후 PyTorch 모델로 분류한다.
학습 데이터 수집 모드도 지원.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.gesture_model import GestureNet


@dataclass
class GestureResult:
    gesture: str
    confidence: float
    all_scores: dict[str, float]


class GestureClassifier:
    """손 랜드마크 기반 제스처 분류기. 학습/추론/데이터 수집 통합."""

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = GestureNet().to(self.device)
        self.classes = GestureNet.GESTURE_CLASSES

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "classes" in checkpoint:
                self.classes = checkpoint["classes"]
            print(f"[GestureClassifier] 모델 로드: {model_path}")
        else:
            print("[GestureClassifier] 사전학습 모델 없음 → 규칙 기반 폴백 사용")

        self.model.eval()
        self._collecting = False
        self._collected_data: list[dict] = []

    @staticmethod
    def normalize_landmarks(landmarks: list[list[float]]) -> np.ndarray:
        """손목(index 0) 기준 상대좌표로 정규화. 스케일 불변성 확보."""
        arr = np.array(landmarks, dtype=np.float32)
        wrist = arr[0]
        relative = arr - wrist
        max_dist = np.max(np.abs(relative)) + 1e-6
        normalized = relative / max_dist
        return normalized.flatten()

    def classify(self, hand_landmarks: list[list[float]]) -> GestureResult:
        if len(hand_landmarks) != 21:
            return GestureResult("none", 0.0, {})

        features = self.normalize_landmarks(hand_landmarks)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        scores = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        best_idx = int(probs.argmax())
        best_class = self.classes[best_idx]
        best_conf = float(probs[best_idx])

        if best_conf < 0.4:
            rule_result = self._rule_based_classify(hand_landmarks)
            if rule_result.confidence > best_conf:
                return rule_result

        return GestureResult(best_class, best_conf, scores)

    def _rule_based_classify(self, landmarks: list[list[float]]) -> GestureResult:
        """모델 신뢰도가 낮을 때 사용하는 규칙 기반 폴백."""
        arr = np.array(landmarks)
        wrist = arr[0]

        finger_tips = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 소지
        finger_pips = [3, 6, 10, 14, 18]

        fingers_up = []
        for tip, pip_ in zip(finger_tips[1:], finger_pips[1:]):
            fingers_up.append(arr[tip][1] < arr[pip_][1])

        thumb_tip = arr[4]
        thumb_ip = arr[3]
        thumb_up = abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip[0] - wrist[0])

        extended = [thumb_up] + fingers_up
        num_extended = sum(extended)

        if num_extended == 5:
            return GestureResult("open_palm", 0.85, {})
        elif num_extended == 0:
            return GestureResult("fist", 0.85, {})
        elif extended[1] and extended[2] and num_extended == 2:
            return GestureResult("peace", 0.80, {})
        elif extended[1] and num_extended == 1:
            return GestureResult("point", 0.80, {})
        elif extended[0] and num_extended == 1:
            return GestureResult("thumbs_up", 0.75, {})

        return GestureResult("none", 0.3, {})

    # --- 데이터 수집 (학습용) ---

    def start_collecting(self):
        self._collecting = True
        self._collected_data = []
        print("[GestureClassifier] 데이터 수집 시작")

    def collect_sample(self, hand_landmarks: list[list[float]], label: str):
        if not self._collecting:
            return
        features = self.normalize_landmarks(hand_landmarks).tolist()
        self._collected_data.append({"features": features, "label": label})

    def save_collected_data(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        existing.extend(self._collected_data)
        with open(path, "w") as f:
            json.dump(existing, f)
        print(f"[GestureClassifier] {len(self._collected_data)}개 샘플 저장 → {path}")
        self._collecting = False

    def train(self, data_path: str, epochs: int = 100, lr: float = 0.001, save_path: str = "gesture_model.pt"):
        """수집된 데이터로 모델 학습."""
        with open(data_path) as f:
            data = json.load(f)

        label_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        features, labels = [], []
        for sample in data:
            if sample["label"] in label_to_idx:
                features.append(sample["features"])
                labels.append(label_to_idx[sample["label"]])

        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (out.argmax(1) == batch_y).sum().item()
                total += len(batch_y)

            if (epoch + 1) % 10 == 0:
                acc = correct / total * 100
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {acc:.1f}%")

        torch.save(
            {"model_state_dict": self.model.state_dict(), "classes": self.classes},
            save_path,
        )
        print(f"[GestureClassifier] 모델 저장 → {save_path}")
        self.model.eval()
