#!/usr/bin/env python3
"""MediaPipe 모델 파일 다운로드 스크립트."""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = [
    {
        "name": "pose_landmarker_heavy.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "description": "Pose Landmarker (Heavy)",
    },
    {
        "name": "hand_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "description": "Hand Landmarker",
    },
    {
        "name": "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "description": "Face Landmarker",
    },
]

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"


def download_file(url: str, dest: Path, description: str) -> bool:
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {description} ({size_mb:.1f}MB, 이미 존재)")
        return True

    print(f"  [DOWN] {description} ...")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print()
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"         → {size_mb:.1f}MB 완료")
        return True
    except Exception as e:
        print(f"\n  [FAIL] {e}")
        return False


def _progress(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar = "█" * (percent // 3) + "░" * (33 - percent // 3)
        sys.stdout.write(f"\r         {bar} {percent}%")
        sys.stdout.flush()


def main():
    print("=" * 50)
    print("  AI Motion VFX Studio — Model Downloader")
    print("=" * 50)
    print()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    for model in MODELS:
        dest = MODEL_DIR / model["name"]
        if download_file(model["url"], dest, model["description"]):
            success += 1

    print()
    print(f"  {success}/{len(MODELS)} 모델 준비 완료")

    if success == len(MODELS):
        print("  → python main.py 로 실행하세요!")
    else:
        print("  [WARN] 일부 모델 다운로드 실패. 네트워크 확인 후 재시도하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
