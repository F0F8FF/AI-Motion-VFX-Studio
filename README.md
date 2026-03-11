<p align="center">
  <h1 align="center">AI Motion VFX Studio</h1>
  <p align="center">
    Real-time AI motion tracking pipeline for interactive VFX with TouchDesigner
  </p>
  <p align="center">
    <a href="https://github.com/F0F8FF/ai-motion-vfx/actions"><img src="https://github.com/F0F8FF/ai-motion-vfx/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch">
    <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-00A98F" alt="MediaPipe">
    <img src="https://img.shields.io/badge/TouchDesigner-OSC-9B59B6" alt="TouchDesigner">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</p>

---

웹캠 영상에서 **포즈·손·얼굴을 AI로 실시간 추적**하고, 분석 결과를 **OSC 프로토콜**로 TouchDesigner에 전송하여 인터랙티브 VFX를 생성하는 파이프라인입니다.


## Architecture

```
┌──────────┐     ┌──────────────────────────────────┐     ┌──────────────────┐
│  Webcam  │────▶│      Python AI Pipeline          │────▶│  TouchDesigner   │
│          │     │                                    │     │                  │
└──────────┘     │  ┌────────────┐  ┌─────────────┐  │     │  ┌────────────┐  │
                 │  │ MediaPipe  │  │   PyTorch    │  │     │  │ OSC In     │  │
                 │  │ Tasks API  │  │   Gesture    │  │ OSC │  │ CHOP/DAT   │  │
                 │  │            │  │   Classifier │  │────▶│  │            │  │
                 │  │ - Pose(33) │  │              │  │     │  ├────────────┤  │
                 │  │ - Hand(21) │  ├─────────────┤  │     │  │ Particle   │  │
                 │  │ - Face(478)│  │  Emotion     │  │     │  │ GLSL       │  │
                 │  │            │  │  Detector    │  │     │  │ Instancing │  │
                 │  └────────────┘  └─────────────┘  │     │  └────────────┘  │
                 │                                    │     │                  │
                 │  ┌────────────────────────────┐   │     └──────────────────┘
                 │  │ FastAPI WebSocket Dashboard │   │
                 │  └──────────────┬─────────────┘   │     ┌──────────────────┐
                 │            :8000 │                  │     │  Web Dashboard   │
                 └─────────────────┼──────────────────┘     │  (Real-time UI)  │
                                   └───────────────────────▶└──────────────────┘
```

## Features

| 모듈 | 기술 | 설명 |
|------|------|------|
| **Pose Estimation** | MediaPipe PoseLandmarker | 33개 전신 랜드마크 + 월드 좌표 |
| **Hand Tracking** | MediaPipe HandLandmarker | 양손 각 21개 랜드마크 |
| **Gesture Classification** | PyTorch Custom MLP | 8개 제스처 (open_palm, fist, peace, point, thumbs_up, wave, grab, none) |
| **Emotion Detection** | Geometric Feature Analysis | 7개 감정 + arousal/valence 연속값 |
| **OSC Streaming** | python-osc (UDP) | 실시간 저지연 TouchDesigner 연동 |
| **Web Dashboard** | FastAPI + WebSocket | 20 FPS 실시간 모니터링 UI |
| **Custom Training** | 데이터 수집 → 학습 파이프라인 | CLI로 제스처 데이터 수집 및 모델 학습 |

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/F0F8FF/ai-motion-vfx.git
cd ai-motion-vfx

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Models

```bash
python scripts/download_models.py
```

MediaPipe 모델 파일 (~40MB)을 `data/models/`에 다운로드합니다.

### 3. Run

```bash
# 기본 실행 (웹캠 + AI + OSC)
python main.py

# OSC 없이 AI만 테스트
python main.py --no-osc

# 웹 대시보드 함께 실행
python main.py --dashboard
# → http://localhost:8000 에서 실시간 모니터링

# OSC 포트 변경
python main.py --osc-port 7000
```

### Keyboard Shortcuts

| 키 | 기능 |
|----|------|
| `q` | 종료 |
| `d` | 디버그 오버레이 토글 |
| `s` | 데이터 수집 모드에서 샘플 저장 |

## Custom Gesture Training

자신만의 제스처를 학습시킬 수 있습니다.

```bash
# Step 1: 데이터 수집 (제스처당 50~100샘플 권장)
python main.py --collect wave
python main.py --collect thumbs_up

# Step 2: 모델 학습
python main.py --train
# → data/gesture_model.pt 생성, 다음 실행 시 자동 로드
```

## TouchDesigner Integration

### Setup

1. **OSC In CHOP** 생성 → Network Port: `9000`
2. **Null CHOP** 연결 → 파이썬 표현식으로 값 접근

### OSC Message Reference

| 주소 | 타입 | 설명 |
|------|------|------|
| `/pose/landmarks` | float[132] | 전신 포즈 (33 x [x,y,z,vis]) |
| `/pose/world` | float[132] | 월드 좌표 포즈 |
| `/hand/left/landmarks` | float[63] | 왼손 (21 x [x,y,z]) |
| `/hand/right/landmarks` | float[63] | 오른손 (21 x [x,y,z]) |
| `/gesture/{hand}/name` | string | 제스처 이름 |
| `/gesture/{hand}/confidence` | float | 제스처 신뢰도 (0~1) |
| `/emotion/name` | string | 감정 이름 |
| `/emotion/confidence` | float | 감정 신뢰도 (0~1) |
| `/emotion/arousal` | float | 각성도 (0~1) → **VFX 강도** |
| `/emotion/valence` | float | 긍정도 (-1~1) → **VFX 색조** |
| `/face/landmarks` | float[54] | 얼굴 주요 18포인트 |

### VFX Mapping Ideas

```
arousal  ──▶ 파티클 개수, 셰이더 강도, 노이즈 스케일
valence  ──▶ HSV Hue, 색온도, 블러 양
gesture  ──▶ 이펙트 전환 트리거 (fist→폭발, peace→네온)
hand pos ──▶ Instancing 위치, 트레일 소스
pose     ──▶ 프로젝션 매핑, 실루엣 이펙트
```

## Project Structure

```
ai-motion-vfx/
├── main.py                       # 메인 파이프라인 & CLI
├── config.py                     # 설정 데이터클래스
├── pyproject.toml                # 프로젝트 메타데이터
├── requirements.txt
│
├── ai/                           # AI 모듈
│   ├── pose_detector.py          # MediaPipe Tasks 래퍼
│   ├── gesture_classifier.py     # PyTorch 제스처 분류 + 학습
│   └── emotion_detector.py       # 기하학적 감정 추정
│
├── models/
│   └── gesture_model.py          # GestureNet MLP 아키텍처
│
├── osc/
│   └── sender.py                 # OSC → TouchDesigner
│
├── dashboard/
│   └── app.py                    # FastAPI + WebSocket
│
├── static/
│   └── index.html                # 실시간 모니터링 UI
│
├── scripts/
│   └── download_models.py        # 모델 다운로더
│
├── tests/                        # pytest 테스트 (24 tests)
│   ├── test_config.py
│   ├── test_emotion_detector.py
│   ├── test_gesture_classifier.py
│   └── test_gesture_model.py
│
└── .github/workflows/ci.yml     # GitHub Actions CI
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| Computer Vision | MediaPipe Tasks API (Pose + Hand + Face) |
| Deep Learning | PyTorch (Custom MLP Classifier) |
| Communication | python-osc (OSC over UDP) |
| Web Backend | FastAPI + WebSocket |
| Frontend | Vanilla JS + CSS Grid |
| VFX Engine | TouchDesigner |
| CI/CD | GitHub Actions |
| Testing | pytest (24 tests) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Lint
ruff check .
ruff format .
```

## License

MIT
