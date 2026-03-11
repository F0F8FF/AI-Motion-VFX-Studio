<p align="center">
  <h1 align="center">AI Motion VFX Studio</h1>
  <p align="center">
    Real-time AI motion tracking pipeline for interactive VFX with TouchDesigner
  </p>
  <p align="center">
    <a href="https://github.com/F0F8FF/AI-Motion-VFX-Studio/actions"><img src="https://github.com/F0F8FF/AI-Motion-VFX-Studio/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch">
    <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-00A98F" alt="MediaPipe">
    <img src="https://img.shields.io/badge/TouchDesigner-OSC-9B59B6" alt="TouchDesigner">
    <img src="https://img.shields.io/badge/Docker-ready-2496ED" alt="Docker">
    <img src="https://img.shields.io/badge/tests-36%20passed-brightgreen" alt="Tests">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</p>

---

웹캠 영상에서 **포즈·손·얼굴을 AI로 실시간 추적**하고, **오디오 분석**과 결합하여 분석 결과를 **OSC 프로토콜**로 TouchDesigner에 전송, 인터랙티브 VFX를 생성하는 풀스택 파이프라인입니다.


## Architecture

```
┌──────────┐     ┌──────────────────────────────────┐     ┌──────────────────┐
│  Webcam  │────▶│      Python AI Pipeline          │────▶│  TouchDesigner   │
│          │     │                                    │     │                  │
└──────────┘     │  ┌────────────┐  ┌─────────────┐  │     │  ┌────────────┐  │
                 │  │ MediaPipe  │  │   PyTorch    │  │     │  │ OSC In     │  │
┌──────────┐     │  │ Tasks API  │  │   Gesture    │  │ OSC │  │ CHOP/DAT   │  │
│   Mic    │────▶│  │ (parallel) │  │   Classifier │  │────▶│  │            │  │
│          │     │  │            │  │              │  │     │  ├────────────┤  │
└──────────┘     │  │ - Pose(33) │  ├─────────────┤  │     │  │ Particle   │  │
                 │  │ - Hand(21) │  │  Emotion     │  │     │  │ GLSL       │  │
                 │  │ - Face(478)│  │  Detector    │  │     │  │ Instancing │  │
                 │  └────────────┘  ├─────────────┤  │     │  └────────────┘  │
                 │                  │  Audio       │  │     │                  │
                 │                  │  Analyzer    │  │     └──────────────────┘
                 │                  │  (beat/BPM)  │  │
                 │                  └─────────────┘  │
                 │                                    │
                 │  ┌────────────────────────────┐   │     ┌──────────────────┐
                 │  │ FastAPI Dashboard + MJPEG  │   │     │  Web Dashboard   │
                 │  └──────────────┬─────────────┘   │     │  + Live Camera   │
                 │            :8000 │                  │     │  + Audio Viz     │
                 └─────────────────┼──────────────────┘     │  + Event Log     │
                                   └───────────────────────▶└──────────────────┘
```

## Features

| 모듈 | 기술 | 설명 |
|------|------|------|
| **Pose Estimation** | MediaPipe PoseLandmarker | 33개 전신 랜드마크 + 월드 좌표 |
| **Hand Tracking** | MediaPipe HandLandmarker | 양손 각 21개 랜드마크 |
| **Gesture Classification** | PyTorch Custom MLP | 8개 제스처 + 규칙 기반 폴백 + 커스텀 학습 |
| **Emotion Detection** | Geometric Feature Analysis | 7개 감정 + arousal/valence 연속값 |
| **Audio Reactive** | numpy FFT + Beat Detection | 16밴드 스펙트럼, RMS, 비트 감지, BPM 추정 |
| **Parallel Inference** | ThreadPoolExecutor | 3개 모델 병렬 추론으로 ~2-3x FPS 향상 |
| **OSC Streaming** | python-osc (UDP) | 실시간 저지연 TouchDesigner 연동 |
| **Web Dashboard** | FastAPI + WebSocket + MJPEG | 실시간 모니터링 + 카메라 스트리밍 |
| **Session Record/Play** | JSONL 기반 | 세션 녹화 후 카메라 없이 재생 가능 |
| **Custom Training** | 데이터 수집 → 학습 파이프라인 | CLI로 제스처 데이터 수집 및 모델 학습 |
| **Docker** | Dockerfile + Compose | 원클릭 컨테이너 배포 |

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/F0F8FF/AI-Motion-VFX-Studio.git
cd AI-Motion-VFX-Studio

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

# 풀 기능 실행 (대시보드 + 오디오 분석)
python main.py --dashboard --audio

# OSC 없이 AI만 테스트
python main.py --no-osc

# 세션 녹화
python main.py --record my_session

# 세션 재생 (카메라 불필요)
python main.py --play my_session --dashboard

# OSC 포트 변경
python main.py --osc-port 7000
```

### 4. Docker

```bash
docker compose up --build
# → http://localhost:8000 에서 대시보드 확인
```

### Keyboard Shortcuts

| 키 | 기능 |
|----|------|
| `q` | 종료 |
| `d` | 디버그 오버레이 토글 |
| `s` | 데이터 수집 모드에서 샘플 저장 |

## Web Dashboard

`--dashboard` 옵션으로 실행하면 **http://localhost:8000** 에서 실시간 모니터링 가능:

- **Live Camera** — MJPEG 스트리밍으로 랜드마크 오버레이 영상 확인
- **Gesture** — 양손 제스처 이름 + 신뢰도
- **Emotion** — 감정 이름 + arousal/valence 미터
- **Audio** — 16밴드 스펙트럼 바 + BPM + 레벨 미터
- **Performance** — 실시간 FPS
- **Event Log** — 제스처/감정 변화 타임라인

## Audio Reactive

`--audio` 옵션으로 마이크 입력을 실시간 분석합니다.

| OSC 주소 | 타입 | 설명 |
|----------|------|------|
| `/audio/level` | float (0~1) | RMS 볼륨 |
| `/audio/beat` | float (0/1) | 비트 감지 |
| `/audio/spectrum` | float[16] | 16밴드 스펙트럼 |

TouchDesigner에서 `arousal + beat` 조합으로 음악에 반응하는 VFX를 만들 수 있습니다.

## Session Recording

카메라 없이 테스트하거나 데모를 만들 때 유용합니다.

```bash
# 녹화
python main.py --record demo1
# → data/sessions/demo1.jsonl 생성

# 재생 (카메라 불필요, OSC로 TouchDesigner에 전송)
python main.py --play demo1
```

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
| `/audio/level` | float | 마이크 볼륨 (0~1) |
| `/audio/beat` | float | 비트 감지 (0/1) |
| `/audio/spectrum` | float[16] | 16밴드 스펙트럼 |

### VFX Mapping Ideas

```
arousal  ──▶ 파티클 개수, 셰이더 강도, 노이즈 스케일
valence  ──▶ HSV Hue, 색온도, 블러 양
gesture  ──▶ 이펙트 전환 트리거 (fist→폭발, peace→네온)
hand pos ──▶ Instancing 위치, 트레일 소스
pose     ──▶ 프로젝션 매핑, 실루엣 이펙트
beat     ──▶ 파티클 버스트, 화면 플래시, 스케일 펄스
spectrum ──▶ 이퀄라이저 비주얼, 주파수별 색상 매핑
```

## Project Structure

```
ai-motion-vfx/
├── main.py                       # 메인 파이프라인 & CLI
├── config.py                     # 설정 데이터클래스
├── session.py                    # 세션 녹화/재생
├── pyproject.toml                # 프로젝트 메타데이터
├── requirements.txt
├── Dockerfile                    # Docker 이미지
├── docker-compose.yml            # Docker Compose
│
├── ai/                           # AI 모듈
│   ├── pose_detector.py          # MediaPipe Tasks (병렬 추론)
│   ├── gesture_classifier.py     # PyTorch 제스처 분류 + 학습
│   ├── emotion_detector.py       # 기하학적 감정 추정
│   └── audio_analyzer.py         # 실시간 오디오 분석
│
├── models/
│   └── gesture_model.py          # GestureNet MLP 아키텍처
│
├── osc/
│   └── sender.py                 # OSC → TouchDesigner
│
├── dashboard/
│   └── app.py                    # FastAPI + WebSocket + MJPEG
│
├── static/
│   └── index.html                # 실시간 모니터링 UI
│
├── scripts/
│   └── download_models.py        # 모델 다운로더
│
├── tests/                        # pytest (36 tests)
│   ├── test_config.py
│   ├── test_emotion_detector.py
│   ├── test_gesture_classifier.py
│   ├── test_gesture_model.py
│   ├── test_audio_analyzer.py
│   └── test_session.py
│
└── .github/workflows/ci.yml     # GitHub Actions CI
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| Computer Vision | MediaPipe Tasks API (Pose + Hand + Face) |
| Deep Learning | PyTorch (Custom MLP Classifier) |
| Audio Analysis | numpy FFT, Beat Detection, BPM Estimation |
| Communication | python-osc (OSC over UDP) |
| Web Backend | FastAPI + WebSocket + MJPEG Streaming |
| Frontend | Vanilla JS + CSS Grid |
| VFX Engine | TouchDesigner |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions (multi-OS, multi-Python) |
| Testing | pytest (36 tests) |

## Performance

| 항목 | 내용 |
|------|------|
| 추론 방식 | 3개 모델 `ThreadPoolExecutor` 병렬 실행 |
| FPS 향상 | 순차 대비 ~2-3x |
| 오디오 분석 | 별도 스레드, 메인 루프 비차단 |
| 대시보드 | 20 FPS WebSocket + MJPEG 스트리밍 |

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
