# Modeling 서브 프로젝트

이 디렉터리는 **정상/이상 행동 데이터 전처리 + LSTM 모델 학습·추론**을 담당하는
`modeling` 서브 프로젝트이다.
모든 실행 예시는 **이 디렉터리(`modeling/`) 기준**으로 작성한다.

---

## 0. 환경 구축 (uv + optional extras)

이 서브 프로젝트는 `uv` 기반으로 Python 환경을 관리한다.
PyTorch는 **optional extra**를 통해 CPU / CUDA 버전을 선택해서 설치한다.

### 0-1. 기본 환경 구축

먼저 `modeling/` 디렉터리로 이동한다.

    cd modeling

이후 자신의 환경에 맞는 extra 를 선택해서 `uv sync`를 실행한다.

    # CPU 전용
    uv sync --extra cpu

    # CUDA 13.0 (예: 리눅스/윈도우에서 CUDA 13.0 환경)
    uv sync --extra cu130

    # CUDA 12.9 / 12.8 / 12.6 / 12.4 / 12.1
    uv sync --extra cu129
    uv sync --extra cu128
    uv sync --extra cu126
    uv sync --extra cu124
    uv sync --extra cu121

주의사항:

- **한 번에 하나의 extra만** 선택해야 한다.
  (pyproject의 `[tool.uv.conflicts]`에서 서로 동시에 쓸 수 없도록 막아 둔다.)
- CUDA 버전을 잘 모르면:
  - GPU를 쓰지 않을 계획이면 `cpu` 로 시작한다.
  - GPU를 쓰고 싶다면, 로컬 CUDA 버전에 맞는 extra를 고른다.
    맞는 것이 애매하면, 같은 major/minor 라인의 **가장 높은 버전**을 우선 시도한다.

### 0-2. 리눅스(OpenCV) 시스템 의존성

리눅스 환경에서 OpenCV(`opencv-python`)를 사용할 때는
Python 패키지 외에 **시스템 라이브러리(GL/GLib 계열)**가 필요할 수 있다.

예를 들어 **Ubuntu / Debian 계열**에서는 다음 패키지를 먼저 설치한다.

    sudo apt update
    sudo apt install -y libgl1-mesa-glx libglib2.0-0

다른 배포판(예: Fedora, Rocky, Arch 등)을 사용할 경우에는
해당 배포판의 패키지 매니저(dnf, pacman 등)로
OpenCV가 요구하는 GL/GLib 런타임 패키지를 설치해야 한다.
(일반적으로 `mesa` / `glib2` 계열 패키지에 포함된다.)

---

## 1. 구조 및 주요 모듈

`modeling/` 디렉터리의 구조는 대략 다음과 같다.

    modeling/
      pyproject.toml
      README.md
      src/modeling/
        __init__.py
        py.typed
        preprocessing/
          __init__.py
          dataset_common.py
          dataset_normal.py
          dataset_anomaly.py
        training/
          __init__.py
          trainer_lstm.py
        inference/
          __init__.py
          inference_lstm.py
      checkpoints/   # (선택) 학습된 모델 체크포인트 저장 디렉터리

각 서브 패키지의 역할은 다음과 같다.

- `preprocessing/`
  - 정상/이상 행동 데이터셋(np.ndarray) 생성
  - `dataset_common.py` : 공통 전처리 파이프라인
  - `dataset_normal.py` : 정상(구매/일상) 행동용 데이터셋 CLI
  - `dataset_anomaly.py` : 이상 행동용 데이터셋 CLI
- `training/`
  - LSTM 멀티라벨 분류 모델 학습
  - `trainer_lstm.py` : 학습 루틴 + CLI
- `inference/`
  - 학습된 LSTM 모델을 사용한 추론
  - `inference_lstm.py` : 단일 시퀀스 추론 CLI

---

## 2. 커맨드라인 엔트리(uv run)

`pyproject.toml`에는 다음과 같은 엔트리가 등록되어 있다.

- `build-dataset-normal`
  - 엔트리: `modeling.preprocessing.dataset_normal:main`
  - 역할: 정상 행동(normal, purchase/regular)용 LSTM 데이터셋 생성
- `build-dataset-anomaly`
  - 엔트리: `modeling.preprocessing.dataset_anomaly:main`
  - 역할: 이상 행동(anomaly)용 LSTM 데이터셋 생성
- `train-lstm-model`
  - 엔트리: `modeling.training.trainer_lstm:main`
  - 역할: LSTM 멀티라벨 분류 모델 학습
- `run-lstm-inference`
  - 엔트리: `modeling.inference.inference_lstm:main`
  - 역할: 학습된 LSTM 체크포인트를 사용해 단일 시퀀스에 대해 추론 실행

실행할 때는 항상 `uv run <엔트리>` 형식을 사용한다.

예:

    uv run build-dataset-normal --help
    uv run build-dataset-anomaly --help
    uv run train-lstm-model --help
    uv run run-lstm-inference --help

---

## 3. 데이터셋 생성 (정상/이상 행동)

### 3-1. 정상 행동 데이터셋 생성

엔트리: `build-dataset-normal`
모듈: `modeling.preprocessing.dataset_normal`

이 스크립트는 **정상(구매/일상) 행동 데이터셋**을 생성한다.
`dataset_common.py`에 정의된 공통 전처리 파이프라인을 사용하며,
정상 행동에 해당하는 구간을 중심으로 윈도우를 만든다.

기본 사용 예시는 다음과 같다.

    uv run build-dataset-normal \
        --video-root /path/to/videos \
        --xml-root /path/to/cvat_xml \
        --out-dir /path/to/output_dir

주요 옵션(요약):

- `--video-root` : 입력 영상 루트 디렉터리
  root directory containing input videos
- `--xml-root` : CVAT XML 루트 디렉터리
  root directory containing CVAT XML annotation files
- `--out-dir` : 출력 디렉터리
  output directory for `X.npy`, `Y.npy`, `meta.json`
- `--window-size`, `--stride`, `--overlap` : 시퀀스 윈도우 설정
- `--resize-w` : OpenCV 리사이즈 너비
- `--model-complexity` : MediaPipe Pose model complexity (0/1/2)

출력 파일:

- `X.npy` : (N, T, F) 형태의 윈도우 시퀀스 배열
- `Y.npy` : (N, C) 형태의 멀티라벨 원-핫 벡터
- `meta.json` : 이벤트 라벨 이름, 전처리 설정 등 메타데이터

### 3-2. 이상 행동 데이터셋 생성

엔트리: `build-dataset-anomaly`
모듈: `modeling.preprocessing.dataset_anomaly`

CVAT XML + 영상(.mp4)을 이용해 **이상 행동(anomaly)용 LSTM 데이터셋**
`X.npy`, `Y.npy`, `meta.json`을 생성한다.
전체 파이프라인은 정상 데이터셋과 동일하지만,
**이상 이벤트 구간을 중심으로 윈도우를 구성한다**는 점이 다르다.

기본 사용 예시는 다음과 같다.

    uv run build-dataset-anomaly \
        --video-root /path/to/videos \
        --xml-root /path/to/cvat_xml \
        --out-dir /path/to/output_dir

나머지 옵션/출력 파일 구조는 `build-dataset-normal`과 동일하다.

---

## 4. LSTM 모델 학습

엔트리: `train-lstm-model`
모듈: `modeling.training.trainer_lstm`

전처리된 윈도우 데이터셋(`X.npy`, `Y.npy`, `meta.json`)을 사용해
LSTM 기반 멀티라벨 분류 모델을 학습한다.

기본 사용 예시:

    uv run train-lstm-model \
        --data-dir /path/to/dataset_dir \
        --epochs 40 \
        --batch 64 \
        --device auto

주요 옵션(요약):

- `--data-dir` : `X.npy`, `Y.npy`, `meta.json`이 있는 디렉터리
- `--epochs` : 학습 epoch 수 (기본값 40)
- `--batch` : 배치 크기
- `--lr` : 학습률
- `--val-ratio` : 검증 세트 비율
- `--device` : `auto` / `cpu` / `cuda` (PyTorch 디바이스)
- `--save` : 학습 완료 후 체크포인트를 저장할 `.pt` 파일 경로

출력(저장) 체크포인트에는 대략 다음 정보가 포함된다.

- `state_dict` : LSTM 모델 가중치
- `meta` : 전처리 단계에서 만든 `meta.json` 내용
- `thresholds` : 클래스별 decision threshold 리스트
- 그 외 학습 관련 메트릭

---

## 5. LSTM 추론 (단일 시퀀스)

엔트리: `run-lstm-inference`
모듈: `modeling.inference.inference_lstm`

학습된 LSTM 체크포인트와 입력 시퀀스(JSON 파일)을 사용해
단일 시퀀스에 대해 정상/이상 추론을 수행하고, 결과를 JSON으로 출력한다.

입력 JSON 형식은 다음 두 가지를 지원한다.

- 최상위가 `[[...], [...], ...]` : 프레임별 feature 벡터 리스트
- 또는 `{"frames": [[...], [...], ...]}` : `frames` 키 아래에 리스트

기본 사용 예시:

    uv run run-lstm-inference \
        --checkpoint /path/to/checkpoint.pt \
        --input-json /path/to/frames.json \
        --device auto \
        --pretty

주요 옵션:

- `-c`, `--checkpoint` : 체크포인트(.pt) 파일 경로
- `-i`, `--input-json` : 입력 프레임 시퀀스를 담은 JSON 파일 경로
- `-d`, `--device` : `auto` / `cpu` / `cuda`
- `--pretty` : 결과 JSON을 들여쓰기하여 보기 좋게 출력

표준 출력(stdout)으로 JSON 결과를 내보내므로,
파일로 저장하고 싶다면 셸에서 리다이렉트를 사용한다.

    uv run run-lstm-inference ... > result.json

---

## 6. 전형적인 워크플로 예시

1. **환경 구축**

       cd modeling
       uv sync --extra cpu      # 또는 필요한 CUDA extra 선택

2. **데이터셋 생성 (정상 → 이상 순서)**

       uv run build-dataset-normal  --video-root ... --xml-root ... --out-dir ...
       uv run build-dataset-anomaly --video-root ... --xml-root ... --out-dir ...

3. **LSTM 모델 학습**

       uv run train-lstm-model --data-dir /path/to/dataset_dir --device auto

4. **단일 시퀀스 추론 테스트**

       uv run run-lstm-inference \
           --checkpoint /path/to/checkpoint.pt \
           --input-json /path/to/frames.json \
           --device auto \
           --pretty

이 워크플로를 기반으로, backend / frontend 에서 사용하는
실시간 추론 파이프라인(예: `/behavior/analyze` API)은
`modeling.inference.inference_lstm`의 로직을 호출하도록 설계한다.
