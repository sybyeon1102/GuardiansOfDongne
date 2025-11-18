# 행동 이상 탐지 프로젝트 (GoD)

이 레포는 **포즈 기반 정상/이상 행동 탐지**를 위한 전체 파이프라인을 담는 monorepo이다.
정상(normal)/이상(anomaly) 행동 데이터를 전처리하고 LSTM 모델을 학습한 뒤,
FastAPI 백엔드를 통해 실시간 스트림/웹 UI와 연동하는 구조로 설계한다.

상위 구조는 다음 네 축으로 본다.

- `core/`      : 언어 공용 코어 유틸 계층 (`project-core` Python 패키지)
- `modeling/`  : 데이터 전처리 + LSTM 모델 학습·추론
- `backend/`   : FastAPI 기반 HTTP 백엔드 서버
- `frontend/`  : 웹 UI(`web/`) + 스트리밍 에이전트(`agent/`)

모든 Python 서브 프로젝트는 **각각의 `pyproject.toml + uv`** 로 독립 관리한다.

---

## 0. 공통 환경

### 0-1. Python / uv

- Python 버전: **3.12 이상**
- Python 패키지/환경 관리 도구: **[uv](https://docs.astral.sh/uv/)**

각 서브 프로젝트에서 공통으로 사용하는 기본 패턴은 다음과 같다.

1. 서브 프로젝트 디렉터리로 이동

       cd <서브 프로젝트 경로>   # 예: cd modeling

2. 의존성 설치 (최초 1회 또는 의존성 변경 시)

       uv sync

3. 실행

       uv run <엔트리 이름>

각 서브 프로젝트의 구체적인 실행 엔트리와 옵션은
해당 디렉터리의 README를 참고한다.

### 0-2. PyTorch / CUDA (modeling, backend)

`modeling/`, `backend/` 서브 프로젝트는 PyTorch를 optional extra로 설치한다.
CPU 전용 또는 CUDA 버전에 맞는 extra를 선택해야 한다.

예시는 각 서브 프로젝트의 README를 따른다.

- CPU 예시:

      uv sync --extra cpu

- CUDA 예시(발췌):

      uv sync --extra cu130
      uv sync --extra cu129
      ...

(주의: 한 번에 하나의 extra만 선택한다.)

### 0-3. 리눅스(OpenCV) 시스템 의존성

리눅스 환경에서 OpenCV(`opencv-python`)를 사용할 때는 Python 패키지 외에도
GL/GLib 시스템 라이브러리가 필요할 수 있다.

Ubuntu / Debian 계열 예시:

    sudo apt update
    sudo apt install -y libgl1-mesa-glx libglib2.0-0

다른 배포판(Fedora, Rocky, Arch 등)을 사용할 경우에는
각 배포판의 패키지 매니저(dnf, pacman 등)로 OpenCV 실행에 필요한
GL/GLib 관련 패키지를 설치해야 한다.

---

## 1. 레포 구조

최상위 디렉터리 구조는 대략 다음과 같다.

    core/
      README.md
      python/
        pyproject.toml
        src/project_core/...
    modeling/
      pyproject.toml
      README.md
      src/modeling/...
    backend/
      pyproject.toml
      README.md
      app/...
    frontend/
      README.md
      web/
        index.html
        (기타 정적 자원)
      agent/
        pyproject.toml
        README.md
        frontend_agent/...
    legacy/
      ...

### 1-1. core/

- `core/python` 에서 `project-core` 서브 프로젝트를 제공한다.
- Python 패키지 이름은 `project_core` 이며,
  `Result[T, E]`, `Ok[T]`, `Err[E]` 등 **공용 에러/결과 표현 계층**을 제공한다.
- 다른 서브 프로젝트는 `from project_core import Result, Ok, Err` 형태로 사용한다.
- 자세한 내용: `core/README.md`, `core/python/README.md`

### 1-2. modeling/

- 정상(normal) / 이상(anomaly) 행동 데이터 전처리 + LSTM 모델 학습/추론을 담당한다.
- 주요 역할:
  - CVAT XML + 동영상으로부터 feature 시퀀스 생성
  - `X.npy`, `Y.npy`, `meta.json` 형태의 데이터셋 생성
  - LSTM 멀티라벨 분류 모델 학습 및 체크포인트 저장
  - 단일 시퀀스에 대한 LSTM 추론
- 주요 엔트리(발췌):
  - `build-dataset-normal`
  - `build-dataset-anomaly`
  - `train-lstm-model`
  - `run-lstm-inference`
- 자세한 내용: `modeling/README.md`

### 1-3. backend/

- **FastAPI 기반 HTTP 백엔드 서버**를 제공한다.
- 주요 역할:
  - `/behavior/analyze` : LSTM 기반 정상/이상 행동 분석 API
  - `/event/payload`, `/event/recent` : 이벤트 로그 기록 및 조회
  - Kakao “나에게 보내기” 알림 연동 (END 이벤트 등 조건에 따라 발송)
  - `/health`, `/echo` : 헬스체크/디버그 엔드포인트
- `modeling` 서브 프로젝트의 추론 코드를 호출해 결과를 HTTP 응답으로 변환한다.
- 주요 엔트리:
  - `run-backend-dev` : 개발용 FastAPI 서버 (자동 리로드)
  - `run-backend`     : 실제 구동용 FastAPI 서버
- 자세한 내용: `backend/README.md`

### 1-4. frontend/

- 사용자와 직접 맞닿는 프론트엔드 계층이다.
- 두 축으로 나뉜다.

1. `frontend/web/`
   - 현재는 **단일 `index.html` + JavaScript** 로 된 정적 웹 UI이다.
   - 장기적으로는 TypeScript 기반 프레임워크(React/Svelte 등)로 확장 가능하다.
   - 정적 파일은 `frontend-agent` 서버 또는 별도 정적 서버에서 서빙할 수 있다.

2. `frontend/agent/`
   - Python + FastAPI 기반 스트리밍 에이전트 서브 프로젝트이다.
   - 역할:
     - 카메라/RTSP 스트림에서 프레임 캡처
     - MediaPipe/OpenCV 로 포즈/feature 추출
     - Backend `/behavior/analyze` 호출 → 정상/이상 판단
     - `/event/payload` 로 이벤트/확률 정보 전송
     - (필요 시) `frontend/web/index.html` 서빙
   - 주요 엔트리:
     - `run-agent-dev`      : 개발용 에이전트 서버
     - `run-agent`          : 실행용 에이전트 서버
     - `run-realtime-client` : 실시간 스트림 클라이언트 스크립트
   - 자세한 내용: `frontend/README.md`, `frontend/agent/README.md`

---

## 3. 의존 계층 및 설계 원칙

이 레포는 다음과 같은 **의존 계층(import 방향)**을 가진다.

    project_core  →  modeling  →  backend  →  frontend

규칙(발췌):

- `project_core` 는 어떤 서브 프로젝트도 import 하지 않는다.
- `modeling` 은 `project_core`를 import 할 수 있지만 `backend`, `frontend`를 import 하지 않는다.
- `backend` 는 `project_core`, `modeling`을 import 할 수 있지만 `frontend`를 import 하지 않는다.
- `frontend` 는 원칙적으로 Python import로 `backend`/`modeling` 에 직접 의존하지 않고,
  HTTP API를 통해 backend와 통신한다. (필요 시 `project_core` 정도만 import)

에러 표현/도메인 경계에서는 **Result 스타일**을 사용한다.

- `Result[T, E] = Ok[T] | Err[E]`
- FastAPI 라우터/서비스에서 `match result:` 구문으로 해석하는 방식을 기본으로 한다.
- 예외는 불변식 위반/치명적 오류에만 사용한다.

자세한 설계/스타일 관련 규칙은 별도 문서로 정리한 **불변식(INV-xxx)** 를 따른다.

---

## 4. 전형적인 end-to-end 실행 흐름

아래는 로컬 개발 환경에서 **정상/이상 행동 분석 파이프라인**을 한 번에 훑는 예시이다.

1. **모델링: 데이터셋 생성 + LSTM 학습**

   1) 환경 구축

          cd modeling
          uv sync --extra cpu     # 또는 CUDA extra 선택

   2) 정상/이상 데이터셋 생성 (경로/옵션은 실제 환경에 맞게 조정)

          uv run build-dataset-normal  --video-root ... --xml-root ... --out-dir ...
          uv run build-dataset-anomaly --video-root ... --xml-root ... --out-dir ...

   3) LSTM 모델 학습

          uv run train-lstm-model --data-dir /path/to/dataset_dir --device auto

      학습이 끝나면 LSTM 체크포인트(.pt)를 얻는다.

2. **백엔드 서버 실행**

   1) 환경 구축

          cd backend
          uv sync --extra cpu

   2) `.env` 또는 환경 변수 설정

      - `MODEL_CHECKPOINT_PATH=/path/to/checkpoint.pt`
      - Kakao 알림에 필요한 값들 (`KAKAO_ACCESS_TOKEN`, `ALERT_LINK_URL`, ...)

   3) 개발용 서버 실행

          uv run run-backend-dev

      - `http://127.0.0.1:8000/health`
      - `http://127.0.0.1:8000/docs` 로 상태/스펙 확인

3. **frontend-agent 실행**

   1) 환경 구축

          cd frontend/agent
          uv sync

   2) 에이전트 서버 실행

          uv run run-agent-dev    # 포트 예: 8001

   3) (선택) 실시간 클라이언트 실행

          uv run run-realtime-client \
              --backend-url http://127.0.0.1:8000 \
              --camera 0

4. **웹 UI 확인**

   - 브라우저에서 `http://127.0.0.1:8001/` 로 접속해
     스트림 상태, 분석 결과, 이벤트 등을 확인한다.

---

## 5. 개발 시 참고 사항

- 새 Python 코드는 PEP 8 스타일과 현대적 타입 힌트를 따르도록 한다.
- 제너릭/타입 별칭은 Python 3.12의 PEP 695 문법(`class X[T]`, `type Alias[T] = ...`)을 기본으로 사용한다.
- `uv run python`, `uv run python -m ...` 대신
  항상 **`[project.scripts]`에 정의된 엔트리 이름**을 사용하는 것을 목표로 한다.
- 패키지 `__init__.py`에서는 re-export를 최소화하고,
  필요한 경우에만 얇은 퍼블릭 페이사드를 제공한다.

각 서브 프로젝트의 세부적인 규칙과 사용 예시는
`core/python/README.md`, `modeling/README.md`, `backend/README.md`,
`frontend/README.md`, `frontend/agent/README.md` 를 참고한다.
