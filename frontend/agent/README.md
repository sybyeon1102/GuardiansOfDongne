# frontend-agent 서브 프로젝트

이 디렉터리는 **카메라/RTSP 스트림을 받아 Backend와 통신하는 에이전트**를 제공하는
`frontend-agent` 서브 프로젝트이다.
FastAPI 기반의 작은 서버와, 실시간 스트림을 Backend로 보내는 클라이언트 스크립트를 함께 포함한다.

- 프로젝트 이름: `frontend-agent` (pyproject.toml 상의 이름)
- 파이썬 패키지 이름: `frontend_agent` (import 에 사용하는 이름)

모든 실행 예시는 **이 디렉터리(`frontend/agent/`) 기준**으로 작성한다.

---

## 0. 환경 구축 (uv)

이 서브 프로젝트는 `uv` 기반으로 Python 환경을 관리한다.

### 0-1. 기본 환경 구축

먼저 `frontend/agent/` 디렉터리로 이동한다.

    cd frontend/agent
    uv sync

주요 의존성:

- fastapi[standard] : FastAPI 서버 및 dev/run 실행기
- opencv-python     : 카메라/RTSP 프레임 캡처
- mediapipe         : 포즈/키포인트 추출
- numpy             : 수치 연산 및 feature 벡터 처리
- requests          : Backend HTTP API 호출

### 0-2. 리눅스(OpenCV) 시스템 의존성

리눅스 환경에서 OpenCV(`opencv-python`)를 사용할 때는
Python 패키지 외에도 GL/GLib 시스템 라이브러리가 필요할 수 있다.

Ubuntu / Debian 계열 예시:

    sudo apt update
    sudo apt install -y libgl1-mesa-glx libglib2.0-0

다른 배포판(Fedora, Rocky, Arch 등)에서는 각 배포판의 패키지 매니저(dnf, pacman 등)를 사용해
OpenCV 실행에 필요한 GL/GLib 관련 패키지를 설치해야 한다.

---

## 1. 구조 및 주요 모듈

`frontend/agent/` 디렉터리 구조는 대략 다음과 같다.

    frontend/agent/
      pyproject.toml
      README.md
      frontend_agent/
        __init__.py
        main.py
        server_stream.py
        client_behavior_realtime.py

각 파일의 역할은 다음과 같다.

- `frontend_agent/__init__.py`
  `frontend_agent` 패키지의 루트 모듈이다. (필요 시 패키지 수준 설명/심볼 노출에 사용한다.)

- `frontend_agent/server_stream.py`
  FastAPI 앱과 스트리밍 관련 엔드포인트를 정의한다.
  - 카메라/RTSP 입력 초기화
  - MediaPipe/OpenCV 기반 포즈/feature 추출
  - Backend `/behavior/analyze`, `/event/payload` 로 HTTP 요청을 보내는 로직
  - (필요 시) `frontend/web/index.html` 을 서빙하는 엔드포인트

- `frontend_agent/main.py`
  FastAPI CLI가 참조하는 엔트리 포인트이다.
  내부에서 `server_stream.app` 을 가져와 `app` 이름으로 재노출하며,
  FastAPI dev/run 명령은 이 모듈의 `app` 객체를 사용해 서버를 실행한다.

- `frontend_agent/client_behavior_realtime.py`
  실시간 스트림 클라이언트 스크립트이다.
  - 로컬 카메라 혹은 RTSP 스트림에서 프레임을 읽는다.
  - 포즈/feature를 추출한 뒤 Backend `/behavior/analyze` 로 전송한다.
  - 필요에 따라 `/event/payload` 로도 이벤트/확률 정보를 전송한다.
  - `uv run run-realtime-client` 로 실행하는 독립형 CLI 도구이다.

---

## 2. 커맨드라인 엔트리 (uv run)

`pyproject.toml` 의 `[project.scripts]` 섹션은 다음과 같이 구성한다.

    [project.scripts]
    run-agent-dev = "fastapi dev frontend_agent/main.py --host 0.0.0.0 --port 8001"
    run-agent = "fastapi run frontend_agent/main.py --host 0.0.0.0 --port 8001"
    run-realtime-client = "frontend_agent.client_behavior_realtime:main"

실행할 때는 항상 `uv run <엔트리>` 형식을 사용한다.

### 2-1. 개발용 에이전트 서버

    uv run run-agent-dev

- 내부적으로 `fastapi dev frontend_agent/main.py --host 0.0.0.0 --port 8001` 를 실행한다.
- 코드 변경 시 자동 리로드가 일어나며, 개발/디버깅용으로 사용한다.
- 브라우저에서 `http://127.0.0.1:8001/` 로 접속해 웹 UI 또는 상태를 확인한다
  (루트 경로에서 `index.html` 을 서빙하도록 구현했다고 가정).

### 2-2. 실행용 에이전트 서버

    uv run run-agent

- 내부적으로 `fastapi run frontend_agent/main.py --host 0.0.0.0 --port 8001` 를 실행한다.
- 자동 리로드 없이 동작하며, 실제 테스트/간단 배포 환경에서 사용한다.

### 2-3. 실시간 스트림 클라이언트

    uv run run-realtime-client --help

- `frontend_agent.client_behavior_realtime:main` 을 실행한다.
- 구체적인 옵션 목록은 `client_behavior_realtime.py` 의 `argparse` 정의를 따른다.
- 예시(개념):

    uv run run-realtime-client \
        --backend-url http://127.0.0.1:8000 \
        --camera 0

---

## 3. Backend 연동

frontend-agent 는 Backend의 HTTP API를 통해 기능을 수행한다.

주요 연동 대상:

- `/behavior/analyze`
  feature 시퀀스를 전송해 정상/이상 결과 및 이벤트 라벨/점수를 받는다.
- `/event/payload`
  실시간 확률/이벤트 정보를 JSONL 로그로 기록하도록 전송한다.

Backend 베이스 URL은 환경 변수나 인자로 지정할 수 있다.

예시:

- 환경 변수: `BACKEND_BASE_URL=http://127.0.0.1:8000`
- 혹은 `--backend-url` 옵션 등 (구현에 따라 다름)

일반적인 개발 환경에서는:

1. Backend 서버를 8000 포트에 띄운다.
2. frontend-agent 서버를 8001 포트에 띄운다.
3. 실시간 클라이언트에서 Backend 주소를 `http://127.0.0.1:8000` 으로 맞춘다.

---

## 4. frontend/web 과의 연동

`frontend/web/index.html` 을 frontend-agent 가 직접 서빙하는 경우:

- `frontend_agent/server_stream.py` 의 루트 엔드포인트(`/` 등)에서
  `frontend/web/index.html` 파일을 읽어 반환하도록 구현할 수 있다.
- 이 경우 사용자는 브라우저에서 다음과 같이 접근한다.

    Backend: http://127.0.0.1:8000
    Agent:   http://127.0.0.1:8001
    Browser: http://127.0.0.1:8001/

정적 파일 서빙 방식은 실제 `server_stream.py` 구현에 따르며,
README에서는 “에이전트가 웹 UI를 서빙할 수 있다”는 역할 수준만 정리한다.

---

## 5. 전형적인 개발 워크플로 요약

1. Backend 환경 구축 및 실행

       cd backend
       uv sync --extra cpu
       uv run run-backend-dev

2. frontend-agent 환경 구축 및 실행

       cd frontend/agent
       uv sync
       uv run run-agent-dev

3. 실시간 클라이언트 실행

       uv run run-realtime-client \
           --backend-url http://127.0.0.1:8000 \
           --camera 0

4. 웹 UI 확인

    브라우저에서 `http://127.0.0.1:8001/` 로 접속해
    스트림 상태와 분석 결과를 확인한다.
