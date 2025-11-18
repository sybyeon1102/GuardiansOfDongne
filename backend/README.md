# Backend 서브 프로젝트

이 디렉터리는 **FastAPI 기반 HTTP 백엔드 서버**를 제공하는 `backend` 서브 프로젝트이다.
행동 분석 API, 이벤트 로그 수신, Kakao 알림 전송 등 프로젝트의 핵심 HTTP 인터페이스를 담당한다.

모든 실행 예시는 **이 디렉터리(`backend/`) 기준**으로 작성한다.

---

## 0. 환경 구축 (uv + optional extras)

이 서브 프로젝트는 `uv` 기반으로 Python 환경을 관리한다.
PyTorch는 **optional extra**를 통해 CPU / CUDA 버전을 선택해서 설치한다.
(`modeling` 서브 프로젝트의 추론 코드를 호출하기 위해 torch 가 필요하다.)

### 0-1. 기본 환경 구축

먼저 `backend/` 디렉터리로 이동한다.

    cd backend

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
  (`pyproject.toml` 의 `[tool.uv.conflicts]`에서 서로 동시에 쓸 수 없도록 막아 둔다.)
- CUDA 버전을 잘 모르면:
  - GPU를 쓰지 않을 계획이면 `cpu` 로 시작한다.
  - GPU를 쓰고 싶다면, 로컬 CUDA 버전에 맞는 extra를 고른다.
    맞는 것이 애매하면, 같은 major/minor 라인의 **가장 높은 버전**을 우선 시도한다.

---

## 1. 구조 및 주요 모듈

`backend/` 디렉터리의 구조는 대략 다음과 같다.

    backend/
      pyproject.toml
      README.md
      app/
        __init__.py
        main.py
        config.py
        errors.py
        models/
          __init__.py
          model_io_behavior.py
          model_io_event.py
        services/
          __init__.py
          service_behavior.py
          service_event.py
        routers/
          __init__.py
          router_behavior.py
          router_event.py
        internal/
          __init__.py
          kakao_notification.py

각 모듈/패키지의 역할은 다음과 같다.

- `app/main.py`
  FastAPI 앱 생성 및 라우터 등록, `/health`, `/echo` 엔드포인트 제공.
- `app/config.py`
  `pydantic-settings` 기반 설정 로딩. `.env` 파일과 환경 변수를 읽어
  모델 체크포인트 경로, Kakao 토큰, 알림용 링크 등을 관리한다.
- `app/errors.py`
  백엔드 전용 `InferenceError`, `InferenceErrorCode` 정의 및
  도메인 에러를 `HTTPException` 으로 매핑하는 헬퍼를 제공한다.
- `app/models/`
  - `model_io_behavior.py` : `/behavior/analyze` 요청/응답용 Pydantic 모델
  - `model_io_event.py` : 이벤트 로그/최근 이벤트 조회용 Pydantic 모델
- `app/services/`
  - `service_behavior.py` : `modeling` 서브 프로젝트의 LSTM 추론을 호출하여
    행동 이상 여부를 판단하는 도메인 서비스.
  - `service_event.py` : 이벤트 JSONL 로그 기록, 최근 이벤트 메모리 보관,
    END 이벤트 시 Kakao 알림을 트리거하는 서비스.
- `app/routers/`
  - `router_behavior.py` : `/behavior/analyze` 라우터 정의.
  - `router_event.py` : `/event/...` 라우터 정의 및 Kakao 디버그 엔드포인트.
- `app/internal/kakao_notification.py`
  Kakao “나에게 보내기” 메모 API 연동,
  토큰 체크 및 실제 메모 전송 로직을 캡슐화한다.

---

## 2. 설정 / 환경 변수

Backend 는 `pydantic-settings` 를 사용해 `.env` 및 환경 변수를 로딩한다.
기본적으로 `backend/.env` 파일 또는 시스템 환경 변수에서 다음과 같은 값을 읽는다.

예시(요약):

- `MODEL_CHECKPOINT_PATH`
  - LSTM 모델 체크포인트(.pt) 파일 경로.
  - `service_behavior` 가 이 경로를 사용해 `modeling` 서브 프로젝트의
    `LstmInferenceContext` 를 초기화한다.
- `KAKAO_ACCESS_TOKEN`
  - Kakao API “나에게 보내기”에 사용할 액세스 토큰.
- `ALERT_LINK_URL`
  - Kakao 알림에 포함할 버튼 링크 URL.
- `ALERT_BUTTON_TITLE`
  - Kakao 알림 버튼에 표시할 텍스트.

자세한 예시는 `backend/.env.example` 를 참고한다.
운영/개발 환경에 따라 `.env` 내용을 복사/수정해서 사용한다.

---

## 3. FastAPI 서버 실행 (개발용/실행용)

`pyproject.toml` 에는 FastAPI 서버 실행을 위한 스크립트 엔트리가 정의된다.
실행 시에는 항상 **`uv run <엔트리>`** 형식을 사용한다.
(불변식 [INV-061]: `uv run python -m ...` 대신 엔트리 이름을 사용한다.)

### 3-1. 개발 서버 (자동 리로드)

개발 중에는 `fastapi dev` 서브커맨드를 사용한다.

    uv run run-backend-dev

위 명령은 내부적으로 다음을 실행한다.

- `fastapi dev app/main.py --host 0.0.0.0 --port 8000`

특징:

- 코드 변경 시 자동 리로드
- 로컬 개발/디버깅 용도에 적합
- PC에서 빠르게 API 스펙을 확인하거나, frontend/agent 와 연동 테스트를 할 때 사용한다.

### 3-2. 실행 서버 (실제 구동용)

좀 더 “실제에 가까운” 모드로 서버를 띄우고 싶다면 `fastapi run` 을 사용한다.

    uv run run-backend

이 엔트리는 내부적으로 다음을 실행한다.

- `fastapi run app/main.py --host 0.0.0.0 --port 8000`

특징:

- 코드 자동 리로드 없음
- fastapi[standard] 에 포함된 uvicorn 실행기를 사용하므로
  별도로 `uvicorn` 을 의존성에 추가하지 않아도 된다.
- 간단한 배포/테스트 환경에서 직접 실행용으로 사용한다.
  (실제 프로덕션에서는 프로세스 매니저나 컨테이너 오케스트레이션과 함께 쓰는 것을 가정한다.)

---

## 4. 주요 HTTP 엔드포인트

백엔드 서버가 제공하는 대표적인 엔드포인트는 다음과 같다.
(정확한 스키마는 `app/models/` 및 각 router 모듈을 참고한다.)

### 4-1. 기본/헬스체크

- `GET /health`
  - 단순 헬스 체크 엔드포인트.
  - 예: `{"status": "ok"}` 형태 응답.
- `POST /echo`
  - 요청 바디를 그대로 되돌려주는 디버그용 엔드포인트.

### 4-2. 행동 분석 (LSTM 기반)

- `POST /behavior/analyze`
  - Body: `BehaviorAnalyzeRequest`
    - feature 시퀀스(프레임별 feature 벡터 리스트)를 포함.
  - Response: `BehaviorAnalyzeResponse`
    - `is_anomaly`, `normal_score`, `anomaly_score`
    - 이벤트 라벨 리스트, per-event score, threshold 등.
  - 내부적으로 `modeling.inference` 의 LSTM 추론을 호출한다.

### 4-3. 이벤트 로그 / 최근 이벤트

- `POST /event/payload`
  - Body: `EventPayload`
    - `event_type` (`"START"|"END"|"PROBS"|"HEARTBEAT"`)
    - 확률 정보, 메타데이터 등.
  - 동작:
    - JSONL 로그 파일에 이벤트를 기록.
    - 최근 이벤트 메모리 캐시를 갱신.
    - 특정 조건(예: END 이벤트 + 이상 행동 의심)에서 Kakao 알림을 트리거.

- `GET /event/recent`
  - 최근에 기록된 이벤트들을 조회.
  - streaming/agent 쪽에서 상태 디버깅용으로 사용할 수 있다.

### 4-4. Kakao 디버그 엔드포인트

- `GET /debug/kakao/token-check`
  - `KAKAO_ACCESS_TOKEN` 이 유효한지 Kakao `/v2/user/me` API로 점검.
  - 토큰 상태를 JSON 형식으로 반환.
- `POST /debug/kakao/send-test`
  - Body: `{ "text": "..." }` (선택, 없으면 기본 테스트 메시지 사용)
  - “나에게 보내기” 테스트 메모를 보낸다.
  - Kakao 알림 설정이 제대로 되었는지 빠르게 확인할 수 있다.

---

## 5. 전형적인 개발/연동 워크플로

1. **환경 구축**

       cd backend
       uv sync --extra cpu      # 또는 필요한 CUDA extra 선택

2. **로컬 개발 서버 실행**

       uv run run-backend-dev

   - 브라우저에서 `http://127.0.0.1:8000/health` 로 상태 확인.
   - 필요하면 `/docs` (Swagger UI)를 열어 API 스펙을 확인한다.

3. **모델 체크포인트/환경 변수 설정**

   - `backend/.env` 또는 시스템 환경 변수에 다음을 설정한다.
     - `MODEL_CHECKPOINT_PATH=/path/to/checkpoint.pt`
     - `KAKAO_ACCESS_TOKEN=...`
     - `ALERT_LINK_URL=...`
     - `ALERT_BUTTON_TITLE=...`

4. **frontend/agent 와 연동**

   - `frontend/agent` 의 `server_stream` 를 실행한다.
   - agent 쪽 설정에서 `BACKEND_BASE_URL` 을
     `http://127.0.0.1:8000` (또는 실제 배포 주소)로 맞춘다.
   - 실시간 스트림에서 `/behavior/analyze`, `/event/payload` 가 정상적으로 호출되는지 확인한다.

5. **실제 구동 모드로 테스트**

       uv run run-backend

   - 자동 리로드가 필요 없는 테스트/배포 환경에서 사용한다.
   - 프로세스 매니저(systemd, supervisord, Docker 등)와 함께 묶어서 운영할 수 있다.
