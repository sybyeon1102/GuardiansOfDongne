# Frontend 서브 프로젝트

이 디렉터리는 **사용자와 직접 맞닿는 프론트엔드 계층**을 담당하는 `frontend` 서브 프로젝트이다.
브라우저에서 동작하는 웹 UI와, 카메라/RTSP 스트림을 처리하는 에이전트(Agent)를 함께 포함한다.

현재 구조는 대략 다음과 같다.

- `frontend/web/`
  정적 `index.html` + JavaScript 로 구성된 웹 UI.
- `frontend/agent/`
  Python(FastAPI 기반)으로 구현된 스트리밍 에이전트.
  카메라/RTSP → 포즈/feature 추출 → Backend `/behavior/analyze` 연동을 담당한다.

이 문서는 **`frontend/` 전체에 대한 개요**를 다루며,
세부 내용은 각 서브 디렉터리의 README를 참고한다.

- `frontend/agent/README.md` : 에이전트 서브 프로젝트 상세
- `frontend/web/README.md`   : 정적 웹 자원/역할에 대한 간단한 설명

---

## 1. 역할 및 위치

이 레포의 상위 구조는 다음과 같다.

- `core/` : 언어 공용 코어 유틸 계층 (`project-core` 등)
- `modeling/` : 데이터 전처리 + LSTM 모델 학습·추론
- `backend/` : FastAPI 기반 HTTP 서버 (행동 분석 API, 이벤트 로그, Kakao 알림)
- `frontend/` : 웹 UI + 스트리밍 에이전트

의존성 방향은 불변식 **[INV-064]** 에 따라 다음과 같이 흐른다.

    project_core → modeling → backend → frontend

즉, frontend 계층은:

- Python 코드 관점에서는 `project_core` 정도만 import 할 수 있고,
- 실제 기능은 **HTTP API를 통해 backend에 요청**하는 것을 기본으로 한다.

---

## 2. frontend/web 개요

`frontend/web/` 디렉터리는 현재 다음과 같은 역할을 가진다.

- 정적 `index.html` 및 내장 JavaScript로 구성된 **간단한 웹 UI**를 제공한다.
- 브라우저에서:
  - 카메라/RTSP 스트림을 선택·제어하거나,
  - Backend/Agent와 연동하는 버튼/폼 등을 제공한다.
- 정적 파일 자체는:
  - `frontend/agent` 의 FastAPI(예: `/`)에서 서빙하거나,
  - 별도의 정적 파일 서버(Nginx, S3, 간단한 HTTP 서버 등)를 통해 서빙할 수 있다.

현재 라운드에서는:

- 별도의 TypeScript/번들링 환경을 두지 않고,
  **단일 HTML + JS 파일** 수준으로 유지한다.
- 프레임워크(React / Svelte 등) 도입 여부는 **차후 라운드에서 결정**한다.

자세한 내용과 향후 계획은 `frontend/web/README.md` 에서 간단히 메모한다.

---

## 3. frontend/agent 개요

`frontend/agent/` 디렉터리는 **스트리밍 에이전트 서브 프로젝트**를 담는다.

역할 요약:

- 로컬 카메라 또는 RTSP 스트림을 읽는다.
- OpenCV + MediaPipe 등을 사용해 포즈/feature를 추출한다.
- 추출된 feature 시퀀스를 Backend의 `/behavior/analyze` API로 전송해
  **정상/이상 행동 분석 결과**를 받아온다.
- 필요하면 `/event/payload` 등 Backend 이벤트 API로도 데이터를 전송한다.
- 브라우저(웹 UI)와 연동하여:
  - 스트림 상태/확률/이상 이벤트를 시각화하거나,
  - 조작 버튼(시작/중지 등)을 제공하는 역할을 한다.

`frontend/agent` 는 별도의 Python 서브 프로젝트로 관리되며:

- 자체 `pyproject.toml` + `uv` 환경을 가진다.
- FastAPI 기반 서버 엔트리(개발용/실행용)를 [project.scripts] 로 정의한다.
- 자세한 커맨드라인 사용법, 환경 구축(uv + OpenCV 시스템 의존성),
  HTTP 엔드포인트는 `frontend/agent/README.md` 에서 다룬다.

---

## 4. Backend 와의 연동 구조

전체 흐름을 단순화하면 다음과 같다.

1. **모델링 계층 (`modeling/`)**
   - 전처리 스크립트로 `X.npy`, `Y.npy`, `meta.json` 생성
   - LSTM 모델 학습 및 체크포인트(.pt) 생성
2. **백엔드 (`backend/`)**
   - LSTM 추론을 호출하는 `/behavior/analyze` API 제공
   - 이벤트 기록용 `/event/payload`, `/event/recent` 제공
   - Kakao 알림용 내부/디버그 엔드포인트 제공
3. **프론트엔드 에이전트 (`frontend/agent`)**
   - 카메라/RTSP 스트림을 읽고 feature 시퀀스를 만든다.
   - Backend `/behavior/analyze` 로 시퀀스를 전송한다.
   - Backend 응답(normal/anomaly, scores, events)을 받아
     웹 UI / 로그 등에 반영한다.
4. **웹 UI (`frontend/web`)**
   - 사용자가 브라우저에서 접근하는 진입점 역할을 한다.
   - 필요 시 `frontend/agent` 와 HTTP/WebSocket 등으로 통신하거나,
     직접 Backend API를 호출할 수도 있다(현재 구조에 따라).

이 설계는 “백엔드는 네트워크/API 계층, 모델링은 별도 서비스로 뽑을 수 있음”이라는
장기적인 리팩터링 방향을 유지하면서,
프론트엔드는 **HTTP API 소비자** 역할에 집중하게 하기 위한 것이다.

---

## 5. 향후 계획 (메모)

- `frontend/web`
  - TypeScript + 번들러(예: Vite) + React/Svelte 기반 구조로 교체/확장할 수 있다.
  - 그 경우에도 **Backend API 계약(/behavior/analyze, /event/...)**은 유지한다.
- `frontend/agent`
  - 현재는 Python + FastAPI 기반 구현을 사용하지만,
  - 장기적으로는 다른 언어/런타임(예: Go, Rust, Node) 기반의 에이전트로 교체할 수 있다.
  - 교체 시에도 Backend API 계약을 그대로 재사용하는 것을 목표로 한다.

프론트엔드와 관련된 구체적인 실행 커맨드, 환경 구축, 의존성 정보는
각 서브 디렉터리(특히 `frontend/agent/README.md`)에서 상세하게 다룬다.
