# GoD Agent (agent)

로컬 / RTSP 영상에서 Mediapipe 포즈를 추출해 추론 서버로 전송하고,
웹 UI가 재생할 수 있는 HLS 스트림을 제공하는 에이전트이다.

- HTTP 서버: FastAPI
- 환경 / 실행: `uv` + `pyproject.toml`
- 영상 스트림: ffmpeg 기반 HLS (H.264 copy, 재인코딩 없음)
- 포즈 추출: Mediapipe Pose

---

## 1. 요구 사항

### System 의존성 (필수)

에이전트는 HLS 생성을 위해 `ffmpeg` CLI를 사용한다. 아래 방법 중 하나로 설치한다.

- Windows (scoop)

      scoop install ffmpeg

- macOS (Homebrew)

      brew install ffmpeg

- Ubuntu / Debian

      sudo apt update
      sudo apt install -y ffmpeg

설치 확인:

    ffmpeg -version

---

## 2. uv로 환경 구성

프로젝트 루트(또는 agent 프로젝트 루트)에서 다음을 실행한다.

    uv sync

`uv sync` 는 `pyproject.toml` 기반으로 가상환경 생성과 패키지 설치를 한 번에 수행한다.

---

## 3. `.env` 설정 (agent/.env)

`agent` 디렉터리 안에 `.env` 파일을 만들고, 아래 템플릿을 필요에 맞게 수정해서 사용한다.

    # 추론 서버 (behavior_inference_server / stream_lstm_app)
    INFERENCE_SERVER_URL=http://localhost:8000

    # 에이전트가 읽을 영상/카메라 소스들
    # - 개발: H.264 mp4 파일
    # - 실전: rtsp://... 주소들
    AGENT_SOURCES=videos/cam01.mp4,videos/cam02.mp4

    # 각 소스에 대응하는 camera_id (콤마로 구분, 인덱스로 매칭)
    AGENT_CAMERA_IDS=cam01,cam02

    # Mediapipe / 윈도우 설정 (legacy c_realtime_client 기본값과 동일)
    AGENT_WINDOW_SIZE=16
    AGENT_STRIDE=4
    AGENT_RESIZE_WIDTH=640

    # 프레임 스킵 (1이면 매 프레임 Pose, 2면 2프레임마다 한 번 Pose)
    AGENT_FRAME_SKIP=1

    # HLS 세그먼트가 저장될 루트 디렉터리
    # FastAPI에서 /hls 경로로 서빙된다.
    AGENT_HLS_ROOT=./hls

예를 들어 실제 RTSP 카메라를 사용한다면 다음과 같이 설정할 수 있다.

    AGENT_SOURCES=rtsp://192.168.0.10/stream1,rtsp://192.168.0.11/stream1
    AGENT_CAMERA_IDS=cam01,cam02

`AGENT_SOURCES` 와 `AGENT_CAMERA_IDS` 는 콤마(`,`)로 구분된 리스트이고, 동일한 인덱스로 매칭된다.

---

## 4. 에이전트 실행

FastAPI 앱 엔트리는 `agent/agent_app.py` 의 `app` 이다.
`agent` 디렉터리에서 다음과 같이 실행한다.

    uv run fastapi dev agent_app.py --host 0.0.0.0 --port 9001

에이전트가 기동되면 다음을 수행한다.

- `.env` 에서 소스 / 카메라 설정을 읽는다.
- 각 소스에 대해 `ffmpeg` 프로세스를 1개씩 띄워 HLS 세그먼트를 생성한다.
- 각 소스에 대해 Mediapipe Pose 워커 스레드를 시작하고,
  추출된 포즈 윈도우를 추론 서버의 `/behavior/analyze_pose` 엔드포인트로 전송한다.

---

## 5. 제공 엔드포인트

### 5.1 Health 체크

    GET /health

예시 응답:

    {
      "status": "ok"
    }

### 5.2 HLS 스트림

HLS 루트는 `/hls` 로 마운트된다. 플레이리스트 위치는 다음과 같다.

- `/hls/<camera_id>/index.m3u8`

예시 (`AGENT_CAMERA_IDS=cam01,cam02` 인 경우):

- `http://localhost:9001/hls/cam01/index.m3u8`
- `http://localhost:9001/hls/cam02/index.m3u8`

React / 웹 UI에서의 사용 예시는 다음과 같다.

    <video
      src="http://localhost:9001/hls/cam01/index.m3u8"
      controls
    />

브라우저가 HLS를 직접 지원하지 않는 경우 HLS.js 등의 라이브러리를 사용한다.

---

## 6. 트러블슈팅

### 6.1 ffmpeg 관련 에러

에이전트 시작 시 다음과 같은 메시지가 출력될 수 있다.

    [Agent] ffmpeg 실행 파일을 찾을 수 없습니다...

이 경우 ffmpeg가 설치되어 있지 않거나 PATH에 등록되어 있지 않은 상태이다.
위 “System 의존성” 섹션을 참고해 ffmpeg를 설치한 뒤 다시 실행한다.

### 6.2 HLS 플레이리스트가 보이지 않는 경우

- `AGENT_HLS_ROOT` 아래에 `<camera_id>/index.m3u8` 가 생성되지 않았다면:
  - `AGENT_SOURCES` 주소가 잘못되었거나
  - RTSP 연결 실패 / 파일 경로 오류일 가능성이 있다.
- 에이전트 로그의 `[HLS:camXX]` 라인을 확인해 ffmpeg 에러가 있는지 확인한다.

### 6.3 추론 서버 연동 문제가 있는 경우

- `.env` 의 `INFERENCE_SERVER_URL` 이 실제 추론 서버 주소와 일치하는지 확인한다.
- 추론 서버가 `/behavior/analyze_pose` 엔드포인트를 제공하는지 확인한다.
- 에이전트 로그에 HTTP 4xx/5xx 에러나 타임아웃이 찍히는지 확인한다.
