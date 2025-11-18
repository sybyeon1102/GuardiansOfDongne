from typing import Any

import time

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.router_behavior import router as router_behavior
from app.routers.router_event import router as router_event


app = FastAPI(
    title="Behavior Backend",
    description=(
        "LSTM 기반 정상/이상 행동 분석 및 이벤트 로깅 백엔드.\n"
        "Backend service for LSTM-based behavior analysis and event logging."
    ),
    version="0.1.0",
)

# CORS 설정 / CORS configuration
# 원래 c_server.py 에서 allow_origins=[\"*\"] 로 열어두었던 것을 계승한다.
# Inherit the open CORS policy from the original c_server.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 나중에 구체적인 오리진으로 좁힐 수 있다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """
    헬스 체크 엔드포인트.
    Health check endpoint.
    """
    return {"status": "ok"}


@app.post("/echo")
async def echo(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    테스트용 echo 엔드포인트.

    원래 c_server.py 의 /echo 와 동일하게,
    요청 바디를 그대로 되돌려주고 현재 시각(UNIX epoch)을 함께 반환한다.

    Test echo endpoint, similar to /echo in the legacy c_server.py.
    """
    return {
        "echo": body,
        "ts": time.time(),
    }


# 도메인 라우터 등록 / Register domain routers
app.include_router(router_behavior, prefix="/behavior", tags=["behavior"])
app.include_router(router_event, prefix="/event", tags=["event"])
