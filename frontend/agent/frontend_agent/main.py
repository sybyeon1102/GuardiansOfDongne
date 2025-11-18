"""
frontend-agent FastAPI 엔트리 포인트.

FastAPI CLI(fastapi dev/run)가 참조하는 app 객체를 노출한다.
The FastAPI CLI (fastapi dev/run) uses this module as the entry point.
"""

from frontend_agent.server_stream import app as agent_app


# FastAPI가 찾을 앱 객체 / FastAPI app object for the CLI
app = agent_app
