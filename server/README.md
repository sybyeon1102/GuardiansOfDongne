uv sync

.env 파일 둘 것

개발용 실행
uv run fastapi dev behavior_inference_server.py --host 127.0.0.1 --port 8000

프로덕션용 실행
uv run fastapi run behavior_inference_server.py --host 127.0.0.1 --port 8000
