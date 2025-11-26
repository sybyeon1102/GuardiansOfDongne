"""
⚠ 주의: 이 스크립트는 DATABASE_URL로 연결되는 DB 안의
       '모든 테이블'을 DROP 합니다. (DB 자체는 남고, 테이블만 전부 삭제)
"""

from pathlib import Path

from sqlalchemy import create_engine, MetaData
from project_core.env import load_env, env_str  # behavior_inference_server.py 와 동일한 env 모듈 사용


# behavior_inference_server.py 와 동일한 방식으로 .env 로드
BASE_DIR = Path(__file__).resolve().parent
load_env(BASE_DIR)

# 같은 .env 에서 DATABASE_URL 가져오기
DATABASE_URL = env_str("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다 (.env 확인 요망).")

engine = create_engine(DATABASE_URL, echo=False)


def drop_all_tables() -> None:
    """
    현재 DB에 존재하는 '모든 테이블'을 리플렉트해서 DROP 한다.
    (row만 지우는 게 아니라, 테이블 자체를 삭제)
    """
    metadata = MetaData()
    # 현재 DB에 존재하는 모든 테이블 메타데이터를 읽어오기
    metadata.reflect(bind=engine)

    if not metadata.tables:
        print("드롭할 테이블이 없습니다.")
        return

    table_names = list(metadata.tables.keys())
    print(f"다음 테이블들을 DROP 합니다: {table_names}")

    with engine.begin() as conn:  # 자동 트랜잭션
        metadata.drop_all(bind=conn)

    print("모든 테이블 DROP 완료.")


if __name__ == "__main__":
    print(f"DATABASE_URL = {DATABASE_URL}")
    confirm = input(
        "⚠ 이 스크립트는 위 DATABASE_URL DB 안에 있는 모든 테이블을 DROP 합니다.\n"
        "정말 진행하려면 YES 를 정확히 입력하세요: "
    )
    if confirm.strip() == "YES":
        drop_all_tables()
    else:
        print("취소되었습니다.")
