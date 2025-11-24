from sqlmodel import SQLModel
from behavior_inference_server import engine

if __name__ == "__main__":
    print("Dropping all tables...")
    SQLModel.metadata.drop_all(engine)
    print("Creating all tables...")
    SQLModel.metadata.create_all(engine)
    print("Done.")
