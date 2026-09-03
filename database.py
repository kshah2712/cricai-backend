import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # device-generated UUID for now, becomes Firebase UID later
    username = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    match_id = Column(String, nullable=False)
    match_name = Column(String, nullable=True)
    teams = Column(String, nullable=True)

    user_predicted_winner = Column(String, nullable=False)
    ai_predicted_winner = Column(String, nullable=True)
    ai_confidence = Column(Integer, nullable=True)
    ai_reasoning = Column(String, nullable=True)

    actual_winner = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending | resolved
    user_correct = Column(Boolean, nullable=True)
    ai_correct = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="predictions")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()