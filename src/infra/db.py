"""Database connection management.

Currently uses SQLite via SQLAlchemy for lightweight storage.
PostgreSQL is the intended production backend — swap DATABASE_URL to switch.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from src.shared.config import settings

DATABASE_URL = settings.db_url

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency: yields a database session, auto-closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
