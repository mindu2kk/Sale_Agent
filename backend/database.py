"""
Async SQLite database setup using SQLAlchemy + aiosqlite.
WAL mode enabled to prevent "database is locked" under concurrent writes.
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./chat.db")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables and enable WAL mode."""
    from backend.models import Thread, Message  # noqa: F401 — ensure models are registered

    async with engine.begin() as conn:
        # WAL mode: allows concurrent reads + 1 writer without locking
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        # Must use run_sync for DDL operations with AsyncConnection
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an AsyncSession, closes after request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
