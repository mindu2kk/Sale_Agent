"""
Async CRUD operations for Thread and Message.
"""

import uuid
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Thread, Message

logger = logging.getLogger(__name__)


async def create_thread(db: AsyncSession) -> Thread:
    """Create a new active thread with a UUID v4 id."""
    thread = Thread(id=str(uuid.uuid4()), status="active")
    db.add(thread)
    await db.commit()
    await db.refresh(thread)
    return thread


async def get_thread(db: AsyncSession, thread_id: str) -> Optional[Thread]:
    """Fetch a thread by id. Returns None if not found."""
    result = await db.execute(select(Thread).where(Thread.id == thread_id))
    return result.scalar_one_or_none()


async def save_message(
    db: AsyncSession,
    thread_id: str,
    role: str,
    content: str,
    *,
    retries: int = 3,
) -> Optional[Message]:
    """
    Persist a message with exponential backoff retry on failure.
    Returns the saved Message or None if all retries fail.
    """
    delays = [1, 2, 4]
    for attempt in range(retries):
        try:
            msg = Message(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                role=role,
                content=content,
            )
            db.add(msg)
            # Update thread updated_at
            await db.execute(
                update(Thread)
                .where(Thread.id == thread_id)
                .values(updated_at=datetime.now(timezone.utc))
            )
            await db.commit()
            await db.refresh(msg)
            return msg
        except Exception as exc:
            await db.rollback()
            if attempt < retries - 1:
                logger.warning("save_message attempt %d failed: %s — retrying", attempt + 1, exc)
                await asyncio.sleep(delays[attempt])
            else:
                logger.error("save_message failed after %d retries: %s", retries, exc)
                return None


async def get_messages(
    db: AsyncSession,
    thread_id: str,
    limit: int = 50,
    before_id: Optional[str] = None,
) -> tuple[list[Message], bool]:
    """
    Fetch messages for a thread in ascending created_at order.
    Supports cursor-based pagination via before_id.
    Returns (messages, has_more).
    """
    query = select(Message).where(Message.thread_id == thread_id)

    if before_id:
        # Get the cursor message's timestamp
        cursor_result = await db.execute(
            select(Message.created_at).where(Message.id == before_id)
        )
        cursor_ts = cursor_result.scalar_one_or_none()
        if cursor_ts:
            query = query.where(Message.created_at < cursor_ts)

    query = query.order_by(Message.created_at.asc()).limit(limit + 1)
    result = await db.execute(query)
    rows = result.scalars().all()

    has_more = len(rows) > limit
    return list(rows[:limit]), has_more


async def update_thread_metadata(
    db: AsyncSession,
    thread_id: str,
    escalated: bool = False,
) -> None:
    """Update thread metadata flags."""
    await db.execute(
        update(Thread)
        .where(Thread.id == thread_id)
        .values(escalated=escalated, updated_at=datetime.now(timezone.utc))
    )
    await db.commit()


async def expire_inactive_threads(db: AsyncSession) -> int:
    """Mark threads inactive for >24h as expired. Returns count of expired threads."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    result = await db.execute(
        update(Thread)
        .where(Thread.status == "active", Thread.updated_at < cutoff)
        .values(status="expired")
        .returning(Thread.id)
    )
    await db.commit()
    expired = result.fetchall()
    return len(expired)
