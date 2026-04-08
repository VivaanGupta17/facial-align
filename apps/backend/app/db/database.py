"""
SQLAlchemy async database engine and session management.
Uses asyncpg driver for PostgreSQL.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, MappedColumn
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Module-level engine and session factory (initialized at startup)
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all ORM models.
    Provides shared metadata and common behavior.
    """
    pass


async def create_db_engine() -> AsyncEngine:
    """
    Create and configure the async SQLAlchemy engine.
    Should be called once at application startup.
    """
    global _engine, _session_factory

    logger.info(
        "creating_db_engine",
        host=settings.db.host,
        port=settings.db.port,
        db=settings.db.name,
        pool_size=settings.db.pool_size,
    )

    engine_kwargs: dict = {
        "echo": settings.db.echo_sql,
        "pool_pre_ping": True,  # Verify connections before use
    }

    # Use NullPool for test environments to avoid connection leaks
    if settings.environment == "test":
        engine_kwargs["poolclass"] = NullPool
    else:
        engine_kwargs.update({
            "pool_size": settings.db.pool_size,
            "max_overflow": settings.db.max_overflow,
            "pool_timeout": settings.db.pool_timeout,
            "pool_recycle": 1800,  # Recycle connections every 30 min
        })

    _engine = create_async_engine(
        settings.db.async_url,
        **engine_kwargs,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )

    # Verify connection at startup
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("database_connection_verified")
    except Exception as exc:
        logger.error("database_connection_failed", error=str(exc))
        raise

    return _engine


async def dispose_db_engine() -> None:
    """Dispose the database engine. Call at application shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("database_engine_disposed")


def get_engine() -> AsyncEngine:
    """Get the configured async engine. Raises if not initialized."""
    if _engine is None:
        raise RuntimeError(
            "Database engine not initialized. Call create_db_engine() first."
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the async session factory."""
    if _session_factory is None:
        raise RuntimeError(
            "Session factory not initialized. Call create_db_engine() first."
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI request lifecycle.
    Used in Celery tasks, pipelines, and CLI scripts.

    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    """
    Create all tables defined in ORM models.
    Development utility — use Alembic migrations in production.
    """
    engine = get_engine()
    # Import all models to register with Base.metadata
    from app.models import audit, case, patient, plan, segmentation, study  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def drop_all_tables() -> None:
    """
    Drop all tables. DANGER: only for test/development.
    """
    if settings.environment == "production":
        raise RuntimeError("Cannot drop tables in production environment")
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("database_tables_dropped", environment=settings.environment)
