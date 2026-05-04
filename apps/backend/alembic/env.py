"""
Alembic environment configuration for Facial Align backend.

Supports both online mode (async, using asyncpg) and offline mode
(synchronous, generates SQL scripts). Database URL is read from
app.core.config.AppSettings so that credentials are never stored in
alembic.ini or committed to source control.

Usage:
    # Apply pending migrations
    alembic upgrade head

    # Generate SQL script for offline deployment
    alembic upgrade head --sql

    # Create a new auto-generated migration
    alembic revision --autogenerate -m "add some_table"

    # Downgrade one step
    alembic downgrade -1
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path
from typing import Optional

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Path setup ───────────────────────────────────────────────────────────────
# Ensure the backend app package is importable when running Alembic from the
# apps/backend/ directory.
_HERE = Path(__file__).resolve().parent          # apps/backend/alembic/
_BACKEND_ROOT = _HERE.parent                     # apps/backend/
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ── Import app models and config ─────────────────────────────────────────────
# Importing all models registers them against Base.metadata so that
# --autogenerate can detect schema differences.
from app.core.config import get_settings                         # noqa: E402
from app.db.database import Base                                 # noqa: E402

# Register all ORM models with Base.metadata
from app.models import audit       # noqa: F401, E402
from app.models import case        # noqa: F401, E402
from app.models import patient     # noqa: F401, E402
from app.models import plan        # noqa: F401, E402
from app.models import segmentation  # noqa: F401, E402
from app.models import study       # noqa: F401, E402
from app.models import user        # noqa: F401, E402

# ── Alembic config object ─────────────────────────────────────────────────────
# Provides access to the alembic.ini values.
config = context.config

# ── Logging ───────────────────────────────────────────────────────────────────
# Interpret the alembic.ini logging section if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Target metadata ───────────────────────────────────────────────────────────
# This is what Alembic compares against the live database for --autogenerate.
target_metadata = Base.metadata


# ── Database URL resolution ───────────────────────────────────────────────────

def _get_database_url() -> str:
    """
    Resolve the database URL for Alembic migrations.

    Priority order:
    1. ALEMBIC_DATABASE_URL environment variable (allows CI/CD override)
    2. DATABASE_URL environment variable (Docker Compose convention)
    3. AppSettings.db.sync_url (psycopg2, required for offline/sync mode)

    Alembic's offline mode uses a synchronous driver; online async mode uses
    the asyncpg-based URL from AppSettings.db.async_url.
    """
    # Allow explicit override for migration-specific credentials
    alembic_url = os.environ.get("ALEMBIC_DATABASE_URL")
    if alembic_url:
        return alembic_url

    generic_url = os.environ.get("DATABASE_URL")
    if generic_url:
        return generic_url

    settings = get_settings()
    # Return synchronous URL for offline mode; env.py replaces driver for async
    return settings.db.sync_url


def _get_async_database_url() -> str:
    """Return asyncpg-based URL for online async migrations."""
    alembic_url = os.environ.get("ALEMBIC_DATABASE_URL")
    if alembic_url:
        # If caller supplied a psycopg2 URL, swap to asyncpg
        return alembic_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://") \
                           .replace("postgresql://", "postgresql+asyncpg://")

    settings = get_settings()
    return settings.db.async_url


# ── Offline migration mode ────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This generates SQL DDL to stdout (or a file when --sql is used) without
    requiring a live database connection. Useful for generating deployment
    scripts reviewed by a DBA before applying.

    In offline mode, SQLAlchemy uses a synchronous driver.
    """
    url = _get_database_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schemas in comparison (useful for multi-schema setups)
        include_schemas=False,
        # Compare server defaults so Alembic detects DEFAULT value changes
        compare_server_defaults=True,
        # Compare column types strictly
        compare_type=True,
        # Render item-level comments in generated SQL
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online migration mode (async) ─────────────────────────────────────────────

def do_run_migrations(connection: Connection) -> None:
    """Execute migrations using the provided synchronous connection handle."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detect server default changes
        compare_server_defaults=True,
        # Detect column type changes
        compare_type=True,
        # Include schema name in object names if using non-public schemas
        include_schemas=False,
        # Transaction per migration for safer rollback on failure
        transaction_per_migration=False,
        # Render item-level comments in generated DDL
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an async engine and run migrations within a synchronous connection
    wrapper, as required by Alembic's connection-based API.

    Uses NullPool so that the engine does not hold idle connections after
    migrations complete — important for one-off migration CLI runs.
    """
    url = _get_async_database_url()

    # Build configuration dict for async engine
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.begin() as connection:
        # Verify connection is alive before proceeding
        await connection.execute(text("SELECT 1"))

        # run_sync wraps the synchronous Alembic migration runner inside the
        # async connection context
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode using the async engine.

    This is the default mode invoked when running `alembic upgrade head`
    against a live database.
    """
    asyncio.run(run_async_migrations())


# ── Entry point ───────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
