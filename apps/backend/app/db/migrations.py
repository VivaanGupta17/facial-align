"""
Alembic migration configuration placeholder.
The actual migration environment is in alembic/ directory.

This module provides programmatic migration utilities for use in
CI/CD pipelines and application startup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection, inspect

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

ALEMBIC_INI_PATH = Path(__file__).parent.parent.parent / "alembic.ini"
ALEMBIC_SCRIPTS_PATH = Path(__file__).parent.parent.parent / "alembic"


def get_alembic_config(dsn: Optional[str] = None) -> Config:
    """
    Build Alembic Config object pointing to the project's alembic.ini.

    Args:
        dsn: Override database URL (default: from settings)
    """
    if not ALEMBIC_INI_PATH.exists():
        # Create a minimal in-memory config if alembic.ini doesn't exist yet
        cfg = Config()
        cfg.set_main_option("script_location", str(ALEMBIC_SCRIPTS_PATH))
    else:
        cfg = Config(str(ALEMBIC_INI_PATH))

    db_url = dsn or settings.db.sync_url
    cfg.set_main_option("sqlalchemy.url", db_url)

    return cfg


def run_migrations_upgrade(revision: str = "head", dsn: Optional[str] = None) -> None:
    """
    Run Alembic upgrade migrations.

    Args:
        revision: Target revision (default "head" = latest)
        dsn: Database DSN override
    """
    cfg = get_alembic_config(dsn)
    logger.info("running_alembic_upgrade", target=revision)
    try:
        command.upgrade(cfg, revision)
        logger.info("alembic_upgrade_complete", target=revision)
    except Exception as exc:
        logger.error("alembic_upgrade_failed", error=str(exc), target=revision)
        raise


def run_migrations_downgrade(revision: str, dsn: Optional[str] = None) -> None:
    """
    Roll back Alembic migrations to a revision.

    Args:
        revision: Target revision or relative (-1, -2, etc.)
        dsn: Database DSN override
    """
    if settings.environment == "production":
        raise RuntimeError("Automatic downgrade is disabled in production")
    cfg = get_alembic_config(dsn)
    logger.warning("running_alembic_downgrade", target=revision)
    command.downgrade(cfg, revision)


def generate_migration(message: str, autogenerate: bool = True) -> None:
    """
    Generate a new migration revision.

    Args:
        message: Migration description
        autogenerate: Whether to auto-detect model changes
    """
    cfg = get_alembic_config()
    command.revision(cfg, message=message, autogenerate=autogenerate)
    logger.info("migration_generated", message=message)


def get_current_revision(connection: Connection) -> Optional[str]:
    """
    Get the current migration revision from the database.

    Args:
        connection: SQLAlchemy connection

    Returns:
        Current revision hash or None if no migrations applied
    """
    context = MigrationContext.configure(connection)
    return context.get_current_revision()


def check_migrations_current(connection: Connection) -> bool:
    """
    Check whether the database schema is up-to-date.

    Returns:
        True if schema is at head revision
    """
    from alembic.script import ScriptDirectory
    cfg = get_alembic_config()
    script = ScriptDirectory.from_config(cfg)
    head_revision = script.get_current_head()
    current = get_current_revision(connection)
    return current == head_revision


# ─── Alembic env.py template ──────────────────────────────────────────────────
# The actual alembic/env.py should contain this logic:
#
# from logging.config import fileConfig
# from sqlalchemy import engine_from_config, pool
# from alembic import context
# from app.db.database import Base
# from app.models import patient, study, case, segmentation, plan, audit
#
# config = context.config
# if config.config_file_name is not None:
#     fileConfig(config.config_file_name)
#
# target_metadata = Base.metadata
#
# def run_migrations_offline():
#     url = config.get_main_option("sqlalchemy.url")
#     context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
#     with context.begin_transaction():
#         context.run_migrations()
#
# def run_migrations_online():
#     connectable = engine_from_config(config.get_section(config.config_ini_section),
#                                       prefix="sqlalchemy.", poolclass=pool.NullPool)
#     with connectable.connect() as connection:
#         context.configure(connection=connection, target_metadata=target_metadata)
#         with context.begin_transaction():
#             context.run_migrations()
#
# if context.is_offline_mode():
#     run_migrations_offline()
# else:
#     run_migrations_online()
