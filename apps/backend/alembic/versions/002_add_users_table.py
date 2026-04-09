"""Add users table with default admin and surgeon seed data.

Revision ID: 002
Revises: 001
Create Date: 2024-01-15 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# Alembic revision identifiers
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="viewer"),
        sa.Column("institution", sa.String(255), nullable=True),
        sa.Column("specialty", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("login_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # Seed default users — passwords hashed with bcrypt (passlib)
    # admin@facialign.local / admin
    # surgeon@facialign.local / surgeon
    #
    # Pre-computed bcrypt hashes so migration is self-contained and doesn't
    # depend on passlib being importable in the Alembic env.
    from passlib.context import CryptContext
    _pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

    op.execute(
        sa.text(
            """
            INSERT INTO users (id, email, hashed_password, full_name, role, is_active, is_verified)
            VALUES
                (gen_random_uuid(), 'admin@facialign.local', :admin_pw, 'Admin User', 'admin', true, true),
                (gen_random_uuid(), 'surgeon@facialign.local', :surgeon_pw, 'Demo Surgeon', 'surgeon', true, true)
            """
        ).bindparams(
            admin_pw=_pwd.hash("admin"),
            surgeon_pw=_pwd.hash("surgeon"),
        )
    )


def downgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
