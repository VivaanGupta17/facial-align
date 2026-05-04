"""Add institution_code to users for institution-scoped authorization.

Revision ID: 006
Revises: 005
Create Date: 2026-05-04 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("institution_code", sa.String(length=32), nullable=True))
    op.create_index("ix_users_institution_code", "users", ["institution_code"])

    op.execute(
        sa.text(
            """
            UPDATE users
            SET institution_code = CASE
                WHEN email = 'surgeon@facialign.local' THEN 'DEMO-INST'
                WHEN email = 'reviewer@facialign.local' THEN 'DEMO-INST'
                ELSE institution_code
            END
            WHERE institution_code IS NULL
            """
        )
    )


def downgrade() -> None:
    op.drop_index("ix_users_institution_code", table_name="users")
    op.drop_column("users", "institution_code")
