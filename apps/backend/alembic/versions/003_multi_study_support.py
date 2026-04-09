"""Add case_studies junction table for multi-study per case support.

Revision ID: 003
Revises: 002
Create Date: 2024-01-20 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "case_studies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "case_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("surgical_cases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "study_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("imaging_studies.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "study_role",
            sa.String(32),
            server_default=sa.text("'pre_op'"),
            nullable=False,
        ),
        sa.Column("study_label", sa.String(128), nullable=True),
        sa.Column(
            "is_primary",
            sa.Boolean(),
            server_default=sa.text("FALSE"),
            nullable=False,
        ),
        sa.Column(
            "display_order",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("case_id", "study_id", name="uq_case_study"),
    )
    op.create_index("ix_case_studies_case_id", "case_studies", ["case_id"])
    op.create_index("ix_case_studies_study_id", "case_studies", ["study_id"])
    op.create_index("ix_case_studies_study_role", "case_studies", ["study_role"])


def downgrade() -> None:
    op.drop_index("ix_case_studies_study_role", table_name="case_studies")
    op.drop_index("ix_case_studies_study_id", table_name="case_studies")
    op.drop_index("ix_case_studies_case_id", table_name="case_studies")
    op.drop_table("case_studies")
