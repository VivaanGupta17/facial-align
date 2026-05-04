"""Add persistent reviews and provenance columns.

Revision ID: 005
Revises: 004
Create Date: 2026-05-03 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "segmentation_results",
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "segmentation_results",
        sa.Column("structures", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "segmentation_results",
        sa.Column("fracture_fragments", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "segmentation_results",
        sa.Column("fragment_mesh_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.add_column(
        "reduction_plans",
        sa.Column("segmentation_id", postgresql.UUID(as_uuid=False), nullable=True),
    )
    op.add_column(
        "reduction_plans",
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_reduction_plans_segmentation_id",
        "reduction_plans",
        ["segmentation_id"],
        unique=False,
    )

    op.create_table(
        "plan_reviews",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("case_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("plan_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("reviewer_id", sa.String(length=64), nullable=True),
        sa.Column(
            "reviewer_name",
            sa.String(length=128),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "decision",
            sa.String(length=32),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column(
            "notes",
            sa.Text(),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "checklist",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("signed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("case_id", name="uq_plan_reviews_case_id"),
    )
    op.create_index("ix_plan_reviews_case_id", "plan_reviews", ["case_id"], unique=False)
    op.create_index("ix_plan_reviews_plan_id", "plan_reviews", ["plan_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_plan_reviews_plan_id", table_name="plan_reviews")
    op.drop_index("ix_plan_reviews_case_id", table_name="plan_reviews")
    op.drop_table("plan_reviews")

    op.drop_index("ix_reduction_plans_segmentation_id", table_name="reduction_plans")
    op.drop_column("reduction_plans", "provenance")
    op.drop_column("reduction_plans", "segmentation_id")

    op.drop_column("segmentation_results", "fragment_mesh_paths")
    op.drop_column("segmentation_results", "fracture_fragments")
    op.drop_column("segmentation_results", "structures")
    op.drop_column("segmentation_results", "provenance")
