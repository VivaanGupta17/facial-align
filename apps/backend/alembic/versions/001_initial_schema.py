"""Initial schema — all tables for Facial Align craniofacial surgical platform.

Creates the full database schema matching the SQLAlchemy ORM models defined in
app.models: patients, imaging_studies, surgical_cases, segmentation_results,
reduction_plans, and audit_logs.

All primary keys use UUID (gen_random_uuid() server default via pgcrypto).
All timestamps use TIMESTAMP WITH TIME ZONE.
JSONB columns are used for complex nested structures.

Revision ID: 001
Revises: (none — initial migration)
Create Date: 2024-01-01 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID

from alembic import op

# Alembic revision identifiers
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _uuid_pk() -> sa.Column:
    """Return a standard UUID primary key column with server-side default."""
    return sa.Column(
        "id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
        nullable=False,
        comment="Unique record identifier (UUID v4)",
    )


def _now_tz() -> sa.text:
    """Return a CURRENT_TIMESTAMP server default (with timezone)."""
    return sa.text("CURRENT_TIMESTAMP")


# ── upgrade ───────────────────────────────────────────────────────────────────

def upgrade() -> None:
    """Create all tables in dependency order."""

    # ── Enable pgcrypto for gen_random_uuid() ────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ── patients ─────────────────────────────────────────────────────────────
    # De-identified patient records. No PHI in plaintext.
    op.create_table(
        "patients",
        _uuid_pk(),
        # De-identified identifiers
        sa.Column(
            "mrn_hash",
            sa.String(64),
            nullable=False,
            unique=True,
            comment="SHA-256(secret_salt + MRN). Allows patient lookup without storing MRN.",
        ),
        # Encrypted PHI blob
        sa.Column(
            "demographics_encrypted",
            sa.Text,
            nullable=True,
            comment=(
                "Base64-encoded AES-256-GCM encrypted JSON: "
                "{dob_year, sex, ethnicity, referring_institution}. Nonce prepended."
            ),
        ),
        # Non-PHI metadata
        sa.Column(
            "institution_code",
            sa.String(32),
            nullable=True,
            comment="Treating institution identifier (anonymized code)",
        ),
        sa.Column(
            "age_at_registration",
            sa.Integer,
            nullable=True,
            comment="Age in years at time of registration (non-identifying for adults)",
        ),
        sa.Column(
            "sex",
            sa.String(1),
            nullable=True,
            comment="Biological sex: M, F, O (other), U (unknown)",
        ),
        # Audit fields
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
            comment="Record creation timestamp (UTC)",
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
            comment="Record last update timestamp (UTC)",
        ),
        sa.Column(
            "created_by",
            sa.String(64),
            nullable=True,
            comment="User ID who created this record",
        ),
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("TRUE"),
            comment="Soft delete flag",
        ),
    )
    op.create_index("ix_patients_mrn_hash", "patients", ["mrn_hash"], unique=True)
    op.create_index("ix_patients_created_at", "patients", ["created_at"])
    op.create_index("ix_patients_institution_code", "patients", ["institution_code"])

    # ── imaging_studies ───────────────────────────────────────────────────────
    # DICOM imaging studies. FK to patients.
    op.create_table(
        "imaging_studies",
        _uuid_pk(),
        # DICOM identifiers
        sa.Column(
            "study_uid",
            sa.String(128),
            nullable=False,
            unique=True,
            comment="DICOM StudyInstanceUID (de-identified)",
        ),
        sa.Column(
            "accession_number",
            sa.String(64),
            nullable=True,
            comment="DICOM AccessionNumber (de-identified)",
        ),
        # Foreign key
        sa.Column(
            "patient_id",
            UUID(as_uuid=False),
            sa.ForeignKey("patients.id", ondelete="RESTRICT", name="fk_imaging_studies_patient_id"),
            nullable=False,
            comment="FK to patients.id",
        ),
        # Study metadata
        sa.Column(
            "modality",
            sa.String(16),
            nullable=False,
            comment="Primary DICOM modality: CT, CBCT, MR, etc.",
        ),
        sa.Column(
            "acquisition_date",
            sa.Date,
            nullable=True,
            comment="Date imaging was acquired",
        ),
        sa.Column(
            "series_count",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
            comment="Number of series in this study",
        ),
        sa.Column(
            "slice_count",
            sa.Integer,
            nullable=True,
            comment="Total number of slices across all series",
        ),
        # Storage
        sa.Column(
            "storage_path",
            sa.Text,
            nullable=False,
            comment="Path to de-identified DICOM directory or s3://bucket/key/prefix",
        ),
        sa.Column(
            "volume_path",
            sa.Text,
            nullable=True,
            comment="Path to reconstructed NIfTI volume (post-ingestion)",
        ),
        # Imaging parameters
        sa.Column(
            "slice_thickness_mm",
            sa.Float,
            nullable=True,
            comment="CT slice thickness in mm",
        ),
        sa.Column(
            "pixel_spacing_mm",
            sa.Float,
            nullable=True,
            comment="In-plane pixel spacing in mm",
        ),
        sa.Column(
            "kv_peak",
            sa.Float,
            nullable=True,
            comment="CT tube voltage (kVp)",
        ),
        sa.Column(
            "body_part_examined",
            sa.String(64),
            nullable=True,
            comment="DICOM BodyPartExamined tag value",
        ),
        # JSON metadata
        sa.Column(
            "metadata_json",
            JSONB,
            nullable=True,
            comment="Selected DICOM metadata tags as key-value pairs (de-identified)",
        ),
        # Quality assessment
        sa.Column(
            "quality_score",
            sa.Float,
            nullable=True,
            comment="Automated CT quality assessment score [0-1]",
        ),
        sa.Column(
            "quality_flags",
            JSONB,
            nullable=True,
            comment="List of quality warning strings detected during ingestion",
        ),
        sa.Column(
            "is_deidentified",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("FALSE"),
            comment="Whether PHI has been removed from DICOM tags",
        ),
        sa.Column(
            "ingestion_status",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'pending'"),
            comment="Ingestion pipeline status: pending, processing, complete, failed",
        ),
        # Audit
        sa.Column("uploaded_by", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
        ),
    )
    op.create_index("ix_imaging_studies_study_uid", "imaging_studies", ["study_uid"], unique=True)
    op.create_index("ix_imaging_studies_patient_id", "imaging_studies", ["patient_id"])
    op.create_index("ix_imaging_studies_acquisition_date", "imaging_studies", ["acquisition_date"])
    op.create_index("ix_imaging_studies_modality", "imaging_studies", ["modality"])
    op.create_index("ix_imaging_studies_ingestion_status", "imaging_studies", ["ingestion_status"])

    # ── surgical_cases ────────────────────────────────────────────────────────
    # Central case entity linking patient, study, segmentation, and plan.
    op.create_table(
        "surgical_cases",
        _uuid_pk(),
        # Human-readable reference
        sa.Column(
            "case_number",
            sa.String(32),
            nullable=False,
            unique=True,
            comment="Human-readable case reference number (e.g. FA-2024-0042)",
        ),
        # Foreign keys
        sa.Column(
            "patient_id",
            UUID(as_uuid=False),
            sa.ForeignKey("patients.id", ondelete="RESTRICT", name="fk_surgical_cases_patient_id"),
            nullable=False,
        ),
        sa.Column(
            "study_id",
            UUID(as_uuid=False),
            sa.ForeignKey("imaging_studies.id", ondelete="RESTRICT", name="fk_surgical_cases_study_id"),
            nullable=False,
        ),
        # Classification
        sa.Column(
            "case_type",
            sa.String(32),
            nullable=False,
            comment="Clinical case type: TRAUMA, ORTHOGNATHIC, RECONSTRUCTION, TUMOR, CONGENITAL",
        ),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'CREATED'"),
            comment="Case lifecycle status: CREATED, DICOM_PROCESSING, SEGMENTED, PLANNING, PLANNED, REVIEWED, APPROVED, ARCHIVED, FAILED",
        ),
        # Clinical context
        sa.Column(
            "diagnosis_codes",
            JSONB,
            nullable=True,
            comment="List of ICD-10 diagnosis codes",
        ),
        sa.Column(
            "fracture_classification",
            sa.String(128),
            nullable=True,
            comment="Fracture classification (e.g., Le Fort I, NOE type III)",
        ),
        sa.Column(
            "clinical_notes_hash",
            sa.String(64),
            nullable=True,
            comment="SHA-256 hash of clinical notes (notes stored in encrypted blob store)",
        ),
        # Surgical planning context
        sa.Column(
            "planned_procedure",
            sa.String(256),
            nullable=True,
            comment="Planned surgical procedure description",
        ),
        sa.Column(
            "target_surgery_date",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
            comment="Target date for surgical procedure",
        ),
        # Celery job tracking
        sa.Column(
            "current_task_id",
            sa.String(64),
            nullable=True,
            comment="Active Celery task ID for async operations",
        ),
        sa.Column(
            "last_error",
            sa.Text,
            nullable=True,
            comment="Error message from last failed operation",
        ),
        # Surgeon and team
        sa.Column(
            "surgeon_id",
            sa.String(64),
            nullable=True,
            comment="Assigned surgeon user ID",
        ),
        sa.Column(
            "reviewer_id",
            sa.String(64),
            nullable=True,
            comment="Case reviewer user ID",
        ),
        sa.Column(
            "team_ids",
            JSONB,
            nullable=True,
            comment="List of user IDs with access to this case",
        ),
        # Audit timestamps
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
        ),
        sa.Column(
            "approved_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
            comment="Timestamp when surgeon approved the plan",
        ),
    )
    op.create_index("ix_surgical_cases_patient_id", "surgical_cases", ["patient_id"])
    op.create_index("ix_surgical_cases_study_id", "surgical_cases", ["study_id"])
    op.create_index("ix_surgical_cases_surgeon_id", "surgical_cases", ["surgeon_id"])
    op.create_index("ix_surgical_cases_status", "surgical_cases", ["status"])
    op.create_index("ix_surgical_cases_case_type", "surgical_cases", ["case_type"])
    op.create_index("ix_surgical_cases_created_at", "surgical_cases", ["created_at"])
    # Composite index for common filter pattern (status + surgeon)
    op.create_index(
        "ix_surgical_cases_status_surgeon",
        "surgical_cases",
        ["status", "surgeon_id"],
    )

    # ── segmentation_results ──────────────────────────────────────────────────
    # ML bone segmentation outputs. Multiple results per case allowed.
    op.create_table(
        "segmentation_results",
        _uuid_pk(),
        sa.Column(
            "case_id",
            UUID(as_uuid=False),
            sa.ForeignKey("surgical_cases.id", ondelete="CASCADE", name="fk_segmentation_results_case_id"),
            nullable=False,
        ),
        # Model provenance
        sa.Column(
            "model_name",
            sa.String(64),
            nullable=False,
            comment="Name of model used: 'totalsegmentator', 'cmf_custom_v1', etc.",
        ),
        sa.Column(
            "model_version",
            sa.String(32),
            nullable=False,
            comment="Semantic version of the model weights",
        ),
        sa.Column(
            "model_checkpoint",
            sa.String(128),
            nullable=True,
            comment="Model checkpoint hash or filename",
        ),
        # Segmentation outputs
        sa.Column(
            "structure_labels",
            JSONB,
            nullable=True,
            comment=(
                "Dict mapping label name to integer mask value. "
                "E.g.: {'mandible': 1, 'maxilla': 2, 'zygoma_L': 3}"
            ),
        ),
        sa.Column(
            "mask_storage_path",
            sa.Text,
            nullable=True,
            comment="Path to NIfTI segmentation mask file (.nii.gz)",
        ),
        sa.Column(
            "mesh_storage_paths",
            JSONB,
            nullable=True,
            comment=(
                "Dict mapping structure name to mesh file paths. "
                "E.g.: {'mandible': {'glb': '/path/mandible.glb', 'stl': '/path/mandible.stl'}}"
            ),
        ),
        # Confidence scores
        sa.Column(
            "confidence_scores",
            JSONB,
            nullable=True,
            comment="Per-structure confidence scores [0.0, 1.0]. E.g.: {'mandible': 0.97}",
        ),
        sa.Column(
            "overall_confidence",
            sa.Float,
            nullable=True,
            comment="Aggregate confidence score across all structures",
        ),
        # Performance metrics
        sa.Column(
            "inference_time_ms",
            sa.Integer,
            nullable=True,
            comment="GPU inference time in milliseconds",
        ),
        sa.Column(
            "total_pipeline_time_ms",
            sa.Integer,
            nullable=True,
            comment="End-to-end pipeline time including pre/post-processing",
        ),
        sa.Column(
            "gpu_device",
            sa.String(32),
            nullable=True,
            comment="GPU device used for inference (e.g., 'cuda:0')",
        ),
        # Volume statistics
        sa.Column(
            "volume_stats",
            JSONB,
            nullable=True,
            comment=(
                "Per-structure volumetric statistics. "
                "E.g.: {'mandible': {'volume_cc': 45.2, 'surface_area_mm2': 3200.5}}"
            ),
        ),
        # Dental segmentation
        sa.Column(
            "dental_mask_path",
            sa.Text,
            nullable=True,
            comment="Path to per-tooth segmentation mask (FDI numbering)",
        ),
        sa.Column(
            "dental_mesh_paths",
            JSONB,
            nullable=True,
            comment="Per-tooth mesh paths keyed by FDI tooth number",
        ),
        # Fragment identification
        sa.Column(
            "fragment_count",
            sa.Integer,
            nullable=True,
            comment="Number of identified bone fragments (trauma cases)",
        ),
        sa.Column(
            "fragment_masks_path",
            sa.Text,
            nullable=True,
            comment="Path to labeled fragment instance mask",
        ),
        # Pipeline status
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'pending'"),
            comment="Pipeline status: pending, running, complete, failed",
        ),
        sa.Column(
            "error_message",
            sa.Text,
            nullable=True,
            comment="Error details if status=failed",
        ),
        sa.Column(
            "celery_task_id",
            sa.String(64),
            nullable=True,
            comment="Celery task ID for async tracking",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
        ),
        sa.Column(
            "completed_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )
    op.create_index("ix_segmentation_results_case_id", "segmentation_results", ["case_id"])
    op.create_index("ix_segmentation_results_status", "segmentation_results", ["status"])
    op.create_index("ix_segmentation_results_created_at", "segmentation_results", ["created_at"])
    op.create_index(
        "ix_segmentation_results_case_status",
        "segmentation_results",
        ["case_id", "status"],
    )

    # ── reduction_plans ───────────────────────────────────────────────────────
    # Surgical fracture reduction plans. Immutable per version.
    op.create_table(
        "reduction_plans",
        _uuid_pk(),
        sa.Column(
            "case_id",
            UUID(as_uuid=False),
            sa.ForeignKey("surgical_cases.id", ondelete="CASCADE", name="fk_reduction_plans_case_id"),
            nullable=False,
        ),
        sa.Column(
            "plan_version",
            sa.Integer,
            nullable=False,
            server_default=sa.text("1"),
            comment="Monotonically increasing plan version for this case",
        ),
        # ML model provenance
        sa.Column(
            "model_name",
            sa.String(64),
            nullable=True,
            comment="Reduction model used: 'baseline_icp', 'learned_v1', etc.",
        ),
        sa.Column(
            "model_version",
            sa.String(32),
            nullable=True,
        ),
        # Fragment geometry and transforms
        sa.Column(
            "fragments",
            JSONB,
            nullable=True,
            comment=(
                "Dict of fragment definitions keyed by fragment ID. "
                "Value = {label: int, mesh_path: str, volume_cc: float, centroid: [x,y,z]}"
            ),
        ),
        sa.Column(
            "transformations",
            JSONB,
            nullable=True,
            comment=(
                "Dict of planned SE(3) rigid body transforms per fragment. "
                "Value = {rotation_matrix: [[3x3]], translation_mm: [x,y,z], "
                "confidence: float, alternative_transforms: [...]}"
            ),
        ),
        # Constraints
        sa.Column(
            "dental_constraints",
            JSONB,
            nullable=True,
            comment=(
                "Occlusal and skeletal constraints used during optimization. "
                "E.g.: {target_overjet_mm: 2.0, target_overbite_mm: 3.0}"
            ),
        ),
        sa.Column(
            "skeletal_constraints",
            JSONB,
            nullable=True,
            comment=(
                "Skeletal symmetry constraints. "
                "E.g.: {bilateral_symmetry_tolerance_mm: 2.0}"
            ),
        ),
        # Computed outcomes
        sa.Column(
            "occlusal_metrics",
            JSONB,
            nullable=True,
            comment=(
                "Post-reduction occlusal metrics. "
                "E.g.: {overjet_mm: 1.8, overbite_mm: 2.9, molar_relationship: 'Class_I'}"
            ),
        ),
        sa.Column(
            "symmetry_metrics",
            JSONB,
            nullable=True,
            comment=(
                "Skeletal symmetry assessment. "
                "E.g.: {facial_midline_deviation_mm: 0.8}"
            ),
        ),
        # Plan quality
        sa.Column(
            "confidence_score",
            sa.Float,
            nullable=True,
            comment="ML model confidence in the proposed reduction [0.0, 1.0]",
        ),
        sa.Column(
            "validation_passed",
            sa.Boolean,
            nullable=True,
            comment="Whether automated validation checks passed",
        ),
        sa.Column(
            "validation_warnings",
            JSONB,
            nullable=True,
            comment="List of validation warning messages",
        ),
        # Surgeon review
        sa.Column(
            "surgeon_approved",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("FALSE"),
            comment="Whether the attending surgeon has approved this plan",
        ),
        sa.Column(
            "surgeon_notes",
            sa.Text,
            nullable=True,
            comment="Surgeon's comments on this plan version",
        ),
        sa.Column(
            "surgeon_edits",
            JSONB,
            nullable=True,
            comment=(
                "History of surgeon manual adjustments applied to this plan version. "
                "Each entry: {fragment_id, original_transform, edited_transform, timestamp}"
            ),
        ),
        sa.Column(
            "parent_plan_id",
            UUID(as_uuid=False),
            nullable=True,
            comment="ID of the plan version this was derived from (surgeon edit lineage)",
        ),
        sa.Column(
            "is_ml_generated",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("TRUE"),
            comment="True if primarily ML-generated; False if manually created",
        ),
        # Pipeline status
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'draft'"),
            comment="Plan status: draft, validated, surgeon_reviewed, approved, archived",
        ),
        sa.Column(
            "generation_time_ms",
            sa.Integer,
            nullable=True,
            comment="Time to generate the plan in milliseconds",
        ),
        # Timestamps
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
        ),
        sa.Column(
            "approved_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column("approved_by", sa.String(64), nullable=True),
    )
    op.create_index("ix_reduction_plans_case_id", "reduction_plans", ["case_id"])
    op.create_index("ix_reduction_plans_plan_version", "reduction_plans", ["plan_version"])
    op.create_index("ix_reduction_plans_status", "reduction_plans", ["status"])
    op.create_index(
        "ix_reduction_plans_case_version",
        "reduction_plans",
        ["case_id", "plan_version"],
    )

    # ── audit_logs ────────────────────────────────────────────────────────────
    # Immutable HIPAA audit trail. Never update or delete records.
    op.create_table(
        "audit_logs",
        _uuid_pk(),
        # Actor
        sa.Column(
            "user_id",
            sa.String(64),
            nullable=False,
            comment="ID of the user performing the action",
        ),
        sa.Column(
            "user_role",
            sa.String(32),
            nullable=False,
            comment="User role at time of action",
        ),
        sa.Column(
            "session_id",
            sa.String(64),
            nullable=False,
            comment="JWT JTI or session identifier",
        ),
        # Action
        sa.Column(
            "action",
            sa.String(64),
            nullable=False,
            comment="Action type: READ, CREATE, UPDATE, DELETE, EXPORT, APPROVE, LOGIN, etc.",
        ),
        sa.Column(
            "action_category",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'phi_access'"),
            comment="HIPAA category: phi_access, authentication, admin, system",
        ),
        # Target resource
        sa.Column(
            "resource_type",
            sa.String(64),
            nullable=False,
            comment="Resource type: patient, study, case, plan, segmentation, mesh",
        ),
        sa.Column(
            "resource_id",
            sa.String(64),
            nullable=False,
            comment="Unique identifier of the accessed resource",
        ),
        # Change summary (non-PHI only)
        sa.Column(
            "changes_json",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment=(
                "Summary of changes (for UPDATE actions). "
                "Must not contain PHI values — only field names and change type."
            ),
        ),
        # Request context
        sa.Column(
            "ip_address",
            sa.String(45),
            nullable=False,
            comment="Client IP address (IPv4 or IPv6)",
        ),
        sa.Column(
            "user_agent",
            sa.String(256),
            nullable=False,
            server_default=sa.text("''"),
            comment="HTTP User-Agent string",
        ),
        sa.Column(
            "request_id",
            sa.String(64),
            nullable=False,
            comment="HTTP request ID for correlation",
        ),
        sa.Column(
            "correlation_id",
            sa.String(64),
            nullable=False,
            server_default=sa.text("''"),
            comment="Distributed tracing correlation ID",
        ),
        # Outcome
        sa.Column(
            "success",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("TRUE"),
            comment="Whether the action succeeded",
        ),
        sa.Column(
            "failure_reason",
            sa.String(256),
            nullable=False,
            server_default=sa.text("''"),
            comment="Reason for failure (if success=False)",
        ),
        sa.Column(
            "http_status_code",
            sa.Integer,
            nullable=False,
            server_default=sa.text("200"),
            comment="HTTP response status code",
        ),
        sa.Column(
            "duration_ms",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
            comment="Request processing duration in milliseconds",
        ),
        # Timestamp
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=_now_tz(),
            comment="UTC timestamp of the audit event",
        ),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_resource", "audit_logs", ["resource_type", "resource_id"])
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_session_id", "audit_logs", ["session_id"])
    # Composite index for HIPAA compliance reporting queries
    op.create_index(
        "ix_audit_logs_user_resource_time",
        "audit_logs",
        ["user_id", "resource_type", "timestamp"],
    )
    # Partial index for failed access attempts (security monitoring)
    op.create_index(
        "ix_audit_logs_failures",
        "audit_logs",
        ["user_id", "timestamp"],
        postgresql_where=sa.text("success = FALSE"),
    )

    # ── updated_at trigger ────────────────────────────────────────────────────
    # Automatically update the updated_at column on every row modification for
    # tables that carry that column.
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    for table_name in ("patients", "imaging_studies", "surgical_cases"):
        op.execute(f"""
            CREATE TRIGGER trg_{table_name}_updated_at
            BEFORE UPDATE ON {table_name}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)


# ── downgrade ─────────────────────────────────────────────────────────────────

def downgrade() -> None:
    """Drop all tables and supporting objects in reverse dependency order."""

    # Drop triggers before dropping tables
    for table_name in ("surgical_cases", "imaging_studies", "patients"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_updated_at ON {table_name}")

    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE")

    # Drop tables in reverse FK dependency order
    op.drop_table("audit_logs")
    op.drop_table("reduction_plans")
    op.drop_table("segmentation_results")
    op.drop_table("surgical_cases")
    op.drop_table("imaging_studies")
    op.drop_table("patients")

    # Drop extension last (only if no other objects depend on it)
    op.execute("DROP EXTENSION IF EXISTS pgcrypto")
