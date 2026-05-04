"""
DICOM upload, ingestion, and study management endpoints.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import DicomValidationError, DuplicateStudyError, StudyNotFoundError
from app.core.logging import get_logger
from app.core.security import CurrentUser, audit_logger, get_current_user
from app.db.database import get_db_session
from app.models.patient import Patient
from app.models.study import ImagingStudy
from app.schemas.dicom import (
    DicomQualityReport,
    DicomSeriesDownloadResponse,
    DicomStudyListItem,
    DicomUploadResponse,
    StudyMetadata,
)

router = APIRouter(prefix="/dicom", tags=["dicom"])
settings = get_settings()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".zip", ".tar", ".gz"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB default chunk size


def _optional_str(value: object) -> Optional[str]:
    return value if isinstance(value, str) else None


@router.post(
    "/upload",
    response_model=DicomUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload DICOM study",
)
async def upload_dicom_study(
    files: List[UploadFile] = File(..., description="DICOM files or a ZIP archive"),
    patient_mrn: str = Form(..., description="Patient MRN (will be hashed)"),
    patient_age: Optional[int] = Form(default=None, ge=0, le=120),
    patient_sex: Optional[str] = Form(default=None),
    institution_code: Optional[str] = Form(default=None),
    case_type: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> DicomUploadResponse:
    """
    Upload DICOM files for a new imaging study.

    Accepts:
    - Multiple .dcm files
    - A single ZIP archive containing DICOM files
    - A DICOM directory structure in a ZIP

    Processing is asynchronous. Poll the returned job_id for status.
    """
    from app.core.security import hash_mrn
    from app.workers.tasks import run_dicom_ingestion_pipeline

    # Validate file types
    for f in files:
        if f.filename:
            suffix = Path(f.filename).suffix.lower()
            if suffix and suffix not in ALLOWED_EXTENSIONS:
                raise DicomValidationError(
                    f"File type not allowed: {suffix}. "
                    f"Accepted: {ALLOWED_EXTENSIONS}"
                )

    # Create upload staging directory
    upload_id = uuid.uuid4()
    upload_path = settings.storage.temp_path / "uploads" / str(upload_id)
    upload_path.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    total_size = 0
    for uploaded_file in files:
        dest = upload_path / (uploaded_file.filename or f"file_{uuid.uuid4()}.dcm")
        with dest.open("wb") as fh:
            chunk_size = 1024 * 1024  # 1 MB chunks
            while chunk := await uploaded_file.read(chunk_size):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    shutil.rmtree(upload_path, ignore_errors=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Upload exceeds maximum size of {MAX_UPLOAD_SIZE_BYTES / 1e9:.0f} GB",
                    )
                fh.write(chunk)

    logger.info(
        "dicom_upload_received",
        upload_id=str(upload_id),
        file_count=len(files),
        total_size_mb=round(total_size / 1e6, 2),
        upload_path=str(upload_path),
    )

    # Find or create patient
    mrn_hash = hash_mrn(patient_mrn)
    patient = (
        await db.execute(select(Patient).where(Patient.mrn_hash == mrn_hash))
    ).scalar_one_or_none()

    if not patient:
        patient = Patient(
            mrn_hash=mrn_hash,
            age_at_registration=patient_age,
            sex=patient_sex.upper() if patient_sex else None,
            institution_code=institution_code,
            created_by=current_user.user_id,
        )
        db.add(patient)
        await db.flush()
        logger.info("patient_created", patient_id=str(patient.id))

    # Create placeholder study record
    study = ImagingStudy(
        study_uid=f"PENDING-{upload_id}",  # Updated during ingestion
        patient_id=patient.id,
        modality="CT",  # Updated during ingestion
        storage_path=str(upload_path),
        ingestion_status="pending",
        uploaded_by=current_user.user_id,
    )
    db.add(study)
    await db.flush()

    # Dispatch ingestion pipeline
    task = run_dicom_ingestion_pipeline.delay(
        study_id=str(study.id),
        upload_path=str(upload_path),
        patient_id=str(patient.id),
        uploader_user_id=current_user.user_id,
    )

    study.current_task_id = task.id
    study.ingestion_status = "processing"

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="CREATE",
        resource_type="study",
        resource_id=str(study.id),
        ip_address="unknown",
        additional_context={"operation": "dicom_upload", "upload_id": str(upload_id)},
    )

    return DicomUploadResponse(
        upload_id=upload_id,
        study_id=study.id,
        patient_id=patient.id,
        ingestion_job_id=task.id,
        study_uid=study.study_uid,
        storage_path=str(upload_path),
        status="processing",
    )


@router.get(
    "/studies",
    response_model=List[DicomStudyListItem],
    summary="List imaging studies",
)
async def list_studies(
    patient_id: Optional[uuid.UUID] = Query(default=None),
    modality: Optional[str] = Query(default=None),
    ingestion_status: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[DicomStudyListItem]:
    """List imaging studies with optional filters."""
    stmt = select(ImagingStudy)
    if patient_id:
        stmt = stmt.where(ImagingStudy.patient_id == patient_id)
    if modality:
        stmt = stmt.where(ImagingStudy.modality == modality.upper())
    if ingestion_status:
        stmt = stmt.where(ImagingStudy.ingestion_status == ingestion_status)

    offset = (page - 1) * page_size
    stmt = stmt.order_by(ImagingStudy.created_at.desc()).offset(offset).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()

    return [
        DicomStudyListItem(
            id=s.id,
            study_uid=s.study_uid,
            patient_id=s.patient_id,
            modality=s.modality,
            acquisition_date=s.acquisition_date,
            series_count=s.series_count,
            ingestion_status=s.ingestion_status,
            quality_score=s.quality_score,
            created_at=s.created_at,
        )
        for s in rows
    ]


@router.get(
    "/studies/{study_id}",
    response_model=StudyMetadata,
    summary="Get study metadata",
)
async def get_study_metadata(
    study_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> StudyMetadata:
    """Get detailed metadata for an imaging study."""
    study = (
        await db.execute(select(ImagingStudy).where(ImagingStudy.id == study_id))
    ).scalar_one_or_none()
    if not study:
        raise StudyNotFoundError(f"Study {study_id} not found")

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="READ",
        resource_type="study",
        resource_id=str(study_id),
        ip_address="unknown",
    )

    meta = study.metadata_json or {}
    return StudyMetadata(
        study_uid=study.study_uid,
        modality=study.modality,
        acquisition_date=study.acquisition_date,
        body_part_examined=_optional_str(getattr(study, "body_part_examined", None)),
        study_description=meta.get("StudyDescription"),
        series=[
            {
                "series_instance_uid": meta.get("SeriesInstanceUID", "unknown"),
                "series_number": meta.get("SeriesNumber", 1),
                "series_description": meta.get("SeriesDescription", "Axial CT"),
                "modality": study.modality,
                "slice_count": study.slice_count or 0,
            }
        ] if study.series_count > 0 else [],
        series_count=study.series_count,
        total_slice_count=study.slice_count or 0,
        institution_name=meta.get("InstitutionName"),
        manufacturer=meta.get("Manufacturer"),
        manufacturer_model=meta.get("ManufacturerModelName"),
        software_versions=meta.get("SoftwareVersions"),
        additional_tags=meta,
    )


@router.get(
    "/studies/{study_id}/quality",
    response_model=DicomQualityReport,
    summary="Get CT quality assessment report",
)
async def get_quality_report(
    study_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> DicomQualityReport:
    """Get the automated CT quality assessment for a study."""
    study = (
        await db.execute(select(ImagingStudy).where(ImagingStudy.id == study_id))
    ).scalar_one_or_none()
    if not study:
        raise StudyNotFoundError(f"Study {study_id} not found")

    if study.quality_score is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quality report not yet available — study may still be processing",
        )

    flags = study.quality_flags or []
    warnings = [f for f in flags if "warning" in f.lower() or "thick" in f.lower()]
    errors = [f for f in flags if "error" in f.lower() or "insufficient" in f.lower()]

    return DicomQualityReport(
        study_id=study_id,
        quality_score=study.quality_score or 0.0,
        slice_thickness_mm=study.slice_thickness_mm,
        pixel_spacing_mm=study.pixel_spacing_mm,
        passed=bool(study.quality_score and study.quality_score >= 0.6),
        warnings=warnings,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Chunked upload endpoints — for 1 GB+ DICOM uploads
# ---------------------------------------------------------------------------


def _chunked_upload_dir(upload_id: uuid.UUID) -> Path:
    return settings.storage.temp_path / "chunked" / str(upload_id)


def _meta_path(upload_id: uuid.UUID) -> Path:
    return _chunked_upload_dir(upload_id) / "_meta.json"


def _read_meta(upload_id: uuid.UUID) -> dict:
    mp = _meta_path(upload_id)
    if not mp.exists():
        raise HTTPException(status_code=404, detail=f"Upload session {upload_id} not found")
    return json.loads(mp.read_text())


def _write_meta(upload_id: uuid.UUID, meta: dict) -> None:
    _meta_path(upload_id).write_text(json.dumps(meta))


@router.post(
    "/upload/init",
    status_code=status.HTTP_201_CREATED,
    summary="Initialise a chunked upload session",
)
async def init_chunked_upload(
    filename: str = Form(..., description="Original filename"),
    total_size: int = Form(..., ge=1, description="Total file size in bytes"),
    patient_mrn: str = Form(..., description="Patient MRN (will be hashed)"),
    patient_age: Optional[int] = Form(default=None, ge=0, le=120),
    patient_sex: Optional[str] = Form(default=None),
    institution_code: Optional[str] = Form(default=None),
    case_type: Optional[str] = Form(default=None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Create an upload session and return upload_id + chunk parameters."""
    if total_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_UPLOAD_SIZE_BYTES / 1e9:.0f} GB limit",
        )

    upload_id = uuid.uuid4()
    chunk_count = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    upload_dir = _chunked_upload_dir(upload_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "upload_id": str(upload_id),
        "filename": filename,
        "total_size": total_size,
        "chunk_size": CHUNK_SIZE,
        "chunk_count": chunk_count,
        "received_chunks": [],
        "patient_mrn": patient_mrn,
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "institution_code": institution_code,
        "case_type": case_type,
        "user_id": current_user.user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "uploading",
    }
    _write_meta(upload_id, meta)

    logger.info(
        "chunked_upload_init",
        upload_id=str(upload_id),
        filename=filename,
        total_size_mb=round(total_size / 1e6, 2),
        chunk_count=chunk_count,
    )

    return {
        "uploadId": str(upload_id),
        "chunkSize": CHUNK_SIZE,
        "chunkCount": chunk_count,
        "totalSize": total_size,
    }


@router.put(
    "/upload/{upload_id}/chunk/{chunk_index}",
    status_code=status.HTTP_200_OK,
    summary="Upload a single chunk",
)
async def upload_chunk(
    upload_id: uuid.UUID,
    chunk_index: int,
    chunk: UploadFile = File(..., description="Chunk data"),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Receive and persist a single chunk."""
    meta = _read_meta(upload_id)

    if chunk_index < 0 or chunk_index >= meta["chunk_count"]:
        raise HTTPException(400, detail=f"chunk_index must be 0..{meta['chunk_count'] - 1}")

    chunk_path = _chunked_upload_dir(upload_id) / f"chunk_{chunk_index:06d}"
    data = await chunk.read()
    chunk_path.write_bytes(data)

    received = set(meta["received_chunks"])
    received.add(chunk_index)
    meta["received_chunks"] = sorted(received)
    _write_meta(upload_id, meta)

    return {
        "chunkIndex": chunk_index,
        "receivedCount": len(meta["received_chunks"]),
        "totalChunks": meta["chunk_count"],
    }


@router.post(
    "/upload/{upload_id}/complete",
    response_model=DicomUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Reassemble chunks and start ingestion",
)
async def complete_chunked_upload(
    upload_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Validate all chunks received, reassemble, and trigger ingestion pipeline."""
    from app.core.security import hash_mrn
    from app.workers.tasks import run_dicom_ingestion_pipeline

    meta = _read_meta(upload_id)
    received = set(meta["received_chunks"])
    expected = set(range(meta["chunk_count"]))
    missing = expected - received
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Missing chunks: {sorted(missing)}",
        )

    # Reassemble file
    upload_dir = _chunked_upload_dir(upload_id)
    assembled_path = upload_dir / meta["filename"]
    with assembled_path.open("wb") as out:
        for i in range(meta["chunk_count"]):
            chunk_path = upload_dir / f"chunk_{i:06d}"
            out.write(chunk_path.read_bytes())
            chunk_path.unlink()

    # Move to standard upload path
    final_dir = settings.storage.temp_path / "uploads" / str(upload_id)
    final_dir.mkdir(parents=True, exist_ok=True)
    assembled_path.rename(final_dir / meta["filename"])

    # Clean up meta
    meta["status"] = "assembled"
    _write_meta(upload_id, meta)

    # Find or create patient
    mrn_hash = hash_mrn(meta["patient_mrn"])
    patient = (
        await db.execute(select(Patient).where(Patient.mrn_hash == mrn_hash))
    ).scalar_one_or_none()

    if not patient:
        patient = Patient(
            mrn_hash=mrn_hash,
            age_at_registration=meta.get("patient_age"),
            sex=meta["patient_sex"].upper() if meta.get("patient_sex") else None,
            institution_code=meta.get("institution_code"),
            created_by=current_user.user_id,
        )
        db.add(patient)
        await db.flush()

    # Create study record
    study = ImagingStudy(
        study_uid=f"PENDING-{upload_id}",
        patient_id=patient.id,
        modality="CT",
        storage_path=str(final_dir),
        ingestion_status="pending",
        uploaded_by=current_user.user_id,
    )
    db.add(study)
    await db.flush()

    # Dispatch ingestion
    task = run_dicom_ingestion_pipeline.delay(
        study_id=str(study.id),
        upload_path=str(final_dir),
        patient_id=str(patient.id),
        uploader_user_id=current_user.user_id,
    )
    study.current_task_id = task.id
    study.ingestion_status = "processing"

    audit_logger.log_phi_access(
        user_id=current_user.user_id,
        action="CREATE",
        resource_type="study",
        resource_id=str(study.id),
        ip_address="unknown",
        additional_context={"operation": "chunked_upload_complete", "upload_id": str(upload_id)},
    )

    return DicomUploadResponse(
        upload_id=upload_id,
        study_id=study.id,
        patient_id=patient.id,
        ingestion_job_id=task.id,
        study_uid=study.study_uid,
        storage_path=str(final_dir),
        status="processing",
    )


@router.get(
    "/upload/{upload_id}/status",
    summary="Get chunked upload status (for resume)",
)
async def get_chunked_upload_status(
    upload_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Return which chunks have been received so a client can resume."""
    meta = _read_meta(upload_id)
    return {
        "uploadId": str(upload_id),
        "status": meta["status"],
        "receivedChunks": meta["received_chunks"],
        "chunkCount": meta["chunk_count"],
        "chunkSize": meta["chunk_size"],
        "totalSize": meta["total_size"],
        "filename": meta["filename"],
    }
