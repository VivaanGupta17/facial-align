"""
Custom exception hierarchy for Facial Align backend.
All domain exceptions map to appropriate HTTP status codes.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException, status


# ─── Base exceptions ──────────────────────────────────────────────────────────


class FacialAlignError(Exception):
    """
    Base exception for all Facial Align domain errors.
    Carry a human-readable message, an error code, and optional context.
    """

    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.error_code
        self.context = context or {}
        self.cause = cause

    def to_http_exception(self) -> HTTPException:
        """Convert to a FastAPI HTTPException for the error handler."""
        detail: dict[str, Any] = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.context:
            detail["context"] = self.context
        return HTTPException(status_code=self.http_status, detail=detail)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code={self.error_code!r}, "
            f"message={self.message!r}, context={self.context!r})"
        )


# ─── Not Found ────────────────────────────────────────────────────────────────


class NotFoundError(FacialAlignError):
    """Resource not found."""

    http_status = status.HTTP_404_NOT_FOUND
    error_code = "NOT_FOUND"


class PatientNotFoundError(NotFoundError):
    error_code = "PATIENT_NOT_FOUND"


class StudyNotFoundError(NotFoundError):
    error_code = "STUDY_NOT_FOUND"


class CaseNotFoundError(NotFoundError):
    error_code = "CASE_NOT_FOUND"


class PlanNotFoundError(NotFoundError):
    error_code = "PLAN_NOT_FOUND"


class SegmentationNotFoundError(NotFoundError):
    error_code = "SEGMENTATION_NOT_FOUND"


# ─── Validation errors ────────────────────────────────────────────────────────


class ValidationError(FacialAlignError):
    """Input validation failure."""

    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "VALIDATION_ERROR"


class InvalidTransformError(ValidationError):
    """Invalid 3D transformation matrix."""
    error_code = "INVALID_TRANSFORM"


class InvalidOcclusalConstraintError(ValidationError):
    """Occlusal constraints are geometrically infeasible."""
    error_code = "INVALID_OCCLUSAL_CONSTRAINT"


# ─── Conflict errors ──────────────────────────────────────────────────────────


class ConflictError(FacialAlignError):
    """Resource state conflict (e.g., duplicate study, invalid status transition)."""

    http_status = status.HTTP_409_CONFLICT
    error_code = "CONFLICT"


class DuplicateStudyError(ConflictError):
    """Study UID already exists."""
    error_code = "DUPLICATE_STUDY"


class InvalidStatusTransitionError(ConflictError):
    """Attempted an invalid case status transition."""
    error_code = "INVALID_STATUS_TRANSITION"


# ─── Authorization errors ─────────────────────────────────────────────────────


class AuthorizationError(FacialAlignError):
    """Authorization failure."""

    http_status = status.HTTP_403_FORBIDDEN
    error_code = "FORBIDDEN"


class InsufficientPermissionsError(AuthorizationError):
    error_code = "INSUFFICIENT_PERMISSIONS"


# ─── DICOM errors ─────────────────────────────────────────────────────────────


class DicomError(FacialAlignError):
    """Base class for DICOM processing errors."""

    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "DICOM_ERROR"


class DicomParseError(DicomError):
    """Failed to parse DICOM file or metadata."""
    error_code = "DICOM_PARSE_ERROR"


class DicomValidationError(DicomError):
    """DICOM data fails quality or completeness checks."""
    error_code = "DICOM_VALIDATION_ERROR"


class DicomDeidentificationError(DicomError):
    """Failed to de-identify DICOM study."""
    error_code = "DICOM_DEIDENTIFICATION_ERROR"


class InsufficientCoverageError(DicomValidationError):
    """CT volume does not cover the required anatomical region."""
    error_code = "INSUFFICIENT_ANATOMICAL_COVERAGE"


class SliceThicknessError(DicomValidationError):
    """CT slice thickness exceeds acceptable limit for surgical planning."""
    error_code = "SLICE_THICKNESS_TOO_LARGE"


# ─── Segmentation errors ──────────────────────────────────────────────────────


class SegmentationError(FacialAlignError):
    """Base class for segmentation pipeline errors."""

    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "SEGMENTATION_ERROR"


class ModelLoadError(SegmentationError):
    """Failed to load ML model weights."""
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "MODEL_LOAD_ERROR"


class InferenceError(SegmentationError):
    """ML inference failed."""
    error_code = "INFERENCE_ERROR"


class ModelNotAvailableError(SegmentationError):
    """Requested model not available in registry."""
    http_status = status.HTTP_404_NOT_FOUND
    error_code = "MODEL_NOT_AVAILABLE"


class PostProcessingError(SegmentationError):
    """Segmentation post-processing failed."""
    error_code = "SEGMENTATION_POSTPROCESSING_ERROR"


class LowConfidenceSegmentationError(SegmentationError):
    """Segmentation confidence below acceptable threshold."""
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "LOW_CONFIDENCE_SEGMENTATION"


# ─── Mesh errors ──────────────────────────────────────────────────────────────


class MeshError(FacialAlignError):
    """Base class for mesh processing errors."""

    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "MESH_ERROR"


class MeshExtractionError(MeshError):
    """Failed to extract surface mesh from mask."""
    error_code = "MESH_EXTRACTION_ERROR"


class MeshSimplificationError(MeshError):
    """Mesh simplification/decimation failed."""
    error_code = "MESH_SIMPLIFICATION_ERROR"


class EmptyMaskError(MeshError):
    """Mask is empty; cannot extract mesh."""
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "EMPTY_MASK"


class MeshQualityError(MeshError):
    """Extracted mesh does not meet quality thresholds."""
    error_code = "MESH_QUALITY_ERROR"


# ─── Registration errors ──────────────────────────────────────────────────────


class RegistrationError(FacialAlignError):
    """Base class for registration errors."""

    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "REGISTRATION_ERROR"


class ICPConvergenceError(RegistrationError):
    """ICP registration did not converge."""
    error_code = "ICP_CONVERGENCE_ERROR"


class RegistrationDivergenceError(RegistrationError):
    """Registration result has unacceptably high error."""
    error_code = "REGISTRATION_DIVERGENCE"


class InsufficientOverlapError(RegistrationError):
    """Source and target meshes have insufficient overlap for registration."""
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "INSUFFICIENT_OVERLAP"


# ─── Reduction planning errors ────────────────────────────────────────────────


class ReductionPlanningError(FacialAlignError):
    """Base class for fracture reduction planning errors."""

    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "REDUCTION_PLANNING_ERROR"


class FractureFragmentError(ReductionPlanningError):
    """Error processing fracture fragment geometry."""
    error_code = "FRAGMENT_PROCESSING_ERROR"


class ReductionConstraintViolation(ReductionPlanningError):
    """Reduction plan violates anatomical or occlusal constraints."""
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "CONSTRAINT_VIOLATION"


class SymmetryThresholdError(ReductionPlanningError):
    """Planned reduction exceeds skeletal symmetry threshold."""
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "SYMMETRY_THRESHOLD_EXCEEDED"


# ─── Occlusion errors ─────────────────────────────────────────────────────────


class OcclusionError(FacialAlignError):
    """Base class for occlusion analysis errors."""

    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "OCCLUSION_ERROR"


class DentalArchError(OcclusionError):
    """Error processing dental arch geometry."""
    error_code = "DENTAL_ARCH_ERROR"


class OcclusionMetricError(OcclusionError):
    """Failed to compute occlusal metric."""
    error_code = "OCCLUSION_METRIC_ERROR"


# ─── Storage errors ───────────────────────────────────────────────────────────


class StorageError(FacialAlignError):
    """Base class for file storage errors."""

    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "STORAGE_ERROR"


class FileNotFoundStorageError(StorageError):
    http_status = status.HTTP_404_NOT_FOUND
    error_code = "FILE_NOT_FOUND"


class StorageQuotaExceededError(StorageError):
    http_status = status.HTTP_507_INSUFFICIENT_STORAGE
    error_code = "STORAGE_QUOTA_EXCEEDED"


# ─── Task/job errors ──────────────────────────────────────────────────────────


class TaskError(FacialAlignError):
    """Async task/job related error."""

    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "TASK_ERROR"


class TaskNotFoundError(TaskError):
    http_status = status.HTTP_404_NOT_FOUND
    error_code = "TASK_NOT_FOUND"


class TaskTimeoutError(TaskError):
    http_status = status.HTTP_504_GATEWAY_TIMEOUT
    error_code = "TASK_TIMEOUT"


# ─── HTTP status mapping registry ────────────────────────────────────────────


EXCEPTION_STATUS_MAP: dict[type[FacialAlignError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ConflictError: status.HTTP_409_CONFLICT,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    DicomError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    SegmentationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ModelNotAvailableError: status.HTTP_404_NOT_FOUND,
    MeshError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    RegistrationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ReductionPlanningError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    OcclusionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    StorageError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    TaskError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}
