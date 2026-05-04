"""
Unit tests for the custom exception hierarchy in apps/backend/app/core/exceptions.py.

Tests cover:
- All domain exceptions inherit from FacialAlignError
- to_http_exception returns correct HTTP status codes
- Error codes match expected values
- Exception repr format
- Context propagation
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from app.core.exceptions import (
    AuthorizationError,
    CaseNotFoundError,
    ConflictError,
    DentalArchError,
    DicomDeidentificationError,
    DicomError,
    DicomParseError,
    DicomValidationError,
    DuplicateStudyError,
    EmptyMaskError,
    FacialAlignError,
    FractureFragmentError,
    ICPConvergenceError,
    InferenceError,
    InsufficientCoverageError,
    InsufficientOverlapError,
    InsufficientPermissionsError,
    InvalidOcclusalConstraintError,
    InvalidStatusTransitionError,
    InvalidTransformError,
    LowConfidenceSegmentationError,
    MeshError,
    MeshExtractionError,
    MeshQualityError,
    MeshSimplificationError,
    ModelLoadError,
    ModelNotAvailableError,
    NotFoundError,
    OcclusionError,
    OcclusionMetricError,
    PatientNotFoundError,
    PlanNotFoundError,
    PostProcessingError,
    ReductionConstraintViolation,
    ReductionPlanningError,
    RegistrationDivergenceError,
    RegistrationError,
    SegmentationError,
    SegmentationNotFoundError,
    SliceThicknessError,
    StorageError,
    StorageQuotaExceededError,
    StudyNotFoundError,
    SymmetryThresholdError,
    TaskError,
    TaskNotFoundError,
    TaskTimeoutError,
    ValidationError,
)


# ─── Inheritance tests ────────────────────────────────────────────────────────


class TestExceptionInheritance:
    """All domain exceptions must inherit from FacialAlignError."""

    @pytest.mark.parametrize("exc_class", [
        # Not-found hierarchy
        NotFoundError,
        PatientNotFoundError,
        StudyNotFoundError,
        CaseNotFoundError,
        PlanNotFoundError,
        SegmentationNotFoundError,
        # Validation hierarchy
        ValidationError,
        InvalidTransformError,
        InvalidOcclusalConstraintError,
        # Conflict hierarchy
        ConflictError,
        DuplicateStudyError,
        InvalidStatusTransitionError,
        # Authorization hierarchy
        AuthorizationError,
        InsufficientPermissionsError,
        # DICOM hierarchy
        DicomError,
        DicomParseError,
        DicomValidationError,
        DicomDeidentificationError,
        InsufficientCoverageError,
        SliceThicknessError,
        # Segmentation hierarchy
        SegmentationError,
        ModelLoadError,
        InferenceError,
        ModelNotAvailableError,
        PostProcessingError,
        LowConfidenceSegmentationError,
        # Mesh hierarchy
        MeshError,
        MeshExtractionError,
        MeshSimplificationError,
        EmptyMaskError,
        MeshQualityError,
        # Registration hierarchy
        RegistrationError,
        ICPConvergenceError,
        RegistrationDivergenceError,
        InsufficientOverlapError,
        # Reduction hierarchy
        ReductionPlanningError,
        FractureFragmentError,
        ReductionConstraintViolation,
        SymmetryThresholdError,
        # Occlusion hierarchy
        OcclusionError,
        DentalArchError,
        OcclusionMetricError,
        # Storage hierarchy
        StorageError,
        # Task hierarchy
        TaskError,
        TaskNotFoundError,
        TaskTimeoutError,
    ])
    def test_inherits_from_facial_align_error(self, exc_class):
        assert issubclass(exc_class, FacialAlignError), (
            f"{exc_class.__name__} must inherit from FacialAlignError"
        )

    def test_not_found_subclasses_inherit_from_not_found(self):
        for cls in [PatientNotFoundError, StudyNotFoundError, CaseNotFoundError,
                    PlanNotFoundError, SegmentationNotFoundError]:
            assert issubclass(cls, NotFoundError)

    def test_dicom_subclasses_inherit_from_dicom_error(self):
        for cls in [DicomParseError, DicomValidationError, DicomDeidentificationError,
                    InsufficientCoverageError, SliceThicknessError]:
            assert issubclass(cls, DicomError)

    def test_mesh_subclasses_inherit_from_mesh_error(self):
        for cls in [MeshExtractionError, MeshSimplificationError, EmptyMaskError, MeshQualityError]:
            assert issubclass(cls, MeshError)

    def test_registration_subclasses_inherit_from_registration_error(self):
        for cls in [ICPConvergenceError, RegistrationDivergenceError, InsufficientOverlapError]:
            assert issubclass(cls, RegistrationError)

    def test_occlusion_subclasses_inherit_from_occlusion_error(self):
        for cls in [DentalArchError, OcclusionMetricError]:
            assert issubclass(cls, OcclusionError)

    def test_reduction_subclasses_inherit_from_reduction_planning_error(self):
        for cls in [FractureFragmentError, ReductionConstraintViolation, SymmetryThresholdError]:
            assert issubclass(cls, ReductionPlanningError)


# ─── HTTP status code tests ───────────────────────────────────────────────────


class TestHttpStatusCodes:
    """to_http_exception returns correct HTTP status codes for each exception."""

    @pytest.mark.parametrize("exc_class,expected_status", [
        # 404 Not Found
        (NotFoundError, status.HTTP_404_NOT_FOUND),
        (PatientNotFoundError, status.HTTP_404_NOT_FOUND),
        (StudyNotFoundError, status.HTTP_404_NOT_FOUND),
        (CaseNotFoundError, status.HTTP_404_NOT_FOUND),
        (PlanNotFoundError, status.HTTP_404_NOT_FOUND),
        (SegmentationNotFoundError, status.HTTP_404_NOT_FOUND),
        (ModelNotAvailableError, status.HTTP_404_NOT_FOUND),
        # 422 Unprocessable Entity
        (ValidationError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (InvalidTransformError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (InvalidOcclusalConstraintError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (DicomError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (DicomParseError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (DicomValidationError, status.HTTP_400_BAD_REQUEST),
        (EmptyMaskError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (InsufficientOverlapError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (ReductionConstraintViolation, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (SymmetryThresholdError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (LowConfidenceSegmentationError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        # 409 Conflict
        (ConflictError, status.HTTP_409_CONFLICT),
        (DuplicateStudyError, status.HTTP_409_CONFLICT),
        (InvalidStatusTransitionError, status.HTTP_409_CONFLICT),
        # 403 Forbidden
        (AuthorizationError, status.HTTP_403_FORBIDDEN),
        (InsufficientPermissionsError, status.HTTP_403_FORBIDDEN),
        # 500 Internal Server Error
        (FacialAlignError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (SegmentationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (InferenceError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (PostProcessingError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (MeshError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (MeshExtractionError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (RegistrationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (ICPConvergenceError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (ReductionPlanningError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (OcclusionError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (DentalArchError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (OcclusionMetricError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (StorageError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        (TaskError, status.HTTP_500_INTERNAL_SERVER_ERROR),
        # 503 Service Unavailable
        (ModelLoadError, status.HTTP_503_SERVICE_UNAVAILABLE),
        # 504 Gateway Timeout
        (TaskTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT),
        # 507 Insufficient Storage
        (StorageQuotaExceededError, status.HTTP_507_INSUFFICIENT_STORAGE),
    ])
    def test_http_status_code(self, exc_class, expected_status):
        exc = exc_class("test message")
        http_exc = exc.to_http_exception()
        assert http_exc.status_code == expected_status, (
            f"{exc_class.__name__}: expected {expected_status}, got {http_exc.status_code}"
        )

    def test_to_http_exception_returns_http_exception(self):
        exc = FacialAlignError("test")
        http_exc = exc.to_http_exception()
        assert isinstance(http_exc, HTTPException)

    def test_detail_contains_error_code(self):
        exc = DicomParseError("Bad DICOM")
        http_exc = exc.to_http_exception()
        assert http_exc.detail["error"] == "DICOM_PARSE_ERROR"

    def test_detail_contains_message(self):
        exc = NotFoundError("Resource not found")
        http_exc = exc.to_http_exception()
        assert http_exc.detail["message"] == "Resource not found"

    def test_detail_includes_context_when_provided(self):
        exc = RegistrationError("ICP failed", context={"iteration": 200})
        http_exc = exc.to_http_exception()
        assert "context" in http_exc.detail
        assert http_exc.detail["context"]["iteration"] == 200

    def test_detail_omits_context_when_empty(self):
        exc = FacialAlignError("test message")
        http_exc = exc.to_http_exception()
        assert "context" not in http_exc.detail


# ─── Error code tests ─────────────────────────────────────────────────────────


class TestErrorCodes:
    """Error codes match expected values for all exception classes."""

    @pytest.mark.parametrize("exc_class,expected_code", [
        (FacialAlignError, "INTERNAL_ERROR"),
        (NotFoundError, "NOT_FOUND"),
        (PatientNotFoundError, "PATIENT_NOT_FOUND"),
        (StudyNotFoundError, "STUDY_NOT_FOUND"),
        (CaseNotFoundError, "CASE_NOT_FOUND"),
        (PlanNotFoundError, "PLAN_NOT_FOUND"),
        (SegmentationNotFoundError, "SEGMENTATION_NOT_FOUND"),
        (ValidationError, "VALIDATION_ERROR"),
        (InvalidTransformError, "INVALID_TRANSFORM"),
        (InvalidOcclusalConstraintError, "INVALID_OCCLUSAL_CONSTRAINT"),
        (ConflictError, "CONFLICT"),
        (DuplicateStudyError, "DUPLICATE_STUDY"),
        (InvalidStatusTransitionError, "INVALID_STATUS_TRANSITION"),
        (AuthorizationError, "FORBIDDEN"),
        (InsufficientPermissionsError, "INSUFFICIENT_PERMISSIONS"),
        (DicomError, "DICOM_ERROR"),
        (DicomParseError, "DICOM_PARSE_ERROR"),
        (DicomValidationError, "DICOM_VALIDATION_ERROR"),
        (DicomDeidentificationError, "DICOM_DEIDENTIFICATION_ERROR"),
        (InsufficientCoverageError, "INSUFFICIENT_ANATOMICAL_COVERAGE"),
        (SliceThicknessError, "SLICE_THICKNESS_TOO_LARGE"),
        (SegmentationError, "SEGMENTATION_ERROR"),
        (ModelLoadError, "MODEL_LOAD_ERROR"),
        (InferenceError, "INFERENCE_ERROR"),
        (ModelNotAvailableError, "MODEL_NOT_AVAILABLE"),
        (PostProcessingError, "SEGMENTATION_POSTPROCESSING_ERROR"),
        (LowConfidenceSegmentationError, "LOW_CONFIDENCE_SEGMENTATION"),
        (MeshError, "MESH_ERROR"),
        (MeshExtractionError, "MESH_EXTRACTION_ERROR"),
        (MeshSimplificationError, "MESH_SIMPLIFICATION_ERROR"),
        (EmptyMaskError, "EMPTY_MASK"),
        (MeshQualityError, "MESH_QUALITY_ERROR"),
        (RegistrationError, "REGISTRATION_ERROR"),
        (ICPConvergenceError, "ICP_CONVERGENCE_ERROR"),
        (RegistrationDivergenceError, "REGISTRATION_DIVERGENCE"),
        (InsufficientOverlapError, "INSUFFICIENT_OVERLAP"),
        (ReductionPlanningError, "REDUCTION_PLANNING_ERROR"),
        (FractureFragmentError, "FRAGMENT_PROCESSING_ERROR"),
        (ReductionConstraintViolation, "CONSTRAINT_VIOLATION"),
        (SymmetryThresholdError, "SYMMETRY_THRESHOLD_EXCEEDED"),
        (OcclusionError, "OCCLUSION_ERROR"),
        (DentalArchError, "DENTAL_ARCH_ERROR"),
        (OcclusionMetricError, "OCCLUSION_METRIC_ERROR"),
        (StorageError, "STORAGE_ERROR"),
        (TaskError, "TASK_ERROR"),
        (TaskNotFoundError, "TASK_NOT_FOUND"),
        (TaskTimeoutError, "TASK_TIMEOUT"),
    ])
    def test_error_code(self, exc_class, expected_code):
        exc = exc_class("test")
        assert exc.error_code == expected_code, (
            f"{exc_class.__name__}: expected code {expected_code!r}, got {exc.error_code!r}"
        )

    def test_custom_error_code_overrides_class_default(self):
        exc = FacialAlignError("test", error_code="CUSTOM_CODE")
        assert exc.error_code == "CUSTOM_CODE"


# ─── Exception repr tests ─────────────────────────────────────────────────────


class TestExceptionRepr:
    def test_repr_contains_class_name(self):
        exc = DicomParseError("File corrupted")
        r = repr(exc)
        assert "DicomParseError" in r

    def test_repr_contains_error_code(self):
        exc = DicomParseError("File corrupted")
        r = repr(exc)
        assert "DICOM_PARSE_ERROR" in r

    def test_repr_contains_message(self):
        exc = NotFoundError("Not found")
        r = repr(exc)
        assert "Not found" in r

    def test_repr_contains_context(self):
        exc = RegistrationError("ICP failed", context={"iteration": 100})
        r = repr(exc)
        assert "iteration" in r or "100" in r

    def test_repr_format_structure(self):
        """Repr should match ExcName(code=..., message=..., context=...) format."""
        exc = FacialAlignError("test message")
        r = repr(exc)
        assert "code=" in r
        assert "message=" in r
        assert "context=" in r

    def test_str_gives_message(self):
        """str() of an exception should return the message."""
        exc = FacialAlignError("my message")
        assert str(exc) == "my message"


# ─── Context propagation tests ────────────────────────────────────────────────


class TestContextPropagation:
    def test_context_stored(self):
        exc = RegistrationError("failed", context={"source_points": 5, "target_points": 200})
        assert exc.context["source_points"] == 5
        assert exc.context["target_points"] == 200

    def test_context_defaults_to_empty_dict(self):
        exc = FacialAlignError("test")
        assert exc.context == {}

    def test_cause_stored(self):
        original = ValueError("original cause")
        exc = InferenceError("inference failed", cause=original)
        assert exc.cause is original

    def test_cause_defaults_to_none(self):
        exc = FacialAlignError("test")
        assert exc.cause is None

    def test_context_appears_in_http_exception_detail(self):
        exc = InsufficientOverlapError(
            "Not enough overlap",
            context={"source_points": 3, "target_points": 500},
        )
        http_exc = exc.to_http_exception()
        assert http_exc.detail["context"]["source_points"] == 3

    def test_complex_context_preserved(self):
        ctx = {
            "model": "icp",
            "iterations": 200,
            "rms_error": 5.23,
            "nested": {"key": "value"},
        }
        exc = ICPConvergenceError("ICP diverged", context=ctx)
        assert exc.context == ctx

    def test_context_with_none_value(self):
        exc = FacialAlignError("test", context={"key": None})
        assert exc.context["key"] is None

    def test_message_accessible_via_message_attribute(self):
        exc = DentalArchError("Arch is empty")
        assert exc.message == "Arch is empty"

    def test_multiple_exception_instances_independent(self):
        """Two instances of same exception type should have independent contexts."""
        exc1 = RegistrationError("err1", context={"a": 1})
        exc2 = RegistrationError("err2", context={"b": 2})
        assert "a" in exc1.context
        assert "a" not in exc2.context
        assert "b" in exc2.context

    def test_exception_is_catchable_as_base_class(self):
        """Subclass exceptions can be caught using FacialAlignError."""
        with pytest.raises(FacialAlignError):
            raise DicomParseError("parse error")

    def test_exception_is_catchable_as_python_exception(self):
        """All FacialAlignError exceptions should be standard Python exceptions."""
        with pytest.raises(Exception):
            raise OcclusionMetricError("metric failed")

    def test_not_found_exceptions_catchable_as_not_found_error(self):
        with pytest.raises(NotFoundError):
            raise CaseNotFoundError("case not found")

    def test_registration_exceptions_catchable_as_registration_error(self):
        with pytest.raises(RegistrationError):
            raise ICPConvergenceError("ICP failed")


# ─── FacialAlignError base class tests ───────────────────────────────────────


class TestFacialAlignErrorBase:
    def test_instantiation_with_message_only(self):
        exc = FacialAlignError("Something went wrong")
        assert exc.message == "Something went wrong"

    def test_instantiation_with_all_args(self):
        cause = RuntimeError("root cause")
        exc = FacialAlignError(
            "Full error",
            error_code="FULL_ERROR",
            context={"detail": "extra info"},
            cause=cause,
        )
        assert exc.error_code == "FULL_ERROR"
        assert exc.context == {"detail": "extra info"}
        assert exc.cause is cause

    def test_http_status_defaults_to_500(self):
        exc = FacialAlignError("test")
        assert exc.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_class_error_code_is_internal_error(self):
        assert FacialAlignError.error_code == "INTERNAL_ERROR"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(FacialAlignError) as exc_info:
            raise FacialAlignError("raised!")
        assert exc_info.value.message == "raised!"
