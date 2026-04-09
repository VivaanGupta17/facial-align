"""
Shared pytest fixtures for Facial Align backend tests.
Provides mock data generators, async test helpers, and service stubs.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

# Optional imports for API test fixtures — gracefully degrade if not installed
try:
    import httpx
    from httpx._transports.asgi import ASGITransport
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

# ─── Async event loop ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ─── Test settings ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def test_settings():
    """Override settings for testing."""
    import os
    os.environ.update({
        "ENVIRONMENT": "test",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "facialign_test",
        "DB_USER": "facialign",
        "DB_PASSWORD": "test_password",
        "CELERY_TASK_ALWAYS_EAGER": "true",
        "MODEL_DEFAULT_DEVICE": "cpu",
        "LOG_LEVEL": "DEBUG",
    })
    from app.core.config import get_settings
    get_settings.cache_clear()
    return get_settings()


# ─── CT Volume fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def small_ct_volume() -> np.ndarray:
    """
    Small synthetic CT volume for testing (64x64x64).
    Contains a simple bone-like structure (high HU values) in the center.
    """
    volume = np.full((64, 64, 64), -1000.0, dtype=np.float32)  # Air background
    # Add bone-like region in center
    volume[20:44, 20:44, 20:44] = 700.0  # Cortical bone HU range
    # Add cancellous bone interior
    volume[24:40, 24:40, 24:40] = 300.0
    return volume


@pytest.fixture
def ct_spacing() -> tuple:
    """Standard CT voxel spacing for testing."""
    return (0.5, 0.5, 0.5)  # 0.5mm isotropic


@pytest.fixture
def large_ct_volume() -> np.ndarray:
    """
    Larger synthetic CT volume for pipeline testing (128x128x128).
    Simulates a craniofacial CT with multiple bone structures.
    """
    volume = np.full((128, 128, 128), -1000.0, dtype=np.float32)
    # Skull base region
    volume[10:30, 30:98, 30:98] = 800.0
    # Bilateral zygomas
    volume[40:60, 20:40, 20:50] = 700.0   # Left zygoma
    volume[40:60, 88:108, 20:50] = 700.0  # Right zygoma
    # Maxilla
    volume[50:75, 40:88, 30:60] = 650.0
    # Mandible
    volume[75:100, 30:98, 20:55] = 700.0
    # Teeth (very high HU)
    volume[80:95, 45:83, 35:50] = 2500.0
    return volume


@pytest.fixture
def binary_bone_mask() -> np.ndarray:
    """Simple binary mask for mesh extraction testing."""
    mask = np.zeros((64, 64, 64), dtype=np.int32)
    mask[20:44, 20:44, 20:44] = 1  # Structure label 1
    return mask


@pytest.fixture
def multi_label_mask() -> np.ndarray:
    """Multi-structure segmentation mask."""
    mask = np.zeros((64, 64, 64), dtype=np.int32)
    mask[5:25, 5:25, 5:25] = 1    # mandible
    mask[5:20, 30:55, 5:25] = 2   # maxilla
    mask[10:25, 5:20, 30:50] = 3  # zygoma_L
    mask[10:25, 44:59, 30:50] = 4 # zygoma_R
    return mask


# ─── DICOM mock data ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_dicom_dataset():
    """Mock pydicom Dataset with standard CT attributes."""
    ds = MagicMock()
    ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9.10"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.10.1"
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.10.1.1"
    ds.StudyDescription = "Head CT"
    ds.SeriesDescription = "Axial Head"
    ds.Modality = "CT"
    ds.AcquisitionDate = "20240115"
    ds.StudyDate = "20240115"
    ds.BodyPartExamined = "HEAD"
    ds.Manufacturer = "Siemens"
    ds.ManufacturerModelName = "SOMATOM Definition Flash"
    ds.SliceThickness = 0.625
    ds.PixelSpacing = [0.488, 0.488]
    ds.KVP = 120
    ds.Exposure = 200
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.InstitutionName = "University Medical Center"
    ds.PatientName = "Doe^John"
    ds.PatientID = "MRN123456"
    ds.PatientBirthDate = "19800101"
    return ds


@pytest.fixture
def mock_study_metadata():
    """Mock StudyMetadataInternal for testing ingestion service."""
    from app.services.dicom.ingestion import StudyMetadataInternal, SeriesMetadata

    series = SeriesMetadata(
        series_instance_uid="1.2.3.4.5.6.7.8.9.10.1",
        series_number=1,
        series_description="Axial Head",
        modality="CT",
        slice_count=512,
        slice_thickness_mm=0.625,
        pixel_spacing=(0.488, 0.488),
        image_orientation=[1, 0, 0, 0, 1, 0],
        image_position_first=[0, 0, 0],
        image_position_last=None,
        kvp=120.0,
        exposure_mas=200.0,
        reconstruction_diameter=250.0,
        file_paths=["/tmp/test/slice_001.dcm"],
    )

    return StudyMetadataInternal(
        study_instance_uid="1.2.3.4.5.6.7.8.9.10",
        study_description="Head CT",
        modality="CT",
        acquisition_date="20240115",
        body_part_examined="HEAD",
        institution_name="University Medical Center",
        manufacturer="Siemens",
        manufacturer_model="SOMATOM Definition Flash",
        software_versions="syngo.CT 2021",
        series=[series],
    )


# ─── Database fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_db_session():
    """Mock async database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def sample_patient_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_study_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_case_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_segmentation_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_plan_id() -> str:
    return str(uuid.uuid4())


# ─── Fragment fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_fragments():
    """Sample fracture fragments for reduction planning tests."""
    from app.services.reduction.reduction_service import FragmentMesh

    # Simple cube-like fragments
    def make_cube_points(offset, size=10):
        pts = []
        for x in np.linspace(0, size, 5):
            for y in np.linspace(0, size, 5):
                for z in np.linspace(0, size, 5):
                    pts.append([x + offset[0], y + offset[1], z + offset[2]])
        return np.array(pts, dtype=np.float32)

    return [
        FragmentMesh(
            fragment_id="mandible_body",
            label_value=1,
            points=make_cube_points([0, -50, 0]),
            centroid_mm=np.array([5.0, -45.0, 5.0]),
            volume_mm3=1500.0,
            parent_structure="mandible",
            is_reference=True,
        ),
        FragmentMesh(
            fragment_id="mandible_condyle_L",
            label_value=2,
            points=make_cube_points([-30, -10, -10]),
            centroid_mm=np.array([-25.0, -5.0, -5.0]),
            volume_mm3=200.0,
            parent_structure="mandible",
            is_reference=False,
        ),
        FragmentMesh(
            fragment_id="mandible_condyle_R",
            label_value=3,
            points=make_cube_points([20, -10, -10]),
            centroid_mm=np.array([25.0, -5.0, -5.0]),
            volume_mm3=200.0,
            parent_structure="mandible",
            is_reference=False,
        ),
    ]


# ─── Mock model registry ──────────────────────────────────────────────────────


@pytest.fixture
def mock_model_registry():
    """Mock ML model registry that returns stub models."""
    registry = MagicMock()

    # Return a mock model that produces empty masks
    mock_model = MagicMock()
    mock_model.name = "mock_model"
    mock_model.version = "test_0.1"
    mock_model.predict.return_value = (
        np.zeros((64, 64, 64), dtype=np.int32),
        {"mandible": 1, "maxilla": 2},
        {"mandible": 0.92, "maxilla": 0.88},
    )

    registry.load_model.return_value = mock_model
    registry.get_available_models.return_value = ["totalsegmentator", "mock_model"]
    return registry


# ─── Transform fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def identity_transform():
    """Identity Transform3D."""
    from app.schemas.common import Transform3D
    return Transform3D.identity()


@pytest.fixture
def sample_transform():
    """Sample non-identity rigid body transform."""
    from app.schemas.common import Transform3D
    import math

    # 10-degree rotation about z-axis + 5mm translation
    angle = math.radians(10)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return Transform3D(
        rotation_matrix=[
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0],
        ],
        translation_mm=[5.0, -2.0, 1.0],
    )


# ─── FastAPI app + async HTTP client fixtures ─────────────────────────────────


@pytest.fixture
def app():
    """Create a fresh FastAPI test application instance."""
    from app.main import create_app
    test_app = create_app()
    return test_app


@pytest_asyncio.fixture
async def async_client(app):
    """
    Async HTTP client wired to the FastAPI app via ASGI transport.
    No network required — requests go directly through the ASGI interface.
    """
    if not _HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client


@pytest.fixture
def auth_headers():
    """
    Return Authorization headers with a valid test JWT.
    Uses the security module to mint a real token for a test surgeon user.
    """
    from app.core.security import create_access_token

    token = create_access_token(user_id="test-surgeon-001", role="surgeon")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers():
    """
    Return Authorization headers with a valid admin JWT.
    """
    from app.core.security import create_access_token

    token = create_access_token(user_id="test-admin-001", role="admin")
    return {"Authorization": f"Bearer {token}"}
