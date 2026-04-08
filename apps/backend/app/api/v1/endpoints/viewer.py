"""
3D viewer endpoints: serve GLB meshes, volume slices, and anatomical landmarks.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import CaseNotFoundError, MeshError, SegmentationNotFoundError
from app.core.logging import get_logger
from app.core.security import CurrentUser, get_current_user
from app.db.database import get_db_session
from app.models.plan import ReductionPlan
from app.models.segmentation import SegmentationResult
from app.schemas.common import BaseSchema, Vector3D

router = APIRouter(prefix="/viewer", tags=["viewer"])
settings = get_settings()
logger = get_logger(__name__)


class LandmarkPoint(BaseSchema):
    """An anatomical landmark used for orientation and measurement."""
    name: str
    coordinates_mm: Vector3D
    confidence: Optional[float] = None
    landmark_type: Optional[str] = None  # "cephalometric", "surgical", "dental"
    description: Optional[str] = None


class VolumeSlice(BaseSchema):
    """Metadata for a 2D cross-sectional slice of the CT volume."""
    plane: str  # "axial", "coronal", "sagittal"
    slice_index: int
    position_mm: float
    width: int
    height: int
    window_center: float
    window_width: float
    pixel_spacing_mm: list[float]
    image_url: str  # URL to fetch PNG/WebP image


class SceneAssets(BaseSchema):
    """Complete set of 3D assets for the surgical planning viewer."""
    case_id: uuid.UUID
    segmentation_id: Optional[uuid.UUID]
    plan_id: Optional[uuid.UUID]
    meshes: Dict[str, str] = {}  # structure_name -> GLB URL
    fragment_meshes: Dict[str, str] = {}  # fragment_id -> GLB URL (with planned transforms)
    landmarks: List[LandmarkPoint] = []
    bounding_box_mm: Optional[Dict[str, float]] = None
    coordinate_system: str = "RAS"  # or "LPS"


@router.get(
    "/cases/{case_id}/assets",
    response_model=SceneAssets,
    summary="Get all 3D assets for viewer",
)
async def get_scene_assets(
    case_id: uuid.UUID,
    segmentation_id: Optional[uuid.UUID] = Query(None),
    plan_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> SceneAssets:
    """
    Get all 3D assets needed to render the surgical planning scene.

    Returns URLs to GLB mesh files, anatomical landmarks, and bounding box.
    If plan_id is provided, transforms are baked into fragment meshes.
    """
    # Determine active segmentation
    if segmentation_id:
        seg = (
            await db.execute(
                select(SegmentationResult).where(SegmentationResult.id == segmentation_id)
            )
        ).scalar_one_or_none()
        if not seg:
            raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")
    else:
        seg = (
            await db.execute(
                select(SegmentationResult)
                .where(SegmentationResult.case_id == case_id)
                .where(SegmentationResult.status == "complete")
                .order_by(SegmentationResult.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    # Build mesh URL map
    mesh_urls: Dict[str, str] = {}
    fragment_urls: Dict[str, str] = {}

    if seg and seg.mesh_storage_paths:
        base_url = f"/api/v1/viewer/meshes/{seg.id}"
        for structure, paths in seg.mesh_storage_paths.items():
            if isinstance(paths, dict) and "glb" in paths:
                mesh_urls[structure] = f"{base_url}/{structure}.glb"

        # Fragment meshes
        if seg.fragment_masks_path and plan_id:
            plan = (
                await db.execute(
                    select(ReductionPlan).where(ReductionPlan.id == plan_id)
                )
            ).scalar_one_or_none()
            if plan and plan.fragments:
                for fid in plan.fragments:
                    fragment_urls[fid] = f"{base_url}/fragment_{fid}.glb"

    return SceneAssets(
        case_id=case_id,
        segmentation_id=seg.id if seg else None,
        plan_id=plan_id,
        meshes=mesh_urls,
        fragment_meshes=fragment_urls,
        coordinate_system="RAS",
    )


@router.get(
    "/meshes/{segmentation_id}/{structure_name}.glb",
    summary="Serve GLB mesh file",
    response_class=FileResponse,
)
async def serve_glb_mesh(
    segmentation_id: uuid.UUID,
    structure_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Serve a GLB mesh file for the web viewer.
    GLB is the binary GLTF format optimized for web rendering.
    """
    seg = (
        await db.execute(
            select(SegmentationResult).where(SegmentationResult.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not seg:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    mesh_paths: dict = seg.mesh_storage_paths or {}
    structure_paths = mesh_paths.get(structure_name, {})
    glb_path = structure_paths.get("glb") if isinstance(structure_paths, dict) else None

    if not glb_path or not Path(glb_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GLB mesh not found for structure: {structure_name}",
        )

    return FileResponse(
        path=glb_path,
        media_type="model/gltf-binary",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f"inline; filename={structure_name}.glb",
        },
    )


@router.get(
    "/meshes/{segmentation_id}/{structure_name}.stl",
    summary="Serve STL mesh file",
    response_class=FileResponse,
)
async def serve_stl_mesh(
    segmentation_id: uuid.UUID,
    structure_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Serve an STL mesh file for CAD/CAM/3D printing workflows."""
    seg = (
        await db.execute(
            select(SegmentationResult).where(SegmentationResult.id == segmentation_id)
        )
    ).scalar_one_or_none()
    if not seg:
        raise SegmentationNotFoundError(f"Segmentation {segmentation_id} not found")

    mesh_paths: dict = seg.mesh_storage_paths or {}
    structure_paths = mesh_paths.get(structure_name, {})
    stl_path = structure_paths.get("stl") if isinstance(structure_paths, dict) else None

    if not stl_path or not Path(stl_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"STL mesh not found for structure: {structure_name}",
        )

    return FileResponse(
        path=stl_path,
        media_type="model/stl",
        filename=f"{structure_name}.stl",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get(
    "/cases/{case_id}/slices",
    response_model=List[VolumeSlice],
    summary="Get volume slice metadata",
)
async def get_volume_slices(
    case_id: uuid.UUID,
    plane: str = Query("axial", regex="^(axial|coronal|sagittal)$"),
    start_index: int = Query(0, ge=0),
    count: int = Query(10, ge=1, le=100),
    window_center: float = Query(400.0, description="Hounsfield unit window center (bone: 400)"),
    window_width: float = Query(2000.0, description="Hounsfield unit window width (bone: 1500-3000)"),
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[VolumeSlice]:
    """
    Get metadata for CT volume cross-sections.
    Actual slice images are served via /viewer/slices/{case_id}/{plane}/{index}.webp
    """
    # TODO: Query study volume dimensions from database
    # For now return mock slice descriptors
    slices = []
    for i in range(start_index, start_index + count):
        slices.append(VolumeSlice(
            plane=plane,
            slice_index=i,
            position_mm=float(i) * 0.5,
            width=512,
            height=512,
            window_center=window_center,
            window_width=window_width,
            pixel_spacing_mm=[0.5, 0.5],
            image_url=f"/api/v1/viewer/cases/{case_id}/slices/{plane}/{i}.webp"
            f"?wc={window_center}&ww={window_width}",
        ))
    return slices


@router.get(
    "/cases/{case_id}/landmarks",
    response_model=List[LandmarkPoint],
    summary="Get anatomical landmarks",
)
async def get_landmarks(
    case_id: uuid.UUID,
    landmark_type: Optional[str] = Query(None, description="Filter by type: cephalometric, surgical, dental"),
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[LandmarkPoint]:
    """
    Get anatomical landmarks for a case.
    Landmarks are detected during segmentation (nasion, ANS, PNS, menton, etc.).
    """
    # TODO: Load from landmark detection service output stored in segmentation result
    # Standard craniofacial cephalometric landmarks
    standard_landmarks = [
        LandmarkPoint(name="N", coordinates_mm=Vector3D(x=0, y=60, z=0),
                     landmark_type="cephalometric", description="Nasion"),
        LandmarkPoint(name="ANS", coordinates_mm=Vector3D(x=0, y=30, z=-10),
                     landmark_type="cephalometric", description="Anterior Nasal Spine"),
        LandmarkPoint(name="PNS", coordinates_mm=Vector3D(x=0, y=35, z=-40),
                     landmark_type="cephalometric", description="Posterior Nasal Spine"),
        LandmarkPoint(name="Me", coordinates_mm=Vector3D(x=0, y=-50, z=-5),
                     landmark_type="cephalometric", description="Menton"),
        LandmarkPoint(name="Pg", coordinates_mm=Vector3D(x=0, y=-45, z=5),
                     landmark_type="cephalometric", description="Pogonion"),
    ]

    if landmark_type:
        standard_landmarks = [l for l in standard_landmarks if l.landmark_type == landmark_type]

    logger.info(
        "landmarks_requested",
        case_id=str(case_id),
        type=landmark_type,
        count=len(standard_landmarks),
    )
    return standard_landmarks
