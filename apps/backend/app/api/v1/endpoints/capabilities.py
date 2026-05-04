"""Runtime capability registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.capabilities import CapabilitySnapshotResponse
from app.services.capabilities import get_capability_snapshot

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


@router.get("", response_model=CapabilitySnapshotResponse, summary="Get runtime capability registry")
async def list_capabilities() -> CapabilitySnapshotResponse:
    """Return the currently supported baseline and beta capabilities."""

    return get_capability_snapshot()
