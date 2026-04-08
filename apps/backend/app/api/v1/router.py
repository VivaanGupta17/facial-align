"""
API v1 aggregate router.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import cases, dicom, health, planning, segmentation, viewer

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(cases.router)
api_router.include_router(dicom.router)
api_router.include_router(segmentation.router)
api_router.include_router(planning.router)
api_router.include_router(viewer.router)
