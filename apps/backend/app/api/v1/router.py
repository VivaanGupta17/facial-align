"""
API v1 aggregate router.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    capabilities,
    cases,
    dicom,
    health,
    jobs,
    planning,
    reviews,
    segmentation,
    viewer,
    websocket,
)

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(capabilities.router)
api_router.include_router(health.router)
api_router.include_router(cases.router)
api_router.include_router(dicom.router)
api_router.include_router(segmentation.router)
api_router.include_router(planning.router)
api_router.include_router(reviews.router)
api_router.include_router(viewer.router)
api_router.include_router(jobs.router)
api_router.include_router(websocket.router)
