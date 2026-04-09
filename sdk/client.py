"""
Facial Align Python SDK Client.

Provides a typed, async-first interface to the Facial Align REST API.
Used by:
- Integration tests
- CLI tools
- Jupyter notebooks
- External service integrations

Usage:
    async with FacialAlignClient("http://localhost:8000") as client:
        cases = await client.list_cases()
        case = await client.create_case(surgeon_id="dr-smith", ...)
        await client.upload_dicom(case.id, Path("./dicom_dir"))
        seg = await client.run_segmentation(case.id, model="totalsegmentator")
        plan = await client.run_reduction(case.id)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin


# ─── Response types ───────────────────────────────────────────────────────────

@dataclass
class ApiResponse:
    """Wrapper for API responses."""
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass
class Case:
    """Case resource."""
    id: str
    patient_id: str
    status: str
    case_type: str
    fracture_classification: str
    surgeon: str
    created_at: str
    updated_at: str
    fragments: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Case:
        return cls(
            id=data["id"],
            patient_id=data.get("patient_id", ""),
            status=data.get("status", "unknown"),
            case_type=data.get("case_type", ""),
            fracture_classification=data.get("fracture_classification", ""),
            surgeon=data.get("surgeon", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            fragments=data.get("fragments", []),
        )


@dataclass
class SegmentationResult:
    """Segmentation result resource."""
    id: str
    case_id: str
    model: str
    model_version: str
    status: str
    structures: Dict[str, Any] = field(default_factory=dict)
    overall_confidence: float = 0.0
    processing_time_seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SegmentationResult:
        return cls(
            id=data.get("id", ""),
            case_id=data.get("case_id", ""),
            model=data.get("model", ""),
            model_version=data.get("model_version", ""),
            status=data.get("status", ""),
            structures=data.get("structures", {}),
            overall_confidence=data.get("overall_confidence", 0.0),
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
        )


@dataclass
class ReductionPlan:
    """Reduction plan resource."""
    id: str
    case_id: str
    algorithm: str
    status: str
    overall_grade: str = ""
    fragment_transforms: Dict[str, Any] = field(default_factory=dict)
    evaluation: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReductionPlan:
        return cls(
            id=data.get("id", ""),
            case_id=data.get("case_id", ""),
            algorithm=data.get("algorithm", ""),
            status=data.get("status", ""),
            overall_grade=data.get("overall_grade", ""),
            fragment_transforms=data.get("fragment_transforms", {}),
            evaluation=data.get("evaluation"),
        )


@dataclass
class JobStatus:
    """Async job status."""
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JobStatus:
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", "unknown"),
            progress=data.get("progress", 0.0),
            message=data.get("message", ""),
            result=data.get("result"),
            error=data.get("error"),
        )


# ─── Exceptions ───────────────────────────────────────────────────────────────

class FacialAlignSDKError(Exception):
    """Base SDK error."""
    pass


class ApiError(FacialAlignSDKError):
    """API returned an error response."""
    def __init__(self, status_code: int, error_code: str, message: str, request_id: Optional[str] = None):
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        super().__init__(f"[{status_code}] {error_code}: {message}")


class ConnectionError(FacialAlignSDKError):
    """Failed to connect to the API."""
    pass


class TimeoutError(FacialAlignSDKError):
    """Request timed out."""
    pass


# ─── Client ──────────────────────────────────────────────────────────────────

class FacialAlignClient:
    """
    Async HTTP client for the Facial Align API.

    Supports both async context manager and manual lifecycle:

        # Context manager (recommended)
        async with FacialAlignClient("http://localhost:8000") as client:
            cases = await client.list_cases()

        # Manual lifecycle
        client = FacialAlignClient("http://localhost:8000")
        await client.connect()
        cases = await client.list_cases()
        await client.close()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._session = None

    async def connect(self) -> None:
        """Initialize the HTTP session."""
        try:
            import httpx
            headers = {"Accept": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        except ImportError:
            # Fallback: use aiohttp or urllib
            self._session = _FallbackSession(self._base_url, self._api_key, self._timeout)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and hasattr(self._session, "aclose"):
            await self._session.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ── Core HTTP methods ─────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None,
    ) -> ApiResponse:
        """Make an HTTP request with retry logic."""
        if not self._session:
            await self.connect()

        url = f"/api/v1{path}"
        last_error = None

        for attempt in range(self._max_retries):
            start = time.perf_counter()
            try:
                response = await self._session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    files=files,
                )
                duration_ms = (time.perf_counter() - start) * 1000

                request_id = response.headers.get("X-Request-ID")

                if response.status_code >= 400:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = {"error": "UNKNOWN", "message": response.text}

                    raise ApiError(
                        status_code=response.status_code,
                        error_code=error_body.get("error", "UNKNOWN"),
                        message=error_body.get("message", "Unknown error"),
                        request_id=request_id,
                    )

                return ApiResponse(
                    status_code=response.status_code,
                    data=response.json() if response.text else None,
                    headers=dict(response.headers),
                    request_id=request_id,
                    duration_ms=duration_ms,
                )

            except ApiError:
                raise  # Don't retry client errors
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise ConnectionError(f"Failed after {self._max_retries} attempts: {last_error}")

    async def _get(self, path: str, **params) -> ApiResponse:
        return await self._request("GET", path, params=params or None)

    async def _post(self, path: str, data: Optional[Dict] = None, **kwargs) -> ApiResponse:
        return await self._request("POST", path, json_data=data, **kwargs)

    async def _put(self, path: str, data: Optional[Dict] = None) -> ApiResponse:
        return await self._request("PUT", path, json_data=data)

    async def _delete(self, path: str) -> ApiResponse:
        return await self._request("DELETE", path)

    # ── Health ────────────────────────────────────────────────────────────────

    async def health(self) -> Dict[str, Any]:
        """Check API health."""
        resp = await self._get("/health")
        return resp.data

    async def ping(self) -> float:
        """Measure API round-trip time in ms."""
        start = time.perf_counter()
        await self._get("/health/ping")
        return (time.perf_counter() - start) * 1000

    # ── Cases ─────────────────────────────────────────────────────────────────

    async def list_cases(
        self,
        status: Optional[str] = None,
        surgeon: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Case]:
        """List all cases with optional filters."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if surgeon:
            params["surgeon"] = surgeon
        resp = await self._get("/cases", **params)
        items = resp.data.get("items", resp.data) if isinstance(resp.data, dict) else resp.data
        return [Case.from_dict(c) for c in items]

    async def get_case(self, case_id: str) -> Case:
        """Get a single case by ID."""
        resp = await self._get(f"/cases/{case_id}")
        return Case.from_dict(resp.data)

    async def create_case(
        self,
        patient_id: str,
        case_type: str,
        fracture_classification: str,
        surgeon: str,
        notes: str = "",
    ) -> Case:
        """Create a new case."""
        resp = await self._post("/cases", data={
            "patient_id": patient_id,
            "case_type": case_type,
            "fracture_classification": fracture_classification,
            "surgeon": surgeon,
            "notes": notes,
        })
        return Case.from_dict(resp.data)

    async def update_case(self, case_id: str, **fields) -> Case:
        """Update case fields."""
        resp = await self._put(f"/cases/{case_id}", data=fields)
        return Case.from_dict(resp.data)

    async def delete_case(self, case_id: str) -> None:
        """Delete a case."""
        await self._delete(f"/cases/{case_id}")

    # ── DICOM ─────────────────────────────────────────────────────────────────

    async def upload_dicom(
        self,
        case_id: str,
        dicom_path: Union[str, Path],
        auto_qc: bool = True,
        auto_deidentify: bool = True,
    ) -> JobStatus:
        """
        Upload DICOM files for a case.

        Args:
            case_id: Target case ID
            dicom_path: Path to DICOM directory or ZIP file
            auto_qc: Run quality control automatically
            auto_deidentify: Run de-identification automatically

        Returns:
            JobStatus for the async upload job
        """
        path = Path(dicom_path)
        if not path.exists():
            raise FileNotFoundError(f"DICOM path not found: {path}")

        resp = await self._post(f"/dicom/upload/{case_id}", data={
            "source_path": str(path),
            "auto_qc": auto_qc,
            "auto_deidentify": auto_deidentify,
        })
        return JobStatus.from_dict(resp.data)

    async def get_dicom_quality(self, case_id: str) -> Dict[str, Any]:
        """Get DICOM quality control report for a case."""
        resp = await self._get(f"/dicom/{case_id}/quality")
        return resp.data

    # ── Segmentation ──────────────────────────────────────────────────────────

    async def run_segmentation(
        self,
        case_id: str,
        model: str = "totalsegmentator",
        structures: Optional[List[str]] = None,
    ) -> JobStatus:
        """
        Trigger segmentation on a case.

        Returns:
            JobStatus for the async segmentation job
        """
        data: Dict[str, Any] = {"model": model}
        if structures:
            data["structures"] = structures
        resp = await self._post(f"/segmentation/{case_id}/run", data=data)
        return JobStatus.from_dict(resp.data)

    async def get_segmentation(self, case_id: str) -> SegmentationResult:
        """Get segmentation results for a case."""
        resp = await self._get(f"/segmentation/{case_id}")
        return SegmentationResult.from_dict(resp.data)

    # ── Planning ──────────────────────────────────────────────────────────────

    async def run_reduction(
        self,
        case_id: str,
        algorithm: str = "icp_baseline",
    ) -> JobStatus:
        """
        Trigger fracture reduction planning.

        Returns:
            JobStatus for the async planning job
        """
        resp = await self._post(f"/planning/{case_id}/reduce", data={
            "algorithm": algorithm,
        })
        return JobStatus.from_dict(resp.data)

    async def get_reduction_plan(self, case_id: str) -> ReductionPlan:
        """Get the current reduction plan for a case."""
        resp = await self._get(f"/planning/{case_id}/plan")
        return ReductionPlan.from_dict(resp.data)

    async def approve_plan(self, case_id: str, plan_id: str, notes: str = "") -> Dict[str, Any]:
        """Approve a reduction plan (surgeon action)."""
        resp = await self._post(f"/planning/{case_id}/approve/{plan_id}", data={
            "notes": notes,
        })
        return resp.data

    async def update_fragment_transform(
        self,
        case_id: str,
        plan_id: str,
        fragment_id: str,
        rotation_matrix: List[List[float]],
        translation_mm: List[float],
        rationale: str = "",
    ) -> Dict[str, Any]:
        """Update a single fragment transform (surgeon edit)."""
        resp = await self._put(
            f"/planning/{case_id}/plan/{plan_id}/fragments/{fragment_id}",
            data={
                "rotation_matrix": rotation_matrix,
                "translation_mm": translation_mm,
                "rationale": rationale,
            },
        )
        return resp.data

    # ── Viewer ────────────────────────────────────────────────────────────────

    async def get_viewer_scene(self, case_id: str) -> Dict[str, Any]:
        """Get 3D viewer scene data (meshes, transforms, metadata)."""
        resp = await self._get(f"/viewer/{case_id}/scene")
        return resp.data

    async def get_mesh_url(self, case_id: str, structure: str, resolution: str = "medium") -> str:
        """Get download URL for a structure mesh."""
        resp = await self._get(f"/viewer/{case_id}/mesh/{structure}", resolution=resolution)
        return resp.data.get("url", "")

    # ── Jobs ──────────────────────────────────────────────────────────────────

    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of an async job."""
        resp = await self._get(f"/jobs/{job_id}")
        return JobStatus.from_dict(resp.data)

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval_seconds: float = 2.0,
        timeout_seconds: float = 300.0,
    ) -> JobStatus:
        """
        Poll a job until completion or timeout.

        Args:
            job_id: Job ID to monitor
            poll_interval_seconds: Seconds between polls
            timeout_seconds: Maximum wait time

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If the job doesn't complete within timeout
            ApiError: If the job fails
        """
        start = time.perf_counter()
        while True:
            status = await self.get_job_status(job_id)
            if status.is_complete:
                if status.is_failed:
                    raise ApiError(
                        status_code=500,
                        error_code="JOB_FAILED",
                        message=status.error or "Job failed",
                    )
                return status
            elapsed = time.perf_counter() - start
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout_seconds}s")
            await asyncio.sleep(poll_interval_seconds)


# ─── Fallback session (no httpx) ─────────────────────────────────────────────

class _FallbackSession:
    """Minimal async HTTP client using urllib (for environments without httpx)."""

    def __init__(self, base_url: str, api_key: Optional[str], timeout: float):
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout

    async def request(self, method: str, url: str, **kwargs) -> Any:
        """Make a synchronous HTTP request wrapped in asyncio."""
        import urllib.request
        import urllib.error

        full_url = f"{self._base_url}{url}"

        # Build query string from params
        params = kwargs.get("params")
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query:
                full_url += f"?{query}"

        json_data = kwargs.get("json")
        data = json.dumps(json_data).encode() if json_data else None

        req = urllib.request.Request(full_url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        loop = asyncio.get_event_loop()

        def _do_request():
            try:
                resp = urllib.request.urlopen(req, timeout=self._timeout)
                return _FallbackResponse(
                    status_code=resp.status,
                    text=resp.read().decode(),
                    headers=dict(resp.headers),
                )
            except urllib.error.HTTPError as e:
                return _FallbackResponse(
                    status_code=e.code,
                    text=e.read().decode(),
                    headers=dict(e.headers),
                )

        return await loop.run_in_executor(None, _do_request)

    async def aclose(self):
        pass


@dataclass
class _FallbackResponse:
    status_code: int
    text: str
    headers: Dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.text) if self.text else None
