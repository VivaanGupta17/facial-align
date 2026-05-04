from __future__ import annotations


def test_openapi_contract_exposes_release_routes(app):
    schema = app.openapi()
    paths = schema["paths"]

    required_paths = {
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/cases",
        "/api/v1/dicom/upload",
        "/api/v1/segmentation",
        "/api/v1/planning",
        "/api/v1/reviews/{case_id}",
        "/api/v1/viewer/cases/{case_id}/assets",
        "/api/v1/jobs/{job_id}",
        "/api/v1/health",
        "/api/v1/capabilities",
    }

    missing = required_paths - set(paths.keys())
    assert not missing, f"Missing expected API paths: {sorted(missing)}"
