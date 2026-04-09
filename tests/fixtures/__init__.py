"""
Facial Align test fixtures package.

Provides realistic mock data for DICOM, mesh geometry, surgical plans,
cephalometric analyses, and case/patient records.

Quick import:
    from tests.fixtures import (
        # DICOM
        make_dicom_dataset,
        make_dicom_series,
        make_malformed_dicom,
        TYPICAL_CMF_CT_METADATA,
        DICOM_PHI_TAGS,
        # Mesh
        make_cube_mesh,
        make_sphere_mesh,
        make_mandible_proxy,
        make_fragment_pair,
        make_degenerate_mesh,
        make_point_cloud,
        SimpleMesh,
        # Plans
        make_reduction_plan,
        make_fragment_transform,
        make_occlusion_metrics,
        make_cephalometric_analysis,
        make_abnormal_cephalometric_analysis,
        make_symmetry_report,
        make_quality_report,
        NORMAL_CEPH_VALUES,
        CLASS_II_CEPH_VALUES,
        CLASS_III_CEPH_VALUES,
        # Cases
        make_case,
        make_patient,
        make_ct_study,
        SAMPLE_CASES,
        SAMPLE_FRACTURE_PATTERNS,
    )
"""

from tests.fixtures.dicom_fixtures import (
    DICOM_PHI_TAGS,
    MALFORMED_VARIANTS,
    TYPICAL_CMF_CT_METADATA,
    make_all_malformed_dicoms,
    make_dicom_dataset,
    make_dicom_series,
    make_malformed_dicom,
)
from tests.fixtures.mesh_fixtures import (
    SimpleMesh,
    make_cube_mesh,
    make_degenerate_mesh,
    make_fragment_pair,
    make_mandible_proxy,
    make_point_cloud,
    make_sphere_mesh,
)
from tests.fixtures.plan_fixtures import (
    CLASS_II_CEPH_VALUES,
    CLASS_III_CEPH_VALUES,
    NORMAL_CEPH_VALUES,
    make_abnormal_cephalometric_analysis,
    make_cephalometric_analysis,
    make_fragment_transform,
    make_occlusion_metrics,
    make_quality_report,
    make_reduction_plan,
    make_symmetry_report,
)
from tests.fixtures.case_fixtures import (
    SAMPLE_CASES,
    SAMPLE_FRACTURE_PATTERNS,
    make_case,
    make_ct_study,
    make_patient,
)

__all__ = [
    # DICOM
    "make_dicom_dataset",
    "make_dicom_series",
    "make_malformed_dicom",
    "make_all_malformed_dicoms",
    "TYPICAL_CMF_CT_METADATA",
    "DICOM_PHI_TAGS",
    "MALFORMED_VARIANTS",
    # Mesh
    "SimpleMesh",
    "make_cube_mesh",
    "make_sphere_mesh",
    "make_mandible_proxy",
    "make_fragment_pair",
    "make_degenerate_mesh",
    "make_point_cloud",
    # Plans
    "make_reduction_plan",
    "make_fragment_transform",
    "make_occlusion_metrics",
    "make_cephalometric_analysis",
    "make_abnormal_cephalometric_analysis",
    "make_symmetry_report",
    "make_quality_report",
    "NORMAL_CEPH_VALUES",
    "CLASS_II_CEPH_VALUES",
    "CLASS_III_CEPH_VALUES",
    # Cases
    "make_case",
    "make_patient",
    "make_ct_study",
    "SAMPLE_CASES",
    "SAMPLE_FRACTURE_PATTERNS",
]
