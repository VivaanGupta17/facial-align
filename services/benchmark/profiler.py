"""
ML inference and pipeline benchmarking framework.

Provides standardized profiling of every stage in the surgical planning
pipeline — from DICOM ingestion through final plan evaluation.  Results
are reported as structured BenchmarkReport objects suitable for both
human-readable summaries and CI regression dashboards.

Usage:
    profiler = PipelineProfiler()
    with profiler.stage("segmentation"):
        result = segmentator.run(volume)
    report = profiler.report()

Design notes:
- Pure Python — no GPU profiling dependencies (torch.cuda.Event hooks
  are optional and gated behind `torch.cuda.is_available()`)
- Memory tracking uses psutil (with graceful fallback)
- Thread-safe via threading.local()
"""

from __future__ import annotations

import gc
import statistics
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─── Data structures ──────────────────────────────────────────────────────────

class PipelineStage(str, Enum):
    """Canonical pipeline stage names."""
    DICOM_INGESTION = "dicom_ingestion"
    PREPROCESSING = "preprocessing"
    QUALITY_CONTROL = "quality_control"
    DEIDENTIFICATION = "deidentification"
    SEGMENTATION = "segmentation"
    MESH_EXTRACTION = "mesh_extraction"
    FRACTURE_DETECTION = "fracture_detection"
    REGISTRATION = "registration"
    REDUCTION_PLANNING = "reduction_planning"
    OCCLUSION_ANALYSIS = "occlusion_analysis"
    CEPHALOMETRIC_ANALYSIS = "cephalometric_analysis"
    SYMMETRY_ANALYSIS = "symmetry_analysis"
    PLAN_EVALUATION = "plan_evaluation"
    REPORT_GENERATION = "report_generation"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class StageTiming:
    """Timing record for a single execution of a pipeline stage."""
    stage: str
    wall_time_seconds: float
    cpu_time_seconds: float
    peak_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    input_size_description: str = ""
    output_size_description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def wall_time_ms(self) -> float:
        return self.wall_time_seconds * 1000

    @property
    def cpu_time_ms(self) -> float:
        return self.cpu_time_seconds * 1000


@dataclass
class StageStatistics:
    """Aggregate statistics over multiple runs of a pipeline stage."""
    stage: str
    n_runs: int
    wall_time_mean_ms: float
    wall_time_std_ms: float
    wall_time_min_ms: float
    wall_time_max_ms: float
    wall_time_p50_ms: float
    wall_time_p95_ms: float
    wall_time_p99_ms: float
    cpu_time_mean_ms: float
    peak_memory_mean_mb: float
    peak_memory_max_mb: float
    gpu_memory_mean_mb: float = 0.0
    throughput_items_per_sec: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a pipeline run or benchmark suite."""
    name: str
    description: str
    stages: List[StageStatistics]
    total_wall_time_ms: float
    total_cpu_time_ms: float
    peak_memory_mb: float
    environment: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_markdown(self) -> str:
        """Generate a Markdown-formatted benchmark report."""
        lines = [
            f"# Benchmark Report: {self.name}",
            "",
            f"**Description:** {self.description}",
            f"**Timestamp:** {self.timestamp}",
            f"**Total wall time:** {self.total_wall_time_ms:.1f} ms",
            f"**Peak memory:** {self.peak_memory_mb:.1f} MB",
            "",
            "## Environment",
            "",
        ]

        for key, value in self.environment.items():
            lines.append(f"- **{key}:** {value}")

        lines.extend([
            "",
            "## Stage Performance",
            "",
            "| Stage | Runs | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Peak Mem (MB) |",
            "|-------|------|-----------|----------|----------|----------|----------|---------------|",
        ])

        for s in self.stages:
            lines.append(
                f"| {s.stage} | {s.n_runs} | {s.wall_time_mean_ms:.1f} | "
                f"{s.wall_time_std_ms:.1f} | {s.wall_time_p50_ms:.1f} | "
                f"{s.wall_time_p95_ms:.1f} | {s.wall_time_p99_ms:.1f} | "
                f"{s.peak_memory_max_mb:.1f} |"
            )

        lines.extend([
            "",
            "## Bottleneck Analysis",
            "",
        ])

        if self.stages:
            sorted_stages = sorted(self.stages, key=lambda s: s.wall_time_mean_ms, reverse=True)
            total = sum(s.wall_time_mean_ms * s.n_runs for s in self.stages)
            for i, s in enumerate(sorted_stages[:5]):
                pct = (s.wall_time_mean_ms * s.n_runs / total * 100) if total > 0 else 0
                lines.append(f"{i+1}. **{s.stage}** — {s.wall_time_mean_ms:.1f} ms mean ({pct:.1f}% of total)")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report for JSON export."""
        return {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "total_wall_time_ms": self.total_wall_time_ms,
            "total_cpu_time_ms": self.total_cpu_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "environment": self.environment,
            "stages": [
                {
                    "stage": s.stage,
                    "n_runs": s.n_runs,
                    "wall_time_mean_ms": s.wall_time_mean_ms,
                    "wall_time_std_ms": s.wall_time_std_ms,
                    "wall_time_min_ms": s.wall_time_min_ms,
                    "wall_time_max_ms": s.wall_time_max_ms,
                    "wall_time_p50_ms": s.wall_time_p50_ms,
                    "wall_time_p95_ms": s.wall_time_p95_ms,
                    "wall_time_p99_ms": s.wall_time_p99_ms,
                    "cpu_time_mean_ms": s.cpu_time_mean_ms,
                    "peak_memory_mean_mb": s.peak_memory_mean_mb,
                    "peak_memory_max_mb": s.peak_memory_max_mb,
                    "gpu_memory_mean_mb": s.gpu_memory_mean_mb,
                }
                for s in self.stages
            ],
        }


# ─── Memory tracking ─────────────────────────────────────────────────────────

def _get_process_memory_mb() -> float:
    """Get current process RSS memory in megabytes."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in megabytes (if CUDA available)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def _get_environment_info() -> Dict[str, Any]:
    """Collect environment information for the benchmark report."""
    import platform
    import os

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }

    # Check for GPU
    try:
        import torch
        env["pytorch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
    except ImportError:
        env["pytorch_version"] = "not installed"
        env["cuda_available"] = False

    # Check for numpy
    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except ImportError:
        pass

    return env


# ─── Profiler ─────────────────────────────────────────────────────────────────

class PipelineProfiler:
    """
    Pipeline stage profiler with timing, memory tracking, and reporting.

    Supports both context manager and decorator usage:

        # Context manager
        profiler = PipelineProfiler()
        with profiler.stage("segmentation"):
            result = model.predict(volume)

        # Decorator
        @profiler.profile("segmentation")
        def run_segmentation(volume):
            return model.predict(volume)

        # Multi-run benchmark
        profiler.benchmark(
            name="segmentation",
            fn=lambda: model.predict(volume),
            n_runs=10,
            warmup=2,
        )
    """

    def __init__(self, name: str = "pipeline_benchmark") -> None:
        self._name = name
        self._timings: Dict[str, List[StageTiming]] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear all recorded timings."""
        with self._lock:
            self._timings.clear()

    @contextmanager
    def stage(
        self,
        name: str,
        input_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for timing a pipeline stage.

        Args:
            name: Stage name (use PipelineStage enum values for consistency)
            input_description: Description of input (e.g., "512x512x200 volume")
            metadata: Additional metadata to record
        """
        gc.collect()
        mem_before = _get_process_memory_mb()
        gpu_before = _get_gpu_memory_mb()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        try:
            yield
        finally:
            wall_elapsed = time.perf_counter() - wall_start
            cpu_elapsed = time.process_time() - cpu_start
            mem_after = _get_process_memory_mb()
            gpu_after = _get_gpu_memory_mb()

            timing = StageTiming(
                stage=name,
                wall_time_seconds=wall_elapsed,
                cpu_time_seconds=cpu_elapsed,
                peak_memory_mb=max(0, mem_after - mem_before),
                gpu_memory_mb=max(0, gpu_after - gpu_before),
                input_size_description=input_description,
                metadata=metadata or {},
            )

            with self._lock:
                if name not in self._timings:
                    self._timings[name] = []
                self._timings[name].append(timing)

    def profile(
        self,
        stage_name: str,
        input_description: str = "",
    ) -> Callable:
        """Decorator for profiling a function as a pipeline stage."""
        def decorator(fn: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.stage(stage_name, input_description):
                    return fn(*args, **kwargs)
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper
        return decorator

    def benchmark(
        self,
        name: str,
        fn: Callable[[], Any],
        n_runs: int = 10,
        warmup: int = 2,
        input_description: str = "",
    ) -> StageStatistics:
        """
        Run a function multiple times and collect statistics.

        Args:
            name: Stage name for reporting
            fn: Zero-argument callable to benchmark
            n_runs: Number of timed runs
            warmup: Number of warmup runs (not timed)
            input_description: Description of the input

        Returns:
            StageStatistics for the benchmark
        """
        # Warmup
        for _ in range(warmup):
            fn()

        # Timed runs
        for _ in range(n_runs):
            with self.stage(name, input_description):
                fn()

        return self._compute_statistics(name)

    def _compute_statistics(self, stage_name: str) -> StageStatistics:
        """Compute aggregate statistics for a stage."""
        with self._lock:
            timings = self._timings.get(stage_name, [])

        if not timings:
            return StageStatistics(
                stage=stage_name, n_runs=0,
                wall_time_mean_ms=0, wall_time_std_ms=0,
                wall_time_min_ms=0, wall_time_max_ms=0,
                wall_time_p50_ms=0, wall_time_p95_ms=0, wall_time_p99_ms=0,
                cpu_time_mean_ms=0,
                peak_memory_mean_mb=0, peak_memory_max_mb=0,
            )

        wall_times_ms = [t.wall_time_ms for t in timings]
        cpu_times_ms = [t.cpu_time_ms for t in timings]
        memories = [t.peak_memory_mb for t in timings]
        gpu_memories = [t.gpu_memory_mb for t in timings]
        n = len(wall_times_ms)

        sorted_wall = sorted(wall_times_ms)

        return StageStatistics(
            stage=stage_name,
            n_runs=n,
            wall_time_mean_ms=statistics.mean(wall_times_ms),
            wall_time_std_ms=statistics.stdev(wall_times_ms) if n > 1 else 0.0,
            wall_time_min_ms=min(wall_times_ms),
            wall_time_max_ms=max(wall_times_ms),
            wall_time_p50_ms=self._percentile(sorted_wall, 50),
            wall_time_p95_ms=self._percentile(sorted_wall, 95),
            wall_time_p99_ms=self._percentile(sorted_wall, 99),
            cpu_time_mean_ms=statistics.mean(cpu_times_ms),
            peak_memory_mean_mb=statistics.mean(memories),
            peak_memory_max_mb=max(memories),
            gpu_memory_mean_mb=statistics.mean(gpu_memories) if gpu_memories else 0.0,
        )

    @staticmethod
    def _percentile(sorted_data: List[float], pct: float) -> float:
        """Compute percentile from pre-sorted data."""
        if not sorted_data:
            return 0.0
        idx = (pct / 100) * (len(sorted_data) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_data) - 1)
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

    def report(self, description: str = "") -> BenchmarkReport:
        """
        Generate a complete benchmark report from all recorded timings.

        Returns:
            BenchmarkReport with per-stage statistics and environment info
        """
        from datetime import datetime, timezone

        stage_stats = []
        with self._lock:
            for stage_name in self._timings:
                stats = self._compute_statistics(stage_name)
                stage_stats.append(stats)

        # Sort by mean wall time descending (slowest first)
        stage_stats.sort(key=lambda s: s.wall_time_mean_ms, reverse=True)

        total_wall = sum(s.wall_time_mean_ms * s.n_runs for s in stage_stats)
        total_cpu = sum(s.cpu_time_mean_ms * s.n_runs for s in stage_stats)
        peak_mem = max((s.peak_memory_max_mb for s in stage_stats), default=0.0)

        return BenchmarkReport(
            name=self._name,
            description=description or f"Benchmark of {len(stage_stats)} pipeline stages",
            stages=stage_stats,
            total_wall_time_ms=total_wall,
            total_cpu_time_ms=total_cpu,
            peak_memory_mb=peak_mem,
            environment=_get_environment_info(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_timings(self, stage_name: str) -> List[StageTiming]:
        """Get raw timings for a specific stage."""
        with self._lock:
            return list(self._timings.get(stage_name, []))


# ─── Convenience: full pipeline benchmark ─────────────────────────────────────

class PipelineBenchmarkSuite:
    """
    Standardized benchmark suite for the full surgical planning pipeline.

    Runs each stage with synthetic data of configurable size and collects
    comprehensive performance metrics.

    Usage:
        suite = PipelineBenchmarkSuite(volume_shape=(512, 512, 200))
        report = suite.run(n_iterations=5)
        print(report.to_markdown())
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int] = (512, 512, 200),
        n_structures: int = 15,
        n_fragments: int = 4,
    ) -> None:
        self._volume_shape = volume_shape
        self._n_structures = n_structures
        self._n_fragments = n_fragments
        self._profiler = PipelineProfiler(name="full_pipeline_benchmark")

    def run(
        self,
        n_iterations: int = 5,
        warmup: int = 1,
        stages: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """
        Run the benchmark suite.

        Args:
            n_iterations: Number of timed iterations per stage
            warmup: Number of warmup iterations
            stages: Specific stages to benchmark (None = all)

        Returns:
            BenchmarkReport with results for all benchmarked stages
        """
        import numpy as np

        self._profiler.reset()

        all_stages = stages or [
            PipelineStage.PREPROCESSING.value,
            PipelineStage.QUALITY_CONTROL.value,
            PipelineStage.SEGMENTATION.value,
            PipelineStage.MESH_EXTRACTION.value,
            PipelineStage.REGISTRATION.value,
            PipelineStage.SYMMETRY_ANALYSIS.value,
            PipelineStage.CEPHALOMETRIC_ANALYSIS.value,
            PipelineStage.PLAN_EVALUATION.value,
            PipelineStage.REPORT_GENERATION.value,
        ]

        vol_desc = f"{self._volume_shape[0]}x{self._volume_shape[1]}x{self._volume_shape[2]}"

        for stage_name in all_stages:
            fn = self._get_stage_function(stage_name)
            if fn is not None:
                self._profiler.benchmark(
                    name=stage_name,
                    fn=fn,
                    n_runs=n_iterations,
                    warmup=warmup,
                    input_description=f"volume={vol_desc}, structures={self._n_structures}",
                )

        return self._profiler.report(
            description=(
                f"Full pipeline benchmark: {vol_desc} volume, "
                f"{self._n_structures} structures, {self._n_fragments} fragments, "
                f"{n_iterations} iterations"
            )
        )

    def _get_stage_function(self, stage_name: str) -> Optional[Callable]:
        """Get a benchmark function for a pipeline stage using synthetic data."""
        import numpy as np

        shape = self._volume_shape

        if stage_name == PipelineStage.PREPROCESSING.value:
            def bench_preprocessing():
                # Simulate HU windowing and resampling
                bench_shape = tuple(min(s, 128) for s in shape)
                volume = np.random.randint(-1000, 3000, size=bench_shape, dtype=np.int16)
                bone_min, bone_max = 200, 2000
                windowed = np.clip(volume, bone_min, bone_max)
                normalized = (windowed - bone_min) / (bone_max - bone_min)
                return normalized
            return bench_preprocessing

        if stage_name == PipelineStage.QUALITY_CONTROL.value:
            def bench_qc():
                bench_shape = tuple(min(s, 128) for s in shape)
                volume = np.random.randint(-1000, 3000, size=bench_shape, dtype=np.int16)
                # Simulate slice consistency check
                slice_means = np.mean(volume, axis=(0, 1))
                slice_stds = np.std(volume, axis=(0, 1))
                cv = slice_stds / (np.abs(slice_means) + 1e-6)
                motion_score = float(np.mean(cv > 0.5))
                # HU histogram analysis
                hist, _ = np.histogram(volume.ravel(), bins=100, range=(-1000, 3000))
                bone_fraction = np.sum(hist[30:]) / np.sum(hist)
                return {"motion_score": motion_score, "bone_fraction": bone_fraction}
            return bench_qc

        if stage_name == PipelineStage.SEGMENTATION.value:
            def bench_segmentation():
                # Simulate mask generation (thresholding baseline)
                # Use a smaller volume for the benchmark to avoid timeouts
                seg_shape = tuple(min(s, 128) for s in shape)
                volume = np.random.randint(-1000, 3000, size=seg_shape, dtype=np.int16)
                mask = np.zeros(seg_shape, dtype=np.uint8)
                mask[volume > 300] = 1   # Bone
                mask[volume > 1200] = 2  # Dense bone
                # Connected component labeling (simulated)
                from scipy import ndimage
                labeled, n_features = ndimage.label(mask > 0)
                return labeled, n_features
            return bench_segmentation

        if stage_name == PipelineStage.MESH_EXTRACTION.value:
            def bench_mesh():
                # Generate a binary mask and extract surface
                mask = np.zeros((64, 64, 64), dtype=np.uint8)
                mask[15:50, 15:50, 15:50] = 1
                # Gaussian smooth
                from scipy.ndimage import gaussian_filter
                smoothed = gaussian_filter(mask.astype(np.float32), sigma=1.0)
                # Marching cubes
                from skimage.measure import marching_cubes
                verts, faces, normals, values = marching_cubes(smoothed, level=0.5)
                return verts, faces
            return bench_mesh

        if stage_name == PipelineStage.REGISTRATION.value:
            def bench_registration():
                # Simulate ICP-like registration
                n_points = 5000
                source = np.random.randn(n_points, 3) * 10
                # Apply known transform
                angle = np.radians(5)
                R = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle),  np.cos(angle), 0],
                    [0, 0, 1],
                ])
                target = (R @ source.T).T + np.array([1.5, -0.8, 0.3])
                target += np.random.randn(n_points, 3) * 0.1
                # Simple nearest-neighbor ICP iteration
                from scipy.spatial import cKDTree
                current = source.copy()
                for _ in range(20):
                    tree = cKDTree(target)
                    _, idx = tree.query(current)
                    matched = target[idx]
                    centroid_s = np.mean(current, axis=0)
                    centroid_t = np.mean(matched, axis=0)
                    H = (current - centroid_s).T @ (matched - centroid_t)
                    U, _, Vt = np.linalg.svd(H)
                    R_est = Vt.T @ U.T
                    if np.linalg.det(R_est) < 0:
                        Vt[-1, :] *= -1
                        R_est = Vt.T @ U.T
                    t_est = centroid_t - R_est @ centroid_s
                    current = (R_est @ current.T).T + t_est
                rms = float(np.sqrt(np.mean(np.sum((current - target) ** 2, axis=1))))
                return rms
            return bench_registration

        if stage_name == PipelineStage.SYMMETRY_ANALYSIS.value:
            def bench_symmetry():
                n_points = 10000
                points = np.random.randn(n_points, 3) * 20
                # PCA for midsagittal plane
                centered = points - np.mean(points, axis=0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                normal = Vt[0]
                # Mirror and compute distances
                mirrored = points - 2 * np.outer(points @ normal, normal)
                from scipy.spatial import cKDTree
                tree = cKDTree(points)
                dists, _ = tree.query(mirrored)
                asymmetry_score = float(np.mean(dists))
                return asymmetry_score
            return bench_symmetry

        if stage_name == PipelineStage.CEPHALOMETRIC_ANALYSIS.value:
            def bench_ceph():
                # Simulate landmark detection from mask
                mask = np.zeros((128, 128, 128), dtype=np.uint8)
                mask[30:100, 30:100, 30:100] = 1
                # Find extremal points (simplified)
                coords = np.argwhere(mask > 0).astype(np.float64)
                landmarks = {
                    "menton": coords[np.argmin(coords[:, 2])],
                    "vertex": coords[np.argmax(coords[:, 2])],
                    "pogonion": coords[np.argmax(coords[:, 1])],
                    "gonion_L": coords[np.argmin(coords[:, 0])],
                    "gonion_R": coords[np.argmax(coords[:, 0])],
                }
                return landmarks
            return bench_ceph

        if stage_name == PipelineStage.PLAN_EVALUATION.value:
            def bench_evaluation():
                # Simulate evaluation with random fragment data
                n_frags = self._n_fragments
                transforms = {}
                for i in range(n_frags):
                    T = np.eye(4)
                    T[:3, 3] = np.random.randn(3) * 2
                    transforms[f"fragment_{i}"] = T
                # Simulate surface distance computation
                from scipy.spatial import cKDTree
                for fid, T in transforms.items():
                    source = np.random.randn(1000, 3) * 10
                    target = np.random.randn(1000, 3) * 10
                    tree = cKDTree(target)
                    dists, _ = tree.query(source)
                    mean_dist = float(np.mean(dists))
                return mean_dist
            return bench_evaluation

        if stage_name == PipelineStage.REPORT_GENERATION.value:
            def bench_report():
                # Simulate Markdown report generation
                lines = ["# Benchmark Report\n"]
                for i in range(50):
                    lines.append(f"| Fragment {i} | {np.random.rand():.3f} | {np.random.rand()*5:.1f} mm |")
                report_text = "\n".join(lines)
                return report_text
            return bench_report

        return None


# ─── Regression tracking ──────────────────────────────────────────────────────

@dataclass
class RegressionBaseline:
    """
    Performance baseline for regression detection.

    Store known-good performance numbers and alert when new benchmarks
    exceed thresholds.
    """
    stage: str
    wall_time_p95_ms: float
    peak_memory_mb: float
    tolerance_pct: float = 20.0  # Alert if >20% regression

    def check(self, stats: StageStatistics) -> Optional[str]:
        """
        Check stats against baseline.

        Returns:
            Warning message if regression detected, None otherwise
        """
        if stats.wall_time_p95_ms > self.wall_time_p95_ms * (1 + self.tolerance_pct / 100):
            pct_increase = (stats.wall_time_p95_ms / self.wall_time_p95_ms - 1) * 100
            return (
                f"Performance regression in {self.stage}: "
                f"P95 latency {stats.wall_time_p95_ms:.1f}ms "
                f"(+{pct_increase:.0f}% vs baseline {self.wall_time_p95_ms:.1f}ms)"
            )
        if stats.peak_memory_max_mb > self.peak_memory_mb * (1 + self.tolerance_pct / 100):
            pct_increase = (stats.peak_memory_max_mb / self.peak_memory_mb - 1) * 100
            return (
                f"Memory regression in {self.stage}: "
                f"peak {stats.peak_memory_max_mb:.1f}MB "
                f"(+{pct_increase:.0f}% vs baseline {self.peak_memory_mb:.1f}MB)"
            )
        return None


# Default baselines (will be updated after first benchmark run)
DEFAULT_BASELINES = {
    PipelineStage.PREPROCESSING.value: RegressionBaseline(
        stage=PipelineStage.PREPROCESSING.value,
        wall_time_p95_ms=500.0,
        peak_memory_mb=2000.0,
    ),
    PipelineStage.SEGMENTATION.value: RegressionBaseline(
        stage=PipelineStage.SEGMENTATION.value,
        wall_time_p95_ms=30000.0,  # ~30s for TotalSegmentator
        peak_memory_mb=4000.0,
    ),
    PipelineStage.MESH_EXTRACTION.value: RegressionBaseline(
        stage=PipelineStage.MESH_EXTRACTION.value,
        wall_time_p95_ms=5000.0,
        peak_memory_mb=1000.0,
    ),
    PipelineStage.REGISTRATION.value: RegressionBaseline(
        stage=PipelineStage.REGISTRATION.value,
        wall_time_p95_ms=2000.0,
        peak_memory_mb=500.0,
    ),
    PipelineStage.PLAN_EVALUATION.value: RegressionBaseline(
        stage=PipelineStage.PLAN_EVALUATION.value,
        wall_time_p95_ms=1000.0,
        peak_memory_mb=200.0,
    ),
}
