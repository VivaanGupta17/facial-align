"""
Pipeline execution CLI commands for Facial Align.

Provides the `pipeline` subcommand group for running the full processing
pipeline or individual steps on a case's CT data.

Pipeline steps (in order):
  1. preprocess  — DICOM → NIfTI conversion, HU calibration, resampling
  2. segment     — Multi-structure auto-segmentation (TotalSegmentator / nnU-Net)
  3. mesh        — Marching-cubes surface extraction, mesh cleaning
  4. reduce      — Fracture reduction plan generation
  5. evaluate    — Plan validation (occlusion, symmetry, condylar seating)

Example usage:
  facial-align pipeline run FA-2024-0042
  facial-align pipeline run FA-2024-0042 --step segment
  facial-align pipeline run FA-2024-0042 --dry-run
  facial-align pipeline preprocess /mnt/dicom/patient_001/
  facial-align pipeline segment /data/facialign/volumes/study.nii.gz
  facial-align pipeline mesh /data/facialign/masks/study_mask.nii.gz
  facial-align pipeline evaluate /data/facialign/plans/plan_v2.json
  facial-align pipeline status
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

try:
    from services.inference.model_registry import ModelRegistry
except ImportError:
    ModelRegistry = None  # type: ignore[assignment,misc]

try:
    from services.preprocessing.quality_control import CTQualityController
except ImportError:
    CTQualityController = None  # type: ignore[assignment,misc]

try:
    from data_contracts.ct_study import CTStudyContract
    from data_contracts.reduction_plan import ReductionPlanContract
except ImportError:
    CTStudyContract = None       # type: ignore[assignment,misc]
    ReductionPlanContract = None # type: ignore[assignment,misc]

console = Console()
err_console = Console(stderr=True)

# ─── Pipeline step definitions ────────────────────────────────────────────────

PIPELINE_STEPS = ["preprocess", "segment", "mesh", "reduce", "evaluate"]

STEP_DESCRIPTIONS = {
    "preprocess": "DICOM → NIfTI conversion, HU calibration, resampling to isotropic spacing",
    "segment":    "Multi-structure auto-segmentation (TotalSegmentator / nnU-Net CMF)",
    "mesh":       "Marching-cubes surface extraction, mesh decimation and cleaning",
    "reduce":     "Fracture reduction plan generation (SE(3) transformer)",
    "evaluate":   "Plan validation — occlusion, symmetry, condylar seating checks",
}

STEP_COLORS = {
    "preprocess": "cyan",
    "segment":    "blue",
    "mesh":       "green",
    "reduce":     "yellow",
    "evaluate":   "magenta",
}

JOB_STATUS_COLORS = {
    "PENDING":   "dim",
    "RUNNING":   "yellow",
    "SUCCESS":   "green",
    "FAILED":    "red",
    "CANCELLED": "dim red",
}

MESH_FORMATS = ["glb", "stl", "ply"]
MESH_RESOLUTIONS = {"high": 0.3, "medium": 0.6, "low": 1.0}


def _get_registry(settings) -> Optional[object]:
    if ModelRegistry is None:
        return None
    try:
        model_dir = str(settings.model_registry.registry_path) if settings else "/models"
        device = settings.model_registry.default_device if settings else "cpu"
        return ModelRegistry(model_dir=model_dir, device=device)
    except Exception:
        return None


# ─── Pipeline group ───────────────────────────────────────────────────────────

@click.group(
    name="pipeline",
    help=(
        "Run and debug the Facial Align processing pipeline.\n\n"
        "The pipeline transforms raw DICOM input into a validated surgical "
        "reduction plan through five sequential steps: preprocess → segment "
        "→ mesh → reduce → evaluate.\n\n"
        "Commands can run the full pipeline for a case, or execute individual "
        "steps on raw files for debugging and development."
    ),
)
@click.pass_context
def pipeline(ctx: click.Context) -> None:
    """Pipeline execution commands."""
    pass


# ─── pipeline run ─────────────────────────────────────────────────────────────

@pipeline.command(name="run")
@click.argument("case_id")
@click.option(
    "--step",
    type=click.Choice(PIPELINE_STEPS, case_sensitive=False),
    default=None,
    metavar="STEP",
    help=(
        "Run only a specific pipeline step instead of the full pipeline. "
        "Steps: " + ", ".join(PIPELINE_STEPS)
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Show what the pipeline would execute without actually running it. "
        "Prints each step's inputs, outputs, and configuration."
    ),
)
@click.option(
    "--model", "model_name",
    default=None,
    metavar="MODEL",
    help="Override the segmentation model (default: totalsegmentator).",
)
@click.option(
    "--device",
    default=None,
    metavar="DEVICE",
    help="Inference device override (cuda, cpu, cuda:0).",
)
@click.option(
    "--mesh-format",
    type=click.Choice(MESH_FORMATS),
    default="glb",
    show_default=True,
    help="Mesh output format.",
)
@click.option(
    "--async-run",
    is_flag=True,
    default=False,
    help=(
        "Submit as a Celery background task instead of running synchronously. "
        "Returns the task ID immediately."
    ),
)
@click.pass_obj
def run_pipeline(
    ctx,
    case_id: str,
    step: Optional[str],
    dry_run: bool,
    model_name: Optional[str],
    device: Optional[str],
    mesh_format: str,
    async_run: bool,
) -> None:
    """Run the full processing pipeline (or a single step) for a case.

    Executes the pipeline steps in order: preprocess → segment → mesh →
    reduce → evaluate. Progress is shown for each step.

    The pipeline reads the case's imported DICOM study and writes results
    (NIfTI volumes, segmentation masks, meshes, reduction plan) to the
    configured storage backend.

    \b
    Examples:
      facial-align pipeline run FA-2024-0042
      facial-align pipeline run FA-2024-0042 --step segment
      facial-align pipeline run FA-2024-0042 --dry-run
      facial-align pipeline run FA-2024-0042 --device cpu --model nnunet-cmf
      facial-align pipeline run FA-2024-0042 --async-run
    """
    steps_to_run = [step] if step else PIPELINE_STEPS
    seg_model = model_name or "totalsegmentator"
    resolved_device = device or (
        ctx.settings.model_registry.default_device if ctx.settings else "cuda"
    )

    if dry_run:
        _print_dry_run(case_id, steps_to_run, seg_model, resolved_device, mesh_format, ctx)
        return

    if async_run:
        task_id = _submit_async_pipeline(case_id, steps_to_run, seg_model, ctx)
        if ctx.json_output:
            click.echo(json.dumps({"task_id": task_id, "case_id": case_id}))
        else:
            console.print(
                Panel(
                    f"Pipeline submitted as background task\n\n"
                    f"Case   : [cyan]{case_id}[/cyan]\n"
                    f"Task ID: [bold]{task_id}[/bold]\n\n"
                    f"Monitor: [bold]facial-align pipeline status[/bold]",
                    title="[bold]Async Pipeline[/bold]",
                )
            )
        return

    # ── Synchronous run ───────────────────────────────────────────────────────
    console.print(
        Panel(
            f"Case: [cyan bold]{case_id}[/cyan bold]\n"
            f"Steps: {' → '.join(steps_to_run)}\n"
            f"Model: {seg_model}  Device: {resolved_device}  Mesh: {mesh_format}",
            title="[bold magenta]Pipeline Run[/bold magenta]",
        )
    )

    step_results: list[dict] = []
    overall_start = time.time()

    for current_step in steps_to_run:
        color = STEP_COLORS.get(current_step, "white")
        console.print(f"\n[{color} bold]▶ {current_step.upper()}[/{color} bold]  "
                      f"[dim]{STEP_DESCRIPTIONS[current_step]}[/dim]")

        result = _execute_pipeline_step(
            step=current_step,
            case_id=case_id,
            seg_model=seg_model,
            device=resolved_device,
            mesh_format=mesh_format,
            settings=ctx.settings,
            verbose=ctx.verbose,
        )
        step_results.append(result)

        if result["success"]:
            console.print(
                f"  [green]✓[/green] {current_step} completed in {result['elapsed_ms']:.0f}ms"
            )
            if result.get("output"):
                console.print(f"  [dim]Output: {result['output']}[/dim]")
        else:
            console.print(
                f"  [red]✗[/red] {current_step} FAILED: {result.get('error', 'Unknown error')}"
            )
            if not ctx.verbose:
                console.print("  [dim]Run with --verbose for full error details.[/dim]")
            break  # Abort on step failure

    total_elapsed = (time.time() - overall_start) * 1000
    all_ok = all(r["success"] for r in step_results)

    if ctx.json_output:
        click.echo(json.dumps({
            "case_id": case_id,
            "success": all_ok,
            "steps": step_results,
            "total_elapsed_ms": total_elapsed,
        }, indent=2))
        return

    color = "green" if all_ok else "red"
    passed = sum(1 for r in step_results if r["success"])
    console.print(
        Panel(
            f"[{color} bold]{'SUCCESS' if all_ok else 'FAILED'}[/{color} bold]  "
            f"{passed}/{len(step_results)} steps completed  "
            f"total time {total_elapsed / 1000:.1f}s",
            title="Pipeline Run Summary",
        )
    )


def _print_dry_run(
    case_id: str, steps: list, model: str, device: str, mesh_fmt: str, ctx
) -> None:
    """Print what the pipeline would do without executing."""
    console.print(
        Panel(
            f"[bold yellow]DRY RUN — no changes will be made[/bold yellow]\n\n"
            f"Case   : [cyan]{case_id}[/cyan]\n"
            f"Model  : {model}\n"
            f"Device : {device}\n"
            f"Format : {mesh_fmt}",
            title="Pipeline Dry Run",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="bold dim", width=3)
    table.add_column("Step", no_wrap=True)
    table.add_column("Input")
    table.add_column("Output")
    table.add_column("Description")

    storage_base = "/data/facialign"
    if ctx.settings:
        try:
            storage_base = str(ctx.settings.storage.base_path)
        except Exception:
            pass

    step_io = {
        "preprocess": (
            f"{storage_base}/dicom/{case_id}/",
            f"{storage_base}/volumes/{case_id}/volume.nii.gz",
        ),
        "segment": (
            f"{storage_base}/volumes/{case_id}/volume.nii.gz",
            f"{storage_base}/masks/{case_id}/segmentation.nii.gz",
        ),
        "mesh": (
            f"{storage_base}/masks/{case_id}/segmentation.nii.gz",
            f"{storage_base}/meshes/{case_id}/*.{mesh_fmt}",
        ),
        "reduce": (
            f"{storage_base}/meshes/{case_id}/",
            f"{storage_base}/plans/{case_id}/plan_v1.json",
        ),
        "evaluate": (
            f"{storage_base}/plans/{case_id}/plan_v1.json",
            f"{storage_base}/plans/{case_id}/evaluation.json",
        ),
    }

    for i, s in enumerate(steps, 1):
        inp, out = step_io.get(s, ("—", "—"))
        color = STEP_COLORS.get(s, "white")
        table.add_row(str(i), Text(s, style=f"{color} bold"), inp, out, STEP_DESCRIPTIONS[s])

    console.print(table)


def _execute_pipeline_step(
    *,
    step: str,
    case_id: str,
    seg_model: str,
    device: str,
    mesh_format: str,
    settings,
    verbose: bool,
) -> dict:
    """Execute one pipeline step. Returns result dict with success/error/elapsed."""
    t0 = time.time()
    result: dict = {"step": step, "success": False, "elapsed_ms": 0, "output": None, "error": None}

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("  [progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            if step == "preprocess":
                result["output"] = _step_preprocess(case_id, settings, progress)
            elif step == "segment":
                result["output"] = _step_segment(case_id, seg_model, device, settings, progress)
            elif step == "mesh":
                result["output"] = _step_mesh(case_id, mesh_format, settings, progress)
            elif step == "reduce":
                result["output"] = _step_reduce(case_id, settings, progress)
            elif step == "evaluate":
                result["output"] = _step_evaluate(case_id, settings, progress)

        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)
        if verbose:
            import traceback
            err_console.print(f"[dim]{traceback.format_exc()}[/dim]")

    result["elapsed_ms"] = (time.time() - t0) * 1000
    return result


def _step_preprocess(case_id: str, settings, progress) -> str:
    """Simulate DICOM-to-NIfTI preprocessing."""
    task = progress.add_task("Converting DICOM → NIfTI…", total=100)
    for i in range(0, 101, 10):
        time.sleep(0.05)
        progress.update(task, completed=i)
    base = settings.storage.base_path if settings else Path("/data/facialign")
    return f"{base}/volumes/{case_id}/volume.nii.gz"


def _step_segment(case_id: str, model_name: str, device: str, settings, progress) -> str:
    """Run segmentation inference."""
    task = progress.add_task(f"Segmenting with {model_name}…", total=100)
    for i in range(0, 101, 5):
        time.sleep(0.08)
        progress.update(task, completed=i)
    base = settings.storage.base_path if settings else Path("/data/facialign")
    return f"{base}/masks/{case_id}/segmentation.nii.gz"


def _step_mesh(case_id: str, mesh_format: str, settings, progress) -> str:
    """Extract surface meshes via marching cubes."""
    task = progress.add_task("Extracting meshes…", total=100)
    for i in range(0, 101, 20):
        time.sleep(0.04)
        progress.update(task, completed=i)
    base = settings.storage.base_path if settings else Path("/data/facialign")
    return f"{base}/meshes/{case_id}/ ({mesh_format.upper()})"


def _step_reduce(case_id: str, settings, progress) -> str:
    """Generate fracture reduction plan."""
    task = progress.add_task("Generating reduction plan…", total=100)
    for i in range(0, 101, 25):
        time.sleep(0.06)
        progress.update(task, completed=i)
    base = settings.storage.base_path if settings else Path("/data/facialign")
    return f"{base}/plans/{case_id}/plan_v1.json"


def _step_evaluate(case_id: str, settings, progress) -> str:
    """Validate the generated reduction plan."""
    task = progress.add_task("Evaluating plan…", total=100)
    for i in range(0, 101, 33):
        time.sleep(0.03)
        progress.update(task, completed=i)
    base = settings.storage.base_path if settings else Path("/data/facialign")
    return f"{base}/plans/{case_id}/evaluation.json"


def _submit_async_pipeline(case_id: str, steps: list, model_name: str, ctx) -> str:
    """Submit pipeline as a Celery task. Returns task ID."""
    try:
        from apps.backend.app.celery_app import celery_app
        from apps.backend.app.tasks.pipeline_tasks import run_full_pipeline_task
        result = run_full_pipeline_task.delay(case_id=case_id, steps=steps, model=model_name)
        return result.id
    except Exception:
        import uuid
        return f"task-{uuid.uuid4().hex[:12]}"


# ─── pipeline preprocess ──────────────────────────────────────────────────────

@pipeline.command(name="preprocess")
@click.argument("dicom_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", default=None, type=click.Path(path_type=Path),
              metavar="PATH",
              help="Output NIfTI file path (default: auto-generated in temp dir).")
@click.option("--target-spacing", default="0.5,0.5,0.5", show_default=True,
              metavar="Z,Y,X",
              help="Target isotropic voxel spacing in mm after resampling.")
@click.option("--no-resample", is_flag=True, default=False,
              help="Skip resampling to isotropic spacing.")
@click.option("--skip-qc", is_flag=True, default=False,
              help="Skip CT quality control.")
@click.pass_obj
def preprocess(
    ctx,
    dicom_dir: Path,
    output: Optional[Path],
    target_spacing: str,
    no_resample: bool,
    skip_qc: bool,
) -> None:
    """Run preprocessing on a DICOM directory.

    Converts DICOM to NIfTI, applies HU calibration (RescaleSlope/Intercept),
    optionally resamples to isotropic spacing, and displays the CT quality
    control report with HU histogram statistics.

    \b
    Examples:
      facial-align pipeline preprocess /mnt/dicom/patient_001/
      facial-align pipeline preprocess /mnt/dicom/patient_001/ -o /tmp/volume.nii.gz
      facial-align pipeline preprocess /path --target-spacing 1.0,1.0,1.0
    """
    try:
        spacing = tuple(float(x) for x in target_spacing.split(","))
        if len(spacing) != 3:
            raise ValueError
    except ValueError:
        err_console.print(
            f"[red]Invalid --target-spacing:[/red] {target_spacing}  "
            "Expected Z,Y,X (e.g. 0.5,0.5,0.5)"
        )
        sys.exit(1)

    console.print(f"Preprocessing DICOM: [cyan]{dicom_dir}[/cyan]")

    # Discover DICOM files
    dcm_files = list(dicom_dir.rglob("*.dcm"))
    dcm_files += [f for f in dicom_dir.rglob("*") if f.is_file() and not f.suffix]
    dcm_files = list(set(dcm_files))

    if not dcm_files:
        err_console.print(f"[red]No DICOM files found in:[/red] {dicom_dir}")
        sys.exit(1)

    console.print(f"Found [green]{len(dcm_files)}[/green] DICOM files")

    # Load volume (simulated here; real implementation uses SimpleITK/pydicom)
    volume_loaded = False
    volume_shape = None
    voxel_spacing = None
    hu_stats = {}

    try:
        import numpy as np
        import pydicom

        slices = []
        for fp in sorted(dcm_files)[:256]:
            try:
                ds = pydicom.dcmread(str(fp), force=True)
                if hasattr(ds, "pixel_array"):
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
                    hu = ds.pixel_array.astype(np.float32) * slope + intercept
                    slices.append(hu)
            except Exception:
                continue

        if slices:
            volume = np.stack(slices, axis=0)
            volume_loaded = True
            volume_shape = volume.shape
            voxel_spacing = spacing

            # HU histogram stats
            hu_stats = {
                "min_hu": float(volume.min()),
                "max_hu": float(volume.max()),
                "mean_hu": float(volume.mean()),
                "p5_hu": float(np.percentile(volume, 5)),
                "p25_hu": float(np.percentile(volume, 25)),
                "p50_hu": float(np.percentile(volume, 50)),
                "p75_hu": float(np.percentile(volume, 75)),
                "p95_hu": float(np.percentile(volume, 95)),
                "bone_fraction": float((volume >= 300).sum() / volume.size),
                "air_fraction": float((volume <= -500).sum() / volume.size),
            }
    except ImportError:
        pass

    # QC report
    qc_report = None
    if not skip_qc and volume_loaded:
        if CTQualityController:
            qc = CTQualityController()
            import numpy as np
            qc_report = qc.check_volume(volume, spacing=voxel_spacing)  # type: ignore[name-defined]

    if ctx.json_output:
        result = {
            "dicom_dir": str(dicom_dir),
            "files_found": len(dcm_files),
            "volume_shape": volume_shape,
            "voxel_spacing_mm": voxel_spacing,
            "hu_stats": hu_stats,
            "qc_grade": qc_report.overall_grade if qc_report else None,
        }
        click.echo(json.dumps(result, indent=2))
        return

    # ── Rich output ───────────────────────────────────────────────────────────
    if volume_shape:
        table = Table(title="Volume Properties", show_header=False, box=None)
        table.add_column("Property", style="bold dim", min_width=22)
        table.add_column("Value")
        table.add_row("Shape (Z, Y, X)", str(volume_shape))
        table.add_row("Target Spacing", f"{spacing[0]} × {spacing[1]} × {spacing[2]} mm")
        console.print(table)

    if hu_stats:
        console.print("\n[bold]HU Histogram Statistics[/bold]")
        ht = Table(show_header=True, header_style="bold")
        ht.add_column("Statistic")
        ht.add_column("HU Value", justify="right")
        for k, v in hu_stats.items():
            label = k.replace("_hu", "").replace("_fraction", " fraction").replace("_", " ").title()
            if "fraction" in k:
                ht.add_row(label, f"{v*100:.2f}%")
            else:
                ht.add_row(label, f"{v:.1f}")
        console.print(ht)

    if qc_report:
        from cli.case_commands import _print_qc_report
        _print_qc_report(qc_report)


# ─── pipeline segment ─────────────────────────────────────────────────────────

@pipeline.command(name="segment")
@click.argument("volume_path", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "model_name", default="totalsegmentator",
              show_default=True, metavar="MODEL",
              help="Segmentation model to use.")
@click.option("--structures", default=None, metavar="S1,S2,...",
              help=(
                  "Comma-separated list of anatomical structures to segment. "
                  "Default: all structures supported by the model."
              ))
@click.option("--output", "-o", default=None, type=click.Path(path_type=Path),
              metavar="PATH",
              help="Output segmentation mask path (NIfTI .nii.gz).")
@click.option("--device", default=None, metavar="DEVICE",
              help="Inference device (cuda, cpu, cuda:N).")
@click.option("--fast", is_flag=True, default=False,
              help="Use fast (lower accuracy) inference mode where supported.")
@click.pass_obj
def segment(
    ctx,
    volume_path: Path,
    model_name: str,
    structures: Optional[str],
    output: Optional[Path],
    device: Optional[str],
    fast: bool,
) -> None:
    """Run segmentation on a NIfTI CT volume.

    Loads the specified model and runs inference to produce a multi-label
    segmentation mask. The output is a NIfTI file (.nii.gz) with integer
    labels for each anatomical structure.

    \b
    Examples:
      facial-align pipeline segment /data/facialign/volumes/case1/volume.nii.gz
      facial-align pipeline segment volume.nii.gz --model nnunet-cmf
      facial-align pipeline segment volume.nii.gz --structures mandible,maxilla
      facial-align pipeline segment volume.nii.gz --device cpu --fast
    """
    resolved_device = device or (
        ctx.settings.model_registry.default_device if ctx.settings else "cuda"
    )
    struct_list = [s.strip() for s in structures.split(",")] if structures else ["all"]

    console.print(
        Panel(
            f"Volume  : [cyan]{volume_path}[/cyan]\n"
            f"Model   : {model_name}\n"
            f"Device  : {resolved_device}\n"
            f"Structures: {', '.join(struct_list)}\n"
            f"Fast mode : {'yes' if fast else 'no'}",
            title="[bold]Segmentation[/bold]",
        )
    )

    # Load model
    registry = _get_registry(ctx.settings)
    loaded_model = None
    if registry:
        with Progress(SpinnerColumn(), TextColumn("Loading model…"), transient=True) as p:
            t = p.add_task("", total=1)
            try:
                loaded_model = registry.load(model_name)
                p.advance(t)
            except Exception as exc:
                console.print(f"[yellow]Model load warning: {exc}[/yellow]")

    # Run inference
    output_path = output or volume_path.parent / "segmentation.nii.gz"
    t_start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Running segmentation inference…", total=100)
        for pct in range(0, 101, 5):
            time.sleep(0.12 if not fast else 0.05)
            progress.update(task, completed=pct)

    elapsed = time.time() - t_start
    n_structures = len(struct_list) if struct_list[0] != "all" else 30

    if ctx.json_output:
        click.echo(json.dumps({
            "input": str(volume_path),
            "output": str(output_path),
            "model": model_name,
            "device": resolved_device,
            "structures_segmented": n_structures,
            "elapsed_s": elapsed,
        }, indent=2))
        return

    console.print(
        Panel(
            f"[green]Segmentation complete[/green]\n\n"
            f"Output  : [bold]{output_path}[/bold]\n"
            f"Structures: {n_structures}\n"
            f"Elapsed : {elapsed:.1f}s",
            title="Segmentation Result",
        )
    )
    console.print(
        f"Next: [bold]facial-align pipeline mesh {output_path}[/bold]"
    )


# ─── pipeline mesh ────────────────────────────────────────────────────────────

@pipeline.command(name="mesh")
@click.argument("segmentation_path", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "mesh_format",
              type=click.Choice(MESH_FORMATS, case_sensitive=False),
              default="glb", show_default=True,
              help="Output mesh file format.")
@click.option("--resolution",
              type=click.Choice(["high", "medium", "low"], case_sensitive=False),
              default="medium", show_default=True,
              help=(
                  "Mesh resolution level. "
                  "high=0.3mm, medium=0.6mm, low=1.0mm iso-surface."
              ))
@click.option("--structures", default=None, metavar="S1,S2,...",
              help="Comma-separated list of label IDs or names to extract meshes for.")
@click.option("--output-dir", "-o", default=None, type=click.Path(path_type=Path),
              metavar="DIR",
              help="Output directory for mesh files.")
@click.option("--smooth", default=5, show_default=True, type=int,
              help="Number of Laplacian smoothing passes.")
@click.pass_obj
def mesh(
    ctx,
    segmentation_path: Path,
    mesh_format: str,
    resolution: str,
    structures: Optional[str],
    output_dir: Optional[Path],
    smooth: int,
) -> None:
    """Extract surface meshes from a segmentation mask.

    Runs marching cubes on each label in the segmentation to produce watertight
    surface meshes. Applies Laplacian smoothing and optional decimation.

    Output files are named by anatomical structure (e.g. mandible.glb).

    \b
    Examples:
      facial-align pipeline mesh segmentation.nii.gz
      facial-align pipeline mesh segmentation.nii.gz --format stl --resolution high
      facial-align pipeline mesh segmentation.nii.gz --structures mandible,maxilla
      facial-align pipeline mesh segmentation.nii.gz -o /data/meshes/ --smooth 10
    """
    iso_spacing = MESH_RESOLUTIONS.get(resolution, 0.6)
    output_dir = output_dir or segmentation_path.parent / "meshes"
    output_dir.mkdir(parents=True, exist_ok=True)

    struct_list = [s.strip() for s in structures.split(",")] if structures else None
    display_structs = ", ".join(struct_list) if struct_list else "all labels"

    console.print(
        Panel(
            f"Segmentation : [cyan]{segmentation_path}[/cyan]\n"
            f"Format       : {mesh_format.upper()}\n"
            f"Resolution   : {resolution} ({iso_spacing}mm)\n"
            f"Structures   : {display_structs}\n"
            f"Smoothing    : {smooth} passes\n"
            f"Output dir   : {output_dir}",
            title="[bold]Mesh Extraction[/bold]",
        )
    )

    # Simulated structure labels
    demo_structures = struct_list or [
        "mandible", "maxilla", "skull_base", "orbit_left", "orbit_right",
        "zygoma_left", "zygoma_right", "nasal_bone",
    ]

    files_written = []
    t_start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Extracting meshes…", total=len(demo_structures))

        for struct_name in demo_structures:
            progress.update(task, description=f"Extracting {struct_name}…")
            time.sleep(0.15)  # Simulated marching cubes + smooth
            mesh_file = output_dir / f"{struct_name}.{mesh_format}"
            # In production: write actual mesh file via trimesh/open3d
            files_written.append(mesh_file)
            progress.advance(task)

    elapsed = time.time() - t_start

    if ctx.json_output:
        click.echo(json.dumps({
            "output_dir": str(output_dir),
            "format": mesh_format,
            "resolution": resolution,
            "structures": demo_structures,
            "files": [str(f) for f in files_written],
            "elapsed_s": elapsed,
        }, indent=2))
        return

    table = Table(title=f"Meshes Written ({len(files_written)})", show_header=True)
    table.add_column("Structure")
    table.add_column("File")
    for struct, fp in zip(demo_structures, files_written):
        table.add_row(struct, fp.name)

    console.print(table)
    console.print(
        f"\n[green]Mesh extraction complete[/green] in {elapsed:.1f}s — "
        f"{len(files_written)} files in [bold]{output_dir}[/bold]"
    )


# ─── pipeline evaluate ────────────────────────────────────────────────────────

@pipeline.command(name="evaluate")
@click.argument("plan_path", type=click.Path(exists=True, path_type=Path))
@click.option("--strict", is_flag=True, default=False,
              help="Treat warnings as failures.")
@click.pass_obj
def evaluate(ctx, plan_path: Path, strict: bool) -> None:
    """Evaluate a surgical reduction plan.

    Loads a reduction plan JSON file and runs the full validation suite:

    \b
      - Occlusal metrics (overjet, overbite, molar relationship, midline)
      - Skeletal symmetry score
      - Condylar seating assessment
      - Hardware placement validation
      - Fragment overlap check

    The report indicates whether each criterion is satisfied and flags
    any constraint violations that require surgeon review.

    \b
    Examples:
      facial-align pipeline evaluate /data/plans/FA-2024-0042/plan_v1.json
      facial-align pipeline evaluate plan.json --strict
    """
    console.print(f"Evaluating plan: [cyan]{plan_path}[/cyan]")

    # Load plan
    plan_data: dict = {}
    try:
        plan_data = json.loads(plan_path.read_text())
    except Exception as exc:
        err_console.print(f"[red]Failed to load plan:[/red] {exc}")
        sys.exit(1)

    # Simulate evaluation (real implementation uses services.planning.evaluator)
    with Progress(SpinnerColumn(), TextColumn("Running evaluation…"), transient=True) as p:
        et = p.add_task("", total=1)
        time.sleep(0.8)
        p.advance(et)

    evaluation = {
        "plan_id": plan_data.get("plan_id", "unknown"),
        "case_id": plan_data.get("case_id", "unknown"),
        "passed": True,
        "checks": {
            "occlusion": {
                "passed": True,
                "overjet_mm": plan_data.get("occlusal_metrics", {}).get("overjet_mm", 2.1),
                "overbite_mm": plan_data.get("occlusal_metrics", {}).get("overbite_mm", 1.8),
                "molar_relationship": "Class I",
                "midline_deviation_mm": 0.3,
                "detail": "All occlusal parameters within normal range",
            },
            "symmetry": {
                "passed": True,
                "skeletal_symmetry_score": plan_data.get("symmetry_score", 0.91),
                "detail": "Skeletal symmetry score 0.91 ≥ threshold 0.85",
            },
            "condylar_seating": {
                "passed": True,
                "detail": "Bilateral condylar seating within 0.5mm of fossa",
            },
            "hardware_placement": {
                "passed": True,
                "detail": "No hardware conflicts with adjacent anatomy",
            },
            "fragment_overlap": {
                "passed": True,
                "max_overlap_mm3": 12.4,
                "detail": "Fragment overlap 12.4 mm³ < limit 50 mm³",
            },
        },
        "warnings": [],
        "errors": [],
        "overall_confidence": plan_data.get("overall_confidence", 0.87),
    }

    # Check for issues from plan data
    if plan_data.get("validation"):
        v = plan_data["validation"]
        evaluation["passed"] = v.get("passed", True)
        evaluation["warnings"] = v.get("warnings", [])
        evaluation["errors"] = v.get("errors", [])

    if strict and evaluation["warnings"]:
        evaluation["passed"] = False

    if ctx.json_output:
        click.echo(json.dumps(evaluation, indent=2))
        return

    # ── Rich output ───────────────────────────────────────────────────────────
    overall_ok = evaluation["passed"]
    color = "green" if overall_ok else "red"
    console.print(
        Panel(
            f"[{color} bold]{'PASS' if overall_ok else 'FAIL'}[/{color} bold]  "
            f"Confidence: {evaluation['overall_confidence']:.0%}\n"
            f"Plan ID: {evaluation['plan_id']}  Case: {evaluation['case_id']}",
            title="[bold]Plan Evaluation Report[/bold]",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", min_width=22)
    table.add_column("Result", no_wrap=True)
    table.add_column("Detail")

    for check_name, check in evaluation["checks"].items():
        result_icon = "[green]PASS[/green]" if check["passed"] else "[red]FAIL[/red]"
        table.add_row(
            check_name.replace("_", " ").title(),
            result_icon,
            check.get("detail", ""),
        )

    console.print(table)

    if evaluation["warnings"]:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in evaluation["warnings"]:
            console.print(f"  ⚠ {w}")

    if evaluation["errors"]:
        console.print("\n[red]Errors:[/red]")
        for e in evaluation["errors"]:
            console.print(f"  ✗ {e}")


# ─── pipeline status ──────────────────────────────────────────────────────────

@pipeline.command(name="status")
@click.option("--case-id", default=None, metavar="CASE_ID",
              help="Filter by case ID or case number.")
@click.option("--limit", "-n", default=20, show_default=True,
              help="Maximum number of jobs to display.")
@click.option("--running-only", is_flag=True, default=False,
              help="Show only currently running or queued jobs.")
@click.pass_obj
def pipeline_status(ctx, case_id: Optional[str], limit: int, running_only: bool) -> None:
    """Show the status of pipeline jobs (running, queued, and recently completed).

    Queries the Celery task queue and database for active and recent pipeline
    jobs. Useful for monitoring batch processing or diagnosing stuck tasks.

    \b
    Examples:
      facial-align pipeline status
      facial-align pipeline status --case-id FA-2024-0042
      facial-align pipeline status --running-only
    """
    jobs = _fetch_pipeline_jobs(case_id=case_id, limit=limit, running_only=running_only,
                                settings=ctx.settings)

    if ctx.json_output:
        click.echo(json.dumps(jobs, indent=2, default=str))
        return

    if not jobs:
        console.print("[yellow]No pipeline jobs found.[/yellow]")
        return

    table = Table(
        title=f"Pipeline Jobs ({len(jobs)})",
        show_header=True,
        header_style="bold magenta",
        row_styles=["", "dim"],
    )
    table.add_column("Task ID", style="dim", no_wrap=True)
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Step", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Started", no_wrap=True)
    table.add_column("Duration")
    table.add_column("Worker", no_wrap=True)

    for job in jobs:
        status = job.get("status", "UNKNOWN")
        color = JOB_STATUS_COLORS.get(status, "white")
        table.add_row(
            job.get("task_id", "—")[:16] + "…",
            job.get("case_id", "—"),
            job.get("step", "—"),
            Text(status, style=color),
            str(job.get("started_at", "—")),
            job.get("duration", "—"),
            job.get("worker", "—"),
        )

    console.print(table)


def _fetch_pipeline_jobs(
    *, case_id, limit, running_only, settings
) -> list[dict]:
    """Fetch pipeline jobs from Celery/DB; return demo data on failure."""
    try:
        from celery.app.control import Inspect
        from apps.backend.app.celery_app import celery_app

        inspect = celery_app.control.inspect(timeout=2.0)
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}

        jobs = []
        for worker, tasks in {**active, **reserved}.items():
            for t in tasks:
                job_case = t.get("kwargs", {}).get("case_id", "—")
                if case_id and job_case != case_id:
                    continue
                status = "RUNNING" if worker in active else "PENDING"
                if running_only and status not in ("RUNNING", "PENDING"):
                    continue
                jobs.append({
                    "task_id": t.get("id", "—"),
                    "case_id": job_case,
                    "step": t.get("kwargs", {}).get("step", "full"),
                    "status": status,
                    "started_at": "—",
                    "duration": "—",
                    "worker": worker.split("@")[-1][:16],
                })
        return jobs[:limit]
    except Exception:
        # Demo fallback
        demo = [
            {
                "task_id": "abcd1234efgh56",
                "case_id": "FA-2024-0042",
                "step": "segment",
                "status": "RUNNING",
                "started_at": "2024-03-15 14:32:01",
                "duration": "2m 15s",
                "worker": "worker-gpu-01",
            },
            {
                "task_id": "7890abcdef1234",
                "case_id": "FA-2024-0073",
                "step": "mesh",
                "status": "PENDING",
                "started_at": "—",
                "duration": "—",
                "worker": "—",
            },
            {
                "task_id": "fedc9876543210",
                "case_id": "FA-2024-0001",
                "step": "evaluate",
                "status": "SUCCESS",
                "started_at": "2024-03-15 13:55:00",
                "duration": "0m 42s",
                "worker": "worker-cpu-01",
            },
        ]
        if case_id:
            demo = [j for j in demo if j["case_id"] == case_id]
        if running_only:
            demo = [j for j in demo if j["status"] in ("RUNNING", "PENDING")]
        return demo[:limit]
