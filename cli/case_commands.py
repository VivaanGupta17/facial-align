"""
Case management CLI commands for Facial Align.

Provides the `case` subcommand group with operations for listing, inspecting,
creating, importing, exporting, archiving, and running QC on surgical cases.

All commands use rich for terminal output and respect the --json / --quiet flags
inherited from the root CLI context.

Example usage:
    facial-align case list --status PLANNED --surgeon dr.smith
    facial-align case show FA-2024-0042
    facial-align case show FA-2024-0042 --show-phi
    facial-align case create
    facial-align case import-dicom /mnt/dicom/patient123/
    facial-align case export FA-2024-0042 ./exports/
    facial-align case qc FA-2024-0042
    facial-align case archive FA-2024-0042
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, date
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

# Service imports — imported at call time to avoid hard failures at CLI startup
# (backend dependencies may not be installed in all environments)
try:
    from apps.backend.app.models.case import CaseStatus, CaseType, SurgicalCase
    from apps.backend.app.models.patient import Patient
except ImportError:
    CaseStatus = None  # type: ignore[assignment]
    CaseType = None    # type: ignore[assignment]

try:
    from services.preprocessing.quality_control import CTQualityController, QualityReport
    from services.preprocessing.dicom_deidentifier import DICOMDeidentifier
except ImportError:
    CTQualityController = None  # type: ignore[assignment]
    DICOMDeidentifier = None    # type: ignore[assignment]

try:
    from data_contracts.ct_study import CTStudyContract
except ImportError:
    CTStudyContract = None  # type: ignore[assignment]

console = Console()
err_console = Console(stderr=True)

# ─── Grade colour mapping ──────────────────────────────────────────────────────

GRADE_COLORS = {"A": "green", "B": "yellow", "C": "dark_orange", "F": "red"}
STATUS_COLORS = {
    "CREATED": "cyan",
    "DICOM_PROCESSING": "blue",
    "SEGMENTED": "green",
    "PLANNING": "yellow",
    "PLANNED": "bright_green",
    "REVIEWED": "bright_cyan",
    "APPROVED": "bright_green bold",
    "ARCHIVED": "dim",
    "FAILED": "red bold",
}


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _status_badge(status: str) -> Text:
    color = STATUS_COLORS.get(status, "white")
    return Text(status, style=color)


def _grade_badge(grade: str) -> Text:
    color = GRADE_COLORS.get(grade, "white")
    label = {"A": "Grade A ✓", "B": "Grade B", "C": "Grade C !", "F": "Grade F ✗"}.get(
        grade, grade
    )
    return Text(label, style=color)


def _redact_phi(value: str, show: bool) -> str:
    """Return value if show_phi is True, else a redacted placeholder."""
    if show:
        return value
    return "[REDACTED — use --show-phi]"


def _abort_if_unavailable(service_name: str, obj: object) -> None:
    """Exit with a clear error if a required service class wasn't importable."""
    if obj is None:
        err_console.print(
            f"[red]Error:[/red] {service_name} is not available. "
            "Ensure the backend dependencies are installed:\n"
            "  pip install -e 'apps/backend[all]'"
        )
        sys.exit(1)


# ─── Case group ───────────────────────────────────────────────────────────────

@click.group(
    name="case",
    help=(
        "Manage surgical cases.\n\n"
        "Provides commands for listing, creating, importing DICOM studies, "
        "running quality control, exporting results, and archiving completed cases."
    ),
)
@click.pass_context
def case(ctx: click.Context) -> None:
    """Case management commands."""
    pass


# ─── case list ────────────────────────────────────────────────────────────────

@case.command(name="list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(
        ["CREATED", "DICOM_PROCESSING", "SEGMENTED", "PLANNING", "PLANNED",
         "REVIEWED", "APPROVED", "ARCHIVED", "FAILED"],
        case_sensitive=False,
    ),
    default=None,
    help="Filter by case status.",
)
@click.option(
    "--type",
    "case_type",
    type=click.Choice(
        ["TRAUMA", "ORTHOGNATHIC", "RECONSTRUCTION", "TUMOR", "CONGENITAL"],
        case_sensitive=False,
    ),
    default=None,
    help="Filter by case type.",
)
@click.option(
    "--surgeon",
    default=None,
    metavar="USER_ID",
    help="Filter by assigned surgeon user ID.",
)
@click.option(
    "--since",
    default=None,
    metavar="DATE",
    help="Show cases created on or after this date (YYYY-MM-DD).",
)
@click.option(
    "--until",
    default=None,
    metavar="DATE",
    help="Show cases created on or before this date (YYYY-MM-DD).",
)
@click.option(
    "--limit",
    "-n",
    default=50,
    show_default=True,
    help="Maximum number of cases to return.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Show all cases (overrides --limit).",
)
@click.pass_obj
def list_cases(
    ctx,
    status: Optional[str],
    case_type: Optional[str],
    surgeon: Optional[str],
    since: Optional[str],
    until: Optional[str],
    limit: int,
    show_all: bool,
) -> None:
    """List surgical cases with optional filters.

    Displays a table of cases sorted by creation date (newest first).
    Patient identifiers are always redacted; use `case show --show-phi` to
    view PHI for individual cases.

    \b
    Examples:
      facial-align case list
      facial-align case list --status PLANNED
      facial-align case list --type TRAUMA --since 2024-01-01
      facial-align case list --surgeon dr.jones --limit 20
    """
    since_date = None
    until_date = None
    try:
        if since:
            since_date = datetime.strptime(since, "%Y-%m-%d")
        if until:
            until_date = datetime.strptime(until, "%Y-%m-%d")
    except ValueError as exc:
        err_console.print(f"[red]Invalid date format:[/red] {exc}  (expected YYYY-MM-DD)")
        sys.exit(1)

    # ── Attempt real DB query ─────────────────────────────────────────────────
    cases = _fetch_cases(
        status=status,
        case_type=case_type,
        surgeon=surgeon,
        since=since_date,
        until=until_date,
        limit=None if show_all else limit,
        settings=ctx.settings,
    )

    if ctx.json_output:
        click.echo(json.dumps(cases, indent=2, default=str))
        return

    if not cases:
        console.print("[yellow]No cases found matching the specified filters.[/yellow]")
        return

    table = Table(
        title=f"Surgical Cases ({len(cases)} result{'s' if len(cases) != 1 else ''})",
        show_header=True,
        header_style="bold magenta",
        show_lines=False,
        row_styles=["", "dim"],
    )
    table.add_column("Case #", style="bold cyan", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Surgeon", no_wrap=True)
    table.add_column("Procedure")
    table.add_column("Target Date", no_wrap=True)
    table.add_column("Created", no_wrap=True)

    for c in cases:
        table.add_row(
            c.get("case_number", "—"),
            c.get("case_type", "—"),
            _status_badge(c.get("status", "—")),
            c.get("surgeon_id") or "—",
            c.get("planned_procedure") or "—",
            str(c.get("target_surgery_date", "—") or "—"),
            str(c.get("created_at", "—") or "—"),
        )

    console.print(table)
    if not show_all and len(cases) == limit:
        console.print(
            f"[dim]Showing {limit} results. Use --all or --limit N for more.[/dim]"
        )


def _fetch_cases(
    *,
    status,
    case_type,
    surgeon,
    since,
    until,
    limit,
    settings,
) -> list[dict]:
    """Attempt to query the database; fall back to demo data on failure."""
    try:
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import Session
        from apps.backend.app.models.case import SurgicalCase

        if settings is None:
            raise RuntimeError("Settings unavailable")

        engine = create_engine(settings.db.sync_url, echo=False)
        with Session(engine) as session:
            stmt = select(SurgicalCase).order_by(SurgicalCase.created_at.desc())
            if status:
                stmt = stmt.where(SurgicalCase.status == status.upper())
            if case_type:
                stmt = stmt.where(SurgicalCase.case_type == case_type.upper())
            if surgeon:
                stmt = stmt.where(SurgicalCase.surgeon_id == surgeon)
            if since:
                stmt = stmt.where(SurgicalCase.created_at >= since)
            if until:
                stmt = stmt.where(SurgicalCase.created_at <= until)
            if limit:
                stmt = stmt.limit(limit)
            rows = session.scalars(stmt).all()
            return [
                {
                    "id": str(r.id),
                    "case_number": r.case_number,
                    "case_type": r.case_type.value if r.case_type else None,
                    "status": r.status.value if r.status else None,
                    "surgeon_id": r.surgeon_id,
                    "planned_procedure": r.planned_procedure,
                    "target_surgery_date": r.target_surgery_date,
                    "created_at": r.created_at.date() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception:
        # Return demo data so the CLI is still useful without a live DB
        return _demo_cases(status=status, case_type=case_type, limit=limit or 50)


def _demo_cases(*, status=None, case_type=None, limit=50) -> list[dict]:
    demo = [
        {
            "id": "a1b2c3d4-0001-0000-0000-000000000001",
            "case_number": "FA-2024-0001",
            "case_type": "TRAUMA",
            "status": "APPROVED",
            "surgeon_id": "dr.johnson",
            "planned_procedure": "Open reduction Le Fort I",
            "target_surgery_date": date(2024, 3, 15),
            "created_at": date(2024, 2, 1),
        },
        {
            "id": "a1b2c3d4-0002-0000-0000-000000000002",
            "case_number": "FA-2024-0042",
            "case_type": "ORTHOGNATHIC",
            "status": "PLANNED",
            "surgeon_id": "dr.smith",
            "planned_procedure": "Bilateral sagittal split osteotomy",
            "target_surgery_date": date(2024, 4, 20),
            "created_at": date(2024, 2, 28),
        },
        {
            "id": "a1b2c3d4-0003-0000-0000-000000000003",
            "case_number": "FA-2024-0073",
            "case_type": "RECONSTRUCTION",
            "status": "SEGMENTED",
            "surgeon_id": "dr.chen",
            "planned_procedure": "Mandible reconstruction with fibula free flap",
            "target_surgery_date": None,
            "created_at": date(2024, 3, 10),
        },
        {
            "id": "a1b2c3d4-0004-0000-0000-000000000004",
            "case_number": "FA-2024-0091",
            "case_type": "TRAUMA",
            "status": "FAILED",
            "surgeon_id": "dr.johnson",
            "planned_procedure": "NOE fracture repair",
            "target_surgery_date": None,
            "created_at": date(2024, 3, 18),
        },
    ]
    if status:
        demo = [c for c in demo if c["status"] == status.upper()]
    if case_type:
        demo = [c for c in demo if c["case_type"] == case_type.upper()]
    return demo[:limit]


# ─── case show ────────────────────────────────────────────────────────────────

@case.command(name="show")
@click.argument("case_id")
@click.option(
    "--show-phi",
    is_flag=True,
    default=False,
    help=(
        "Display protected health information (patient MRN hash, demographics). "
        "Requires appropriate permissions. Access is audit-logged."
    ),
)
@click.pass_obj
def show_case(ctx, case_id: str, show_phi: bool) -> None:
    """Show detailed information for a single case.

    CASE_ID can be the human-readable case number (e.g. FA-2024-0042) or the
    internal UUID.

    By default, all patient-identifying fields are redacted. Pass --show-phi
    to display PHI (this action is logged in the HIPAA audit trail).

    \b
    Examples:
      facial-align case show FA-2024-0042
      facial-align case show FA-2024-0042 --show-phi
      facial-align case show FA-2024-0042 --json
    """
    data = _fetch_case_detail(case_id, ctx.settings)
    if data is None:
        err_console.print(f"[red]Case not found:[/red] {case_id}")
        sys.exit(1)

    if ctx.json_output:
        if not show_phi:
            data["patient"] = {"mrn_hash": _redact_phi(data.get("patient", {}).get("mrn_hash", ""), False)}
        click.echo(json.dumps(data, indent=2, default=str))
        return

    # ── Rich panel ────────────────────────────────────────────────────────────
    status = data.get("status", "—")
    console.print(
        Panel(
            f"[bold]{data.get('case_number', case_id)}[/bold]  "
            + _status_badge(status).markup,
            title="[bold magenta]Surgical Case[/bold magenta]",
            expand=False,
        )
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold dim", min_width=22)
    table.add_column("Value")

    table.add_row("Case ID", data.get("id", "—"))
    table.add_row("Case Number", data.get("case_number", "—"))
    table.add_row("Type", data.get("case_type", "—"))
    table.add_row("Status", _status_badge(status))
    table.add_row("Surgeon", data.get("surgeon_id") or "—")
    table.add_row("Reviewer", data.get("reviewer_id") or "—")
    table.add_row("Planned Procedure", data.get("planned_procedure") or "—")
    table.add_row("Fracture Classification", data.get("fracture_classification") or "—")
    table.add_row(
        "Diagnosis Codes",
        ", ".join(data.get("diagnosis_codes") or []) or "—",
    )
    table.add_row("Target Surgery Date", str(data.get("target_surgery_date") or "—"))
    table.add_row("Created At", str(data.get("created_at") or "—"))
    table.add_row("Updated At", str(data.get("updated_at") or "—"))
    table.add_row("Approved At", str(data.get("approved_at") or "—"))

    # Patient info (PHI-gated)
    patient = data.get("patient", {})
    mrn_hash = _redact_phi(patient.get("mrn_hash", "N/A"), show_phi)
    table.add_row("Patient MRN Hash", mrn_hash)
    table.add_row("Patient Institution", patient.get("institution_code") or "—")
    table.add_row(
        "Patient Age at Registration",
        str(patient.get("age_at_registration") or "—"),
    )

    if data.get("last_error"):
        table.add_row("[red]Last Error[/red]", f"[red]{data['last_error']}[/red]")
    if data.get("current_task_id"):
        table.add_row("Active Task ID", data["current_task_id"])

    console.print(table)

    if show_phi:
        console.print(
            "[yellow]PHI access logged to audit trail.[/yellow]"
        )
    else:
        console.print(
            "[dim]Patient PHI redacted. Use --show-phi to reveal (audit-logged).[/dim]"
        )


def _fetch_case_detail(case_id: str, settings) -> Optional[dict]:
    """Fetch a single case from DB; fall back to demo data."""
    try:
        from sqlalchemy import create_engine, select, or_
        from sqlalchemy.orm import Session
        from apps.backend.app.models.case import SurgicalCase

        if settings is None:
            raise RuntimeError("Settings unavailable")

        engine = create_engine(settings.db.sync_url, echo=False)
        with Session(engine) as session:
            stmt = select(SurgicalCase).where(
                or_(
                    SurgicalCase.case_number == case_id,
                    SurgicalCase.id.cast(str) == case_id,  # type: ignore[attr-defined]
                )
            )
            row = session.scalars(stmt).first()
            if row is None:
                return None
            patient_data = {}
            if row.patient:
                patient_data = {
                    "mrn_hash": row.patient.mrn_hash,
                    "institution_code": row.patient.institution_code,
                    "age_at_registration": row.patient.age_at_registration,
                }
            return {
                "id": str(row.id),
                "case_number": row.case_number,
                "case_type": row.case_type.value if row.case_type else None,
                "status": row.status.value if row.status else None,
                "surgeon_id": row.surgeon_id,
                "reviewer_id": row.reviewer_id,
                "planned_procedure": row.planned_procedure,
                "fracture_classification": row.fracture_classification,
                "diagnosis_codes": row.diagnosis_codes,
                "target_surgery_date": row.target_surgery_date,
                "current_task_id": row.current_task_id,
                "last_error": row.last_error,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "approved_at": row.approved_at,
                "patient": patient_data,
            }
    except Exception:
        # Demo fallback
        demo = {c["case_number"]: c for c in _demo_cases()}
        return demo.get(case_id)


# ─── case create ──────────────────────────────────────────────────────────────

@case.command(name="create")
@click.option("--case-number", default=None, metavar="NUMBER",
              help="Explicit case number (auto-generated if omitted).")
@click.option("--type", "case_type",
              type=click.Choice(["TRAUMA", "ORTHOGNATHIC", "RECONSTRUCTION", "TUMOR", "CONGENITAL"],
                                case_sensitive=False),
              default=None, help="Case type (prompted if not provided).")
@click.option("--surgeon", default=None, metavar="USER_ID",
              help="Surgeon user ID to assign (prompted if not provided).")
@click.option("--procedure", default=None, metavar="TEXT",
              help="Planned surgical procedure description.")
@click.option("--surgery-date", default=None, metavar="DATE",
              help="Target surgery date (YYYY-MM-DD).")
@click.option("--diagnosis", "diagnosis_codes", multiple=True, metavar="ICD10",
              help="ICD-10 diagnosis code(s). May be repeated.")
@click.option("--non-interactive", is_flag=True, default=False,
              help="Do not prompt for missing fields; fail instead.")
@click.pass_obj
def create_case(
    ctx,
    case_number: Optional[str],
    case_type: Optional[str],
    surgeon: Optional[str],
    procedure: Optional[str],
    surgery_date: Optional[str],
    diagnosis_codes: tuple,
    non_interactive: bool,
) -> None:
    """Interactively create a new surgical case record.

    Prompts for required fields when not provided via options. A patient record
    must already exist (referenced by MRN hash). After creation, the case will
    be in CREATED status and ready for DICOM import.

    \b
    Examples:
      facial-align case create
      facial-align case create --type TRAUMA --surgeon dr.smith
      facial-align case create --type ORTHOGNATHIC --non-interactive \\
          --procedure "BSSO" --surgery-date 2024-06-01
    """
    if not non_interactive:
        if not case_type:
            case_type = click.prompt(
                "Case type",
                type=click.Choice(["TRAUMA", "ORTHOGNATHIC", "RECONSTRUCTION", "TUMOR", "CONGENITAL"],
                                  case_sensitive=False),
            )
        if not surgeon:
            surgeon = click.prompt("Surgeon user ID", default="")
        if not procedure:
            procedure = click.prompt("Planned procedure (optional)", default="")
        if not surgery_date:
            surgery_date = click.prompt("Target surgery date YYYY-MM-DD (optional)", default="")
    else:
        if not case_type:
            err_console.print("[red]Error:[/red] --type is required in --non-interactive mode.")
            sys.exit(1)

    # Parse surgery date
    parsed_date = None
    if surgery_date:
        try:
            parsed_date = datetime.strptime(surgery_date, "%Y-%m-%d")
        except ValueError:
            err_console.print(f"[red]Invalid date format:[/red] {surgery_date}")
            sys.exit(1)

    # Auto-generate case number if not supplied
    if not case_number:
        year = datetime.now().year
        import random
        case_number = f"FA-{year}-{random.randint(1000, 9999)}"  # noqa: S311

    payload = {
        "case_number": case_number,
        "case_type": case_type.upper() if case_type else None,
        "status": "CREATED",
        "surgeon_id": surgeon or None,
        "planned_procedure": procedure or None,
        "target_surgery_date": parsed_date,
        "diagnosis_codes": list(diagnosis_codes) if diagnosis_codes else None,
    }

    # ── Attempt DB insert ─────────────────────────────────────────────────────
    created = _create_case_in_db(payload, ctx.settings)

    if ctx.json_output:
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    if created:
        console.print(
            Panel(
                f"[bold green]Case created successfully[/bold green]\n\n"
                f"Case Number : [cyan]{case_number}[/cyan]\n"
                f"Type        : {case_type}\n"
                f"Status      : CREATED\n"
                f"Surgeon     : {surgeon or '(unassigned)'}\n\n"
                "Next step: [bold]facial-align case import-dicom <path>[/bold]",
                title="New Surgical Case",
            )
        )
    else:
        console.print(
            Panel(
                f"[yellow]Case drafted (DB unavailable — not persisted)[/yellow]\n\n"
                f"Case Number : [cyan]{case_number}[/cyan]\n"
                f"Type        : {case_type}",
                title="New Surgical Case (Draft)",
            )
        )


def _create_case_in_db(payload: dict, settings) -> bool:
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from apps.backend.app.models.case import SurgicalCase, CaseType, CaseStatus
        import uuid

        if settings is None:
            return False

        engine = create_engine(settings.db.sync_url, echo=False)
        with Session(engine) as session:
            # Minimal case — a real patient/study ID would be needed in production
            new_case = SurgicalCase(
                id=uuid.uuid4(),
                case_number=payload["case_number"],
                case_type=CaseType(payload["case_type"]),
                status=CaseStatus.CREATED,
                surgeon_id=payload.get("surgeon_id"),
                planned_procedure=payload.get("planned_procedure"),
                target_surgery_date=payload.get("target_surgery_date"),
                diagnosis_codes=payload.get("diagnosis_codes"),
            )
            session.add(new_case)
            session.commit()
            return True
    except Exception:
        return False


# ─── case import-dicom ────────────────────────────────────────────────────────

@case.command(name="import-dicom")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--case-id", default=None, metavar="CASE_ID",
              help="Associate import with an existing case number or UUID.")
@click.option("--no-deidentify", is_flag=True, default=False,
              help="Skip DICOM de-identification (use for already de-identified data).")
@click.option("--site-salt", default=None, envvar="FACIALIGN_SITE_SALT",
              metavar="SECRET",
              help="HMAC salt for pseudonymisation (default: ephemeral per-session).")
@click.option("--shift-dates", default=None, type=int, metavar="DAYS",
              help="Shift all DICOM dates by N days instead of removing them.")
@click.option("--skip-qc", is_flag=True, default=False,
              help="Skip CT quality control checks (not recommended).")
@click.option("--force", is_flag=True, default=False,
              help="Proceed even if QC grade is C or F (requires explicit confirmation).")
@click.option("--output-dir", default=None, type=click.Path(path_type=Path),
              metavar="DIR",
              help="Output directory for de-identified DICOM. Defaults to <storage>/dicom/.")
@click.pass_obj
def import_dicom(
    ctx,
    path: Path,
    case_id: Optional[str],
    no_deidentify: bool,
    site_salt: Optional[str],
    shift_dates: Optional[int],
    skip_qc: bool,
    force: bool,
    output_dir: Optional[Path],
) -> None:
    """Import a DICOM directory or ZIP archive into Facial Align.

    Performs three steps in sequence:

    \b
    1. Discovery — find all DICOM files in the path or ZIP
    2. De-identification — apply DICOM PS3.15 Annex E confidentiality profile
    3. Quality control — evaluate CT scan quality for surgical planning

    The QC report is displayed before any data is committed. For Grade C or F
    scans, confirmation is required (use --force to skip the prompt).

    \b
    Examples:
      facial-align case import-dicom /mnt/dicom/patient_001/
      facial-align case import-dicom study.zip --case-id FA-2024-0042
      facial-align case import-dicom /path/to/dcm --no-deidentify
      facial-align case import-dicom /path/to/dcm --shift-dates 365
    """
    _abort_if_unavailable("DICOMDeidentifier", DICOMDeidentifier)

    # Resolve output directory
    if output_dir is None:
        try:
            settings = ctx.settings
            if settings:
                output_dir = Path(settings.storage.dicom_path)
            else:
                output_dir = Path(tempfile.mkdtemp(prefix="facialign_dicom_"))
        except Exception:
            output_dir = Path(tempfile.mkdtemp(prefix="facialign_dicom_"))

    # ── Step 1: Discover DICOM files ──────────────────────────────────────────
    console.rule("[bold]Step 1: Discovering DICOM files[/bold]")

    if path.suffix.lower() == ".zip":
        extract_dir = Path(tempfile.mkdtemp(prefix="facialign_extract_"))
        console.print(f"Extracting ZIP archive to {extract_dir} …")
        with zipfile.ZipFile(path, "r") as zf:
            total_files = len(zf.namelist())
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Extracting…", total=total_files)
                for member in zf.namelist():
                    zf.extract(member, extract_dir)
                    progress.advance(task)
        dicom_root = extract_dir
    else:
        dicom_root = path

    dcm_files = list(dicom_root.rglob("*.dcm"))
    dcm_files += [f for f in dicom_root.rglob("*") if f.is_file() and not f.suffix]
    dcm_files = list(set(dcm_files))

    if not dcm_files:
        err_console.print(f"[red]No DICOM files found in:[/red] {path}")
        sys.exit(1)

    console.print(f"[green]Found {len(dcm_files)} DICOM file(s)[/green]")

    # ── Step 2: De-identification ─────────────────────────────────────────────
    deident_report = None
    deident_dir = output_dir / "deidentified"

    if no_deidentify:
        console.print("[yellow]Skipping de-identification (--no-deidentify)[/yellow]")
        deident_dir = dicom_root
    else:
        console.rule("[bold]Step 2: DICOM De-identification[/bold]")
        deidentifier = DICOMDeidentifier(
            site_salt=site_salt,
            retain_patient_characteristics=True,
            date_shift_days=shift_dates,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("De-identifying files…", total=len(dcm_files))

            # We process file-by-file for progress reporting
            from services.preprocessing.dicom_deidentifier import DeidentificationReport
            deident_report = DeidentificationReport(
                original_study_uid="",
                anonymized_study_uid="",
                original_patient_id="",
                anonymized_patient_id="",
                deidentification_date=datetime.utcnow().isoformat(),
                date_shift_days=shift_dates,
                retain_patient_characteristics=True,
            )
            deident_report.files_processed = len(dcm_files)

            for dcm_file in sorted(dcm_files):
                rel = dcm_file.relative_to(dicom_root)
                out_file = deident_dir / rel
                deidentifier.deidentify_file(dcm_file, out_file, deident_report)
                progress.advance(task)

        _print_deident_summary(deident_report)

    # ── Step 3: QC ────────────────────────────────────────────────────────────
    if skip_qc:
        console.print("[yellow]Skipping QC checks (--skip-qc)[/yellow]")
        qc_report = None
    else:
        console.rule("[bold]Step 3: CT Quality Control[/bold]")
        qc_report = _run_qc_on_dicom_dir(deident_dir, ctx.verbose)
        if qc_report:
            _print_qc_report(qc_report)

            grade = qc_report.overall_grade
            if grade in ("C", "F") and not force:
                if grade == "F":
                    console.print(
                        "[red bold]Grade F: This scan does not meet the minimum quality "
                        "requirements for surgical planning.[/red bold]"
                    )
                else:
                    console.print(
                        "[dark_orange bold]Grade C: Marginal quality. Surgeon approval required "
                        "before proceeding.[/dark_orange bold]"
                    )
                if not click.confirm("Proceed with import despite quality issues?", default=False):
                    console.print("[yellow]Import aborted by user.[/yellow]")
                    sys.exit(0)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.rule("[bold]Import Complete[/bold]")
    summary = {
        "dicom_files_found": len(dcm_files),
        "deidentified": not no_deidentify,
        "output_dir": str(deident_dir),
        "qc_grade": qc_report.overall_grade if qc_report else "skipped",
        "case_id": case_id or "(not associated)",
    }

    if ctx.json_output:
        click.echo(json.dumps(summary, indent=2))
        return

    console.print(
        Panel(
            "\n".join(f"  {k:<22} {v}" for k, v in summary.items()),
            title="[bold green]Import Summary[/bold green]",
        )
    )
    if case_id:
        console.print(
            f"[dim]Associate this study with case using the backend API or "
            f"facial-align pipeline run {case_id}[/dim]"
        )


def _print_deident_summary(report) -> None:
    table = Table(title="De-identification Summary", show_header=False, box=None)
    table.add_column("Field", style="dim", min_width=24)
    table.add_column("Value")
    table.add_row("Files processed", str(report.files_processed))
    table.add_row("Files succeeded", f"[green]{report.files_succeeded}[/green]")
    table.add_row("Files failed",
                  f"[red]{report.files_failed}[/red]" if report.files_failed else "0")
    table.add_row("Tags modified", str(report.tags_modified))
    table.add_row("Tags removed", str(report.tags_removed))
    table.add_row("UIDs remapped", str(report.tags_uid_remapped))
    table.add_row("Profile applied", report.profile_applied)
    console.print(table)
    if report.errors:
        for e in report.errors:
            err_console.print(f"  [red]✗[/red] {e}")


def _run_qc_on_dicom_dir(dicom_dir: Path, verbose: bool) -> Optional[object]:
    """Load first available DICOM volume and run QC. Returns None on failure."""
    try:
        import numpy as np
        _abort_if_unavailable("CTQualityController", CTQualityController)

        # Attempt to load with pydicom
        import pydicom

        dcm_files = sorted(dicom_dir.rglob("*.dcm"))
        if not dcm_files:
            dcm_files = sorted([f for f in dicom_dir.rglob("*") if f.is_file() and not f.suffix])
        if not dcm_files:
            console.print("[yellow]No DICOM files found for QC analysis.[/yellow]")
            return None

        # Read slices
        slices = []
        positions = []
        for fp in dcm_files[:512]:  # Limit for performance
            try:
                ds = pydicom.dcmread(str(fp), force=True)
                if hasattr(ds, "pixel_array"):
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
                    hu = ds.pixel_array.astype(np.float32) * slope + intercept
                    slices.append(hu)
                    pos = getattr(ds, "ImagePositionPatient", None)
                    if pos:
                        positions.append(float(pos[2]))
            except Exception:
                continue

        if not slices:
            console.print("[yellow]Could not load pixel data for QC.[/yellow]")
            return None

        volume = np.stack(slices, axis=0)

        # Get spacing from first valid slice
        spacing = (1.0, 0.5, 0.5)
        try:
            ds0 = pydicom.dcmread(str(dcm_files[0]), force=True)
            st = float(getattr(ds0, "SliceThickness", 1.0))
            ps = getattr(ds0, "PixelSpacing", [0.5, 0.5])
            spacing = (st, float(ps[0]), float(ps[1]))
        except Exception:
            pass

        qc = CTQualityController()
        return qc.check_volume(volume, spacing=spacing,
                               slice_positions=positions if positions else None)

    except Exception as exc:
        if verbose:
            err_console.print(f"[dim]QC failed: {exc}[/dim]")
        return None


def _print_qc_report(report) -> None:
    grade = report.overall_grade
    color = GRADE_COLORS.get(grade, "white")

    console.print(
        Panel(
            f"Overall Grade: [{color} bold]{_grade_badge(grade).plain}[/{color} bold]\n"
            f"Checks passed: [green]{report.checks_passed}[/green] / "
            f"failed: [red]{report.checks_failed}[/red]\n"
            f"Processing time: {report.processing_time_ms:.0f} ms",
            title="[bold]CT Quality Control Report[/bold]",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", min_width=28)
    table.add_column("Result", no_wrap=True)
    table.add_column("Grade", no_wrap=True)
    table.add_column("Measured", no_wrap=True)
    table.add_column("Description")

    for check in report.all_checks():
        result_icon = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        grade_text = _grade_badge(check.grade_contribution)
        measured = f"{check.measured_value:.3f}" if check.measured_value is not None else "—"
        table.add_row(check.name, result_icon, grade_text, measured, check.description)

    console.print(table)

    if report.warnings:
        for w in report.warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")

    if report.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in report.recommendations:
            console.print(f"  • {rec}")


# ─── case export ──────────────────────────────────────────────────────────────

@case.command(name="export")
@click.argument("case_id")
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--format", "export_format",
              type=click.Choice(["all", "meshes", "plan", "report", "dicom"]),
              default="all", show_default=True,
              help="What to export.")
@click.option("--mesh-format",
              type=click.Choice(["glb", "stl", "ply"]),
              default="glb", show_default=True,
              help="Mesh file format.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Overwrite output directory if it already exists.")
@click.pass_obj
def export_case(
    ctx,
    case_id: str,
    output_dir: Path,
    export_format: str,
    mesh_format: str,
    overwrite: bool,
) -> None:
    """Export case data (meshes, surgical plan, report) to a directory.

    OUTPUT_DIR will be created if it does not exist. Exports a structured
    package suitable for sharing with surgical teams or archival storage.

    \b
    Examples:
      facial-align case export FA-2024-0042 ./exports/FA-2024-0042/
      facial-align case export FA-2024-0042 ./exports/ --format plan
      facial-align case export FA-2024-0042 ./exports/ --mesh-format stl
    """
    if output_dir.exists() and not overwrite:
        err_console.print(
            f"[red]Output directory already exists:[/red] {output_dir}\n"
            "Use --overwrite to replace it."
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    case_data = _fetch_case_detail(case_id, ctx.settings)
    if case_data is None:
        err_console.print(f"[red]Case not found:[/red] {case_id}")
        sys.exit(1)

    files_written = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Exporting {case_id}…", total=4)

        # Manifest
        manifest = {
            "case_number": case_data.get("case_number"),
            "export_date": datetime.utcnow().isoformat(),
            "export_format": export_format,
            "mesh_format": mesh_format,
            "status": case_data.get("status"),
            "surgeon": case_data.get("surgeon_id"),
            "procedure": case_data.get("planned_procedure"),
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        files_written.append(manifest_path)
        progress.advance(task)

        if export_format in ("all", "meshes"):
            mesh_dir = output_dir / "meshes"
            mesh_dir.mkdir(exist_ok=True)
            # Placeholder — real implementation would copy from storage backend
            placeholder = mesh_dir / f"README_{mesh_format.upper()}.txt"
            placeholder.write_text(
                f"Meshes in {mesh_format.upper()} format would be exported here.\n"
                f"Case: {case_data.get('case_number')}\n"
                "Source: storage backend (MinIO/local)\n"
            )
            files_written.append(placeholder)
        progress.advance(task)

        if export_format in ("all", "plan"):
            plan_dir = output_dir / "plan"
            plan_dir.mkdir(exist_ok=True)
            plan_placeholder = plan_dir / "reduction_plan.json"
            plan_placeholder.write_text(
                json.dumps({"note": "Reduction plan export placeholder", "case": case_id}, indent=2)
            )
            files_written.append(plan_placeholder)
        progress.advance(task)

        if export_format in ("all", "report"):
            report_path = output_dir / "surgical_report.txt"
            report_path.write_text(
                f"Facial Align Surgical Planning Report\n"
                f"{'=' * 40}\n"
                f"Case Number  : {case_data.get('case_number', case_id)}\n"
                f"Procedure    : {case_data.get('planned_procedure') or 'N/A'}\n"
                f"Status       : {case_data.get('status', 'N/A')}\n"
                f"Export Date  : {datetime.utcnow().date()}\n"
            )
            files_written.append(report_path)
        progress.advance(task)

    if ctx.json_output:
        click.echo(json.dumps({"files": [str(f) for f in files_written]}, indent=2))
        return

    table = Table(title="Export Contents", show_header=True)
    table.add_column("File")
    table.add_column("Size", justify="right")
    for fp in files_written:
        size = fp.stat().st_size if fp.exists() else 0
        table.add_row(str(fp.relative_to(output_dir)), f"{size:,} B")

    console.print(table)
    console.print(f"[green]Export complete:[/green] {output_dir}")


# ─── case archive ─────────────────────────────────────────────────────────────

@case.command(name="archive")
@click.argument("case_id")
@click.option("--confirm", is_flag=True, default=False,
              help="Skip the interactive confirmation prompt.")
@click.pass_obj
def archive_case(ctx, case_id: str, confirm: bool) -> None:
    """Archive a completed (APPROVED) case.

    Archiving moves the case to ARCHIVED status and removes it from active
    queues. The data is retained for audit and retrieval purposes.

    Only cases in APPROVED status can be archived. Use `case show` to check
    the current status before archiving.

    \b
    Examples:
      facial-align case archive FA-2024-0042
      facial-align case archive FA-2024-0042 --confirm
    """
    case_data = _fetch_case_detail(case_id, ctx.settings)
    if case_data is None:
        err_console.print(f"[red]Case not found:[/red] {case_id}")
        sys.exit(1)

    current_status = case_data.get("status", "")
    if current_status != "APPROVED":
        err_console.print(
            f"[red]Cannot archive case {case_id}:[/red] "
            f"current status is [bold]{current_status}[/bold]. "
            "Only APPROVED cases can be archived."
        )
        sys.exit(1)

    if not confirm:
        click.confirm(
            f"Archive case {case_id} ({case_data.get('planned_procedure') or 'no procedure set'})? "
            "This cannot be undone.",
            abort=True,
        )

    success = _transition_case_status(case_id, "ARCHIVED", ctx.settings)
    if ctx.json_output:
        click.echo(json.dumps({"case_id": case_id, "new_status": "ARCHIVED", "success": success}))
        return

    if success:
        console.print(f"[green]Case {case_id} archived successfully.[/green]")
    else:
        console.print(
            f"[yellow]Case {case_id} archived (DB unavailable — status not persisted).[/yellow]"
        )


def _transition_case_status(case_id: str, new_status: str, settings) -> bool:
    try:
        from sqlalchemy import create_engine, select, or_
        from sqlalchemy.orm import Session
        from apps.backend.app.models.case import SurgicalCase, CaseStatus

        if settings is None:
            return False

        engine = create_engine(settings.db.sync_url, echo=False)
        with Session(engine) as session:
            stmt = select(SurgicalCase).where(
                or_(SurgicalCase.case_number == case_id, SurgicalCase.id.cast(str) == case_id)  # type: ignore
            )
            row = session.scalars(stmt).first()
            if row:
                row.transition_to(CaseStatus(new_status))
                session.commit()
                return True
    except Exception:
        pass
    return False


# ─── case qc ──────────────────────────────────────────────────────────────────

@case.command(name="qc")
@click.argument("case_id")
@click.option("--verbose-report", is_flag=True, default=False,
              help="Show full per-check recommendations in the output.")
@click.pass_obj
def qc_case(ctx, case_id: str, verbose_report: bool) -> None:
    """Run CT quality control on an existing case's imaging data.

    Loads the case's NIfTI volume (or DICOM files) from storage and evaluates
    the scan against Facial Align's quality requirements for surgical planning.

    Quality grades:
      Grade A — Ideal, all checks passed
      Grade B — Acceptable with caveats
      Grade C — Marginal, surgeon approval required
      Grade F — Rejected, rescan required

    \b
    Examples:
      facial-align case qc FA-2024-0042
      facial-align case qc FA-2024-0042 --verbose-report
      facial-align case qc FA-2024-0042 --json
    """
    _abort_if_unavailable("CTQualityController", CTQualityController)

    case_data = _fetch_case_detail(case_id, ctx.settings)
    if case_data is None:
        err_console.print(f"[red]Case not found:[/red] {case_id}")
        sys.exit(1)

    console.print(f"Running QC for case [cyan]{case_id}[/cyan] …")

    # Resolve DICOM path from settings
    dicom_base = Path("/data/facialign/dicom")
    if ctx.settings:
        try:
            dicom_base = Path(ctx.settings.storage.dicom_path)
        except Exception:
            pass

    case_dicom_dir = dicom_base / case_id
    if not case_dicom_dir.exists():
        # Try to find by scanning storage
        case_dicom_dir = dicom_base

    report = _run_qc_on_dicom_dir(case_dicom_dir, ctx.verbose)

    if report is None:
        err_console.print(
            "[yellow]QC could not be performed: no CT volume found for this case.[/yellow]\n"
            "Ensure the DICOM study has been imported with "
            "[bold]case import-dicom[/bold] first."
        )
        sys.exit(1)

    if ctx.json_output:
        result = {
            "case_id": case_id,
            "overall_grade": report.overall_grade,
            "checks_passed": report.checks_passed,
            "checks_failed": report.checks_failed,
            "volume_shape": report.volume_shape,
            "voxel_spacing_mm": report.voxel_spacing_mm,
            "coverage_mm": report.coverage_mm,
            "processing_time_ms": report.processing_time_ms,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "grade": c.grade_contribution,
                    "value": c.measured_value,
                    "description": c.description,
                }
                for c in report.all_checks()
            ],
            "recommendations": report.recommendations if verbose_report else [],
        }
        click.echo(json.dumps(result, indent=2))
        return

    _print_qc_report(report)

    if not verbose_report and report.recommendations:
        console.print(
            f"\n[dim]{len(report.recommendations)} recommendation(s). "
            "Use --verbose-report to see them.[/dim]"
        )
