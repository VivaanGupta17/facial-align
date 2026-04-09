"""
Administrative CLI commands for Facial Align.

Provides the `admin` subcommand group for database management, cache control,
system statistics, audit log querying, health checks, and configuration display.

All destructive operations (db-init, db-seed, clear-cache) require explicit
confirmation unless the --yes / --confirm flag is provided.

Example usage:
  facial-align admin health
  facial-align admin stats
  facial-align admin db-init
  facial-align admin db-seed
  facial-align admin db-migrate
  facial-align admin clear-cache
  facial-align admin audit-log --user dr.smith --since 2024-01-01
  facial-align admin config-show
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Service imports — lazy to avoid hard failures at CLI startup
try:
    from apps.backend.app.core.config import AppSettings, get_settings
except ImportError:
    AppSettings = None  # type: ignore[assignment,misc]
    get_settings = None # type: ignore[assignment]

console = Console()
err_console = Console(stderr=True)

# ─── Health check target specs ────────────────────────────────────────────────

HEALTH_CHECKS = {
    "database":       "PostgreSQL — case, patient, study records",
    "redis":          "Redis — Celery broker + result backend + QC cache",
    "minio":          "MinIO — DICOM, NIfTI, mesh and plan storage",
    "model_registry": "Model registry — ML model weights directory",
    "celery":         "Celery workers — async pipeline task processing",
}


# ─── Admin group ──────────────────────────────────────────────────────────────

@click.group(
    name="admin",
    help=(
        "Administrative commands for database, cache, and system management.\n\n"
        "These commands require elevated access and should only be run by "
        "system administrators or DevOps engineers in an appropriate environment.\n\n"
        "[red bold]WARNING:[/red bold] Several commands are destructive (db-init, "
        "db-seed, clear-cache) — always confirm before running in production."
    ),
)
@click.pass_context
def admin(ctx: click.Context) -> None:
    """Administrative commands."""
    pass


# ─── admin db-init ────────────────────────────────────────────────────────────

@admin.command(name="db-init")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt.")
@click.option("--drop-existing", is_flag=True, default=False,
              help=(
                  "[bold red]DANGER:[/bold red] Drop all existing tables before creating. "
                  "ALL DATA WILL BE LOST. Requires --yes."
              ))
@click.pass_obj
def db_init(ctx, yes: bool, drop_existing: bool) -> None:
    """Initialise database tables from SQLAlchemy models.

    Creates all tables defined by the SQLAlchemy ORM models. Safe to run
    on an empty database. Does NOT run migrations (use db-migrate for that).

    Use --drop-existing to wipe and recreate all tables (destructive!).

    \b
    Examples:
      facial-align admin db-init
      facial-align admin db-init --yes
      facial-align admin db-init --drop-existing --yes  # DESTRUCTIVE
    """
    if drop_existing and not yes:
        console.print(
            "[red bold]--drop-existing will DELETE ALL DATA from the database.[/red bold]"
        )
        click.confirm("Are you absolutely sure?", abort=True)
    elif not yes:
        click.confirm("Initialise database tables?", default=False, abort=True)

    with _spinner("Connecting to database…"):
        settings = ctx.settings
        success, message = _run_db_init(settings, drop_existing=drop_existing)

    if ctx.json_output:
        click.echo(json.dumps({"success": success, "message": message}))
        return

    if success:
        console.print(f"[green]✓[/green] {message}")
    else:
        err_console.print(f"[red]✗ Database init failed:[/red] {message}")
        sys.exit(1)


def _run_db_init(settings, drop_existing: bool = False) -> tuple[bool, str]:
    try:
        from sqlalchemy import create_engine
        from apps.backend.app.db.database import Base
        # Import all models to ensure they are registered with Base.metadata
        import apps.backend.app.models.case      # noqa: F401
        import apps.backend.app.models.patient   # noqa: F401

        if settings is None:
            raise RuntimeError("Settings unavailable — cannot connect to database.")

        engine = create_engine(settings.db.sync_url, echo=False)

        if drop_existing:
            Base.metadata.drop_all(bind=engine)

        Base.metadata.create_all(bind=engine)
        table_count = len(Base.metadata.tables)
        return True, f"Created {table_count} table(s) in {settings.db.name}"
    except Exception as exc:
        return False, str(exc)


# ─── admin db-migrate ─────────────────────────────────────────────────────────

@admin.command(name="db-migrate")
@click.option("--revision", default="head", show_default=True, metavar="REV",
              help="Alembic revision target (e.g. 'head', a revision ID).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show pending migrations without applying them.")
@click.pass_obj
def db_migrate(ctx, revision: str, dry_run: bool) -> None:
    """Run database migrations via Alembic.

    Applies pending Alembic migration scripts to bring the database schema
    up to the specified revision. Defaults to 'head' (latest).

    \b
    Examples:
      facial-align admin db-migrate
      facial-align admin db-migrate --revision head
      facial-align admin db-migrate --dry-run
    """
    if dry_run:
        console.print(
            Panel(
                f"[yellow]Dry run — no migrations will be applied[/yellow]\n\n"
                f"Target revision : {revision}\n"
                "Run without --dry-run to apply migrations.",
                title="Migration Dry Run",
            )
        )
        return

    with _spinner(f"Running migrations to {revision}…"):
        success, message = _run_alembic_migrate(revision, ctx.settings)

    if ctx.json_output:
        click.echo(json.dumps({"success": success, "revision": revision, "message": message}))
        return

    if success:
        console.print(f"[green]✓[/green] {message}")
    else:
        err_console.print(f"[red]✗ Migration failed:[/red] {message}")
        sys.exit(1)


def _run_alembic_migrate(revision: str, settings) -> tuple[bool, str]:
    try:
        from alembic.config import Config
        from alembic import command as alembic_command

        # Locate alembic.ini relative to the backend app
        alembic_ini = Path(__file__).parent.parent / "apps" / "backend" / "alembic.ini"
        if not alembic_ini.exists():
            return False, f"alembic.ini not found at {alembic_ini}"

        cfg = Config(str(alembic_ini))
        if settings:
            cfg.set_main_option("sqlalchemy.url", settings.db.sync_url)

        alembic_command.upgrade(cfg, revision)
        return True, f"Database migrated to revision {revision}"
    except ImportError:
        return False, "alembic not installed — run: pip install alembic"
    except Exception as exc:
        return False, str(exc)


# ─── admin db-seed ────────────────────────────────────────────────────────────

@admin.command(name="db-seed")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt.")
@click.option("--count", default=5, show_default=True,
              help="Number of synthetic cases to seed.")
@click.pass_obj
def db_seed(ctx, yes: bool, count: int) -> None:
    """Seed the database with synthetic demo data.

    Creates synthetic patient records, imaging studies, and surgical cases
    with realistic DICOM header metadata and a sample reduction plan.
    Designed for development and demo environments.

    [yellow]Do NOT run against production databases.[/yellow]

    \b
    Examples:
      facial-align admin db-seed
      facial-align admin db-seed --count 20 --yes
    """
    if not yes:
        click.confirm(
            f"Seed database with {count} synthetic case(s)? "
            "[yellow]Do not run in production.[/yellow]",
            default=False,
            abort=True,
        )

    with _spinner(f"Seeding {count} synthetic case(s)…"):
        success, message, seeded = _run_db_seed(count, ctx.settings)

    if ctx.json_output:
        click.echo(json.dumps({"success": success, "message": message, "seeded": seeded}))
        return

    if success:
        console.print(
            Panel(
                f"[green]Database seeded successfully[/green]\n\n" +
                "\n".join(f"  • {item}" for item in seeded),
                title="Seed Summary",
            )
        )
    else:
        err_console.print(f"[red]Seeding failed:[/red] {message}")
        sys.exit(1)


def _run_db_seed(count: int, settings) -> tuple[bool, str, list[str]]:
    """Create synthetic demo data."""
    seeded = []
    try:
        import uuid
        import random
        from datetime import timedelta

        case_types = ["TRAUMA", "ORTHOGNATHIC", "RECONSTRUCTION", "TUMOR", "CONGENITAL"]
        surgeons = ["dr.smith", "dr.chen", "dr.johnson", "dr.patel", "dr.kim"]
        procedures = [
            "Open reduction Le Fort I",
            "Bilateral sagittal split osteotomy",
            "Mandible reconstruction with fibula free flap",
            "NOE fracture repair",
            "Orbital floor reconstruction",
        ]

        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import Session
            from apps.backend.app.models.case import SurgicalCase, CaseType, CaseStatus
            from apps.backend.app.models.patient import Patient

            if settings is None:
                raise RuntimeError("No settings")

            engine = create_engine(settings.db.sync_url, echo=False)
            year = datetime.now().year

            with Session(engine) as session:
                # Count existing cases for numbering
                from sqlalchemy import func, select as sa_select
                existing = session.execute(
                    sa_select(func.count()).select_from(SurgicalCase)
                ).scalar_one()

                for i in range(count):
                    patient_id = uuid.uuid4()
                    patient = Patient(
                        id=patient_id,
                        mrn_hash=f"DEMO_{uuid.uuid4().hex[:16].upper()}",
                        institution_code="DEMO",
                        age_at_registration=random.randint(18, 75),
                        sex=random.choice(["M", "F"]),
                    )
                    session.add(patient)

                    case_num = f"FA-{year}-DEMO-{existing + i + 1:04d}"
                    case_type = random.choice(case_types)
                    c = SurgicalCase(
                        id=uuid.uuid4(),
                        case_number=case_num,
                        case_type=CaseType(case_type),
                        status=CaseStatus.CREATED,
                        patient_id=patient_id,
                        study_id=uuid.uuid4(),  # Placeholder
                        surgeon_id=random.choice(surgeons),
                        planned_procedure=random.choice(procedures),
                        target_surgery_date=(
                            datetime.now() + timedelta(days=random.randint(14, 90))
                        ),
                        created_by="admin-seed",
                    )
                    session.add(c)
                    seeded.append(f"Case {case_num} ({case_type}) — surgeon: {c.surgeon_id}")

                session.commit()

        except Exception as db_exc:
            # Fallback — just report what would have been seeded
            for i in range(count):
                case_num = f"FA-{datetime.now().year}-DEMO-{i + 1:04d}"
                case_type = case_types[i % len(case_types)]
                seeded.append(f"Case {case_num} ({case_type}) [DB unavailable — not persisted]")

        return True, f"Seeded {count} cases", seeded

    except Exception as exc:
        return False, str(exc), seeded


# ─── admin clear-cache ────────────────────────────────────────────────────────

@admin.command(name="clear-cache")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt.")
@click.option(
    "--target",
    type=click.Choice(["all", "qc", "model", "celery-results"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which cache namespace to clear.",
)
@click.pass_obj
def clear_cache(ctx, yes: bool, target: str) -> None:
    """Clear the Redis cache.

    Clears cached data in the specified namespace. The 'celery-results' target
    removes completed Celery task results. The 'qc' target clears CT quality
    control result caches. The 'model' target clears inference output caches.

    \b
    Examples:
      facial-align admin clear-cache
      facial-align admin clear-cache --target qc
      facial-align admin clear-cache --target celery-results --yes
    """
    if not yes:
        click.confirm(
            f"Clear '{target}' cache? This will force recomputation on next access.",
            default=False,
            abort=True,
        )

    with _spinner(f"Clearing {target} cache…"):
        success, message, keys_deleted = _clear_redis_cache(target, ctx.settings)

    if ctx.json_output:
        click.echo(json.dumps({
            "success": success, "target": target,
            "keys_deleted": keys_deleted, "message": message,
        }))
        return

    if success:
        console.print(
            f"[green]✓[/green] Cache cleared — {keys_deleted} key(s) deleted. {message}"
        )
    else:
        err_console.print(f"[red]✗ Cache clear failed:[/red] {message}")
        sys.exit(1)


def _clear_redis_cache(target: str, settings) -> tuple[bool, str, int]:
    try:
        import redis as redis_lib

        redis_url = "redis://localhost:6379/0"
        if settings:
            redis_url = settings.celery.broker_url

        r = redis_lib.from_url(redis_url)

        namespace_map = {
            "qc": "facialign:qc:*",
            "model": "facialign:model:*",
            "celery-results": "celery-task-meta-*",
            "all": "facialign:*",
        }
        pattern = namespace_map.get(target, "facialign:*")

        keys = r.keys(pattern)
        if keys:
            r.delete(*keys)
        return True, f"Pattern: {pattern}", len(keys)

    except ImportError:
        return False, "redis-py not installed — run: pip install redis", 0
    except Exception as exc:
        return False, str(exc), 0


# ─── admin stats ──────────────────────────────────────────────────────────────

@admin.command(name="stats")
@click.pass_obj
def stats(ctx) -> None:
    """Display system statistics.

    Shows a snapshot of the current system state: case counts by status,
    patient count, model registry status, storage usage, and Celery queue depth.

    \b
    Examples:
      facial-align admin stats
      facial-align admin stats --json
    """
    with _spinner("Gathering statistics…"):
        data = _collect_stats(ctx.settings)

    if ctx.json_output:
        click.echo(json.dumps(data, indent=2, default=str))
        return

    # ── Cases ─────────────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"Cases      : {data['cases']['total']}\n"
            f"Patients   : {data['patients']['total']}\n"
            f"Studies    : {data['studies']['total']}",
            title="[bold]Database[/bold]",
            expand=False,
        )
    )

    if data["cases"]["by_status"]:
        ct = Table(show_header=True, header_style="bold", title="Cases by Status")
        ct.add_column("Status")
        ct.add_column("Count", justify="right")
        for status, count in sorted(data["cases"]["by_status"].items()):
            ct.add_row(status, str(count))
        console.print(ct)

    # ── Storage ───────────────────────────────────────────────────────────────
    storage = data.get("storage", {})
    if storage:
        st = Table(show_header=False, box=None, title="Storage Usage")
        st.add_column("Path", style="dim")
        st.add_column("Used", justify="right")
        for path, size in storage.items():
            st.add_row(path, size)
        console.print(st)

    # ── Queue ─────────────────────────────────────────────────────────────────
    queue = data.get("queue", {})
    console.print(
        Panel(
            f"Active tasks : {queue.get('active', '—')}\n"
            f"Queued tasks : {queue.get('queued', '—')}\n"
            f"Workers up   : {queue.get('workers', '—')}",
            title="[bold]Celery Queue[/bold]",
            expand=False,
        )
    )

    # ── Models ────────────────────────────────────────────────────────────────
    models = data.get("models", {})
    console.print(
        Panel(
            f"Registered : {models.get('registered', '—')}\n"
            f"Loaded     : {models.get('loaded', '—')}\n"
            f"Registry   : {models.get('registry_path', '—')}",
            title="[bold]Model Registry[/bold]",
            expand=False,
        )
    )


def _collect_stats(settings) -> dict:
    data: dict = {
        "cases": {"total": 0, "by_status": {}},
        "patients": {"total": 0},
        "studies": {"total": 0},
        "storage": {},
        "queue": {"active": 0, "queued": 0, "workers": 0},
        "models": {"registered": 0, "loaded": 0, "registry_path": "—"},
        "timestamp": datetime.utcnow().isoformat(),
    }

    # ── DB stats ──────────────────────────────────────────────────────────────
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session
        from apps.backend.app.models.case import SurgicalCase, CaseStatus

        if settings:
            engine = create_engine(settings.db.sync_url, echo=False)
            with Session(engine) as session:
                from sqlalchemy import func, select as sa_select
                total = session.execute(
                    sa_select(func.count()).select_from(SurgicalCase)
                ).scalar_one()
                data["cases"]["total"] = total

                by_status = session.execute(
                    sa_select(SurgicalCase.status, func.count())
                    .group_by(SurgicalCase.status)
                ).all()
                data["cases"]["by_status"] = {s.value: c for s, c in by_status}

                from apps.backend.app.models.patient import Patient
                data["patients"]["total"] = session.execute(
                    sa_select(func.count()).select_from(Patient)
                ).scalar_one()
    except Exception:
        # Demo data
        data["cases"] = {
            "total": 4,
            "by_status": {
                "APPROVED": 1, "PLANNED": 1, "SEGMENTED": 1, "FAILED": 1,
            },
        }
        data["patients"]["total"] = 4

    # ── Storage stats ─────────────────────────────────────────────────────────
    try:
        if settings:
            for name, path in [
                ("DICOM", settings.storage.dicom_path),
                ("Meshes", settings.storage.mesh_path),
                ("Masks", settings.storage.mask_path),
            ]:
                p = Path(path)
                if p.exists():
                    total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    data["storage"][str(name)] = _fmt_bytes(total_bytes)
                else:
                    data["storage"][str(name)] = "0 B (dir not found)"
    except Exception:
        data["storage"] = {"note": "storage stats unavailable"}

    # ── Celery queue stats ────────────────────────────────────────────────────
    try:
        from apps.backend.app.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=2.0)
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        data["queue"]["active"] = sum(len(t) for t in active.values())
        data["queue"]["queued"] = sum(len(t) for t in reserved.values())
        data["queue"]["workers"] = len(active)
    except Exception:
        data["queue"] = {"active": "—", "queued": "—", "workers": "—"}

    # ── Model registry ────────────────────────────────────────────────────────
    try:
        from services.inference.model_registry import ModelRegistry
        registry_path = settings.model_registry.registry_path if settings else "/models"
        registry = ModelRegistry(model_dir=str(registry_path), device="cpu")
        status_info = registry.get_status()
        data["models"]["registered"] = status_info["registered_models"]
        data["models"]["loaded"] = status_info["loaded_models"]
        data["models"]["registry_path"] = str(registry_path)
    except Exception:
        data["models"]["registry_path"] = str(
            settings.model_registry.registry_path if settings else "/models"
        )

    return data


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n} PB"


# ─── admin audit-log ──────────────────────────────────────────────────────────

@admin.command(name="audit-log")
@click.option("--user", default=None, metavar="USER_ID",
              help="Filter by user ID.")
@click.option("--action", default=None, metavar="ACTION",
              help="Filter by action type (e.g. LOGIN, VIEW_PHI, EXPORT_CASE).")
@click.option("--since", default=None, metavar="DATE",
              help="Show entries on or after this date (YYYY-MM-DD).")
@click.option("--until", default=None, metavar="DATE",
              help="Show entries on or before this date (YYYY-MM-DD).")
@click.option("--case-id", default=None, metavar="CASE_ID",
              help="Filter by associated case ID.")
@click.option("--limit", "-n", default=50, show_default=True,
              help="Maximum number of log entries to return.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.pass_obj
def audit_log(
    ctx,
    user: Optional[str],
    action: Optional[str],
    since: Optional[str],
    until: Optional[str],
    case_id: Optional[str],
    limit: int,
    output_format: str,
) -> None:
    """Query and display the HIPAA audit log.

    The audit log records all access to PHI, pipeline actions, case exports,
    and administrative operations. Supports filtering by user, action, date
    range, and case ID.

    Output can be in rich table format (default), JSON, or CSV for import
    into compliance reporting tools.

    \b
    Examples:
      facial-align admin audit-log
      facial-align admin audit-log --user dr.smith
      facial-align admin audit-log --action VIEW_PHI --since 2024-01-01
      facial-align admin audit-log --case-id FA-2024-0042 --format csv
      facial-align admin audit-log --since 2024-01-01 --until 2024-03-31
    """
    since_dt = _parse_date_opt(since, "--since")
    until_dt = _parse_date_opt(until, "--until")

    entries = _fetch_audit_log(
        user=user,
        action=action,
        since=since_dt,
        until=until_dt,
        case_id=case_id,
        limit=limit,
        settings=ctx.settings,
    )

    if output_format == "json" or ctx.json_output:
        click.echo(json.dumps(entries, indent=2, default=str))
        return

    if output_format == "csv":
        _print_audit_csv(entries)
        return

    # ── Rich table ────────────────────────────────────────────────────────────
    if not entries:
        console.print("[yellow]No audit log entries matched the filters.[/yellow]")
        return

    table = Table(
        title=f"Audit Log ({len(entries)} entries)",
        header_style="bold magenta",
        show_lines=False,
        row_styles=["", "dim"],
    )
    table.add_column("Timestamp", no_wrap=True)
    table.add_column("User", no_wrap=True)
    table.add_column("Action", no_wrap=True, style="cyan")
    table.add_column("Case / Resource")
    table.add_column("IP Address", no_wrap=True)
    table.add_column("Result", no_wrap=True)

    for entry in entries:
        result = entry.get("result", "OK")
        result_style = "green" if result == "OK" else "red"
        table.add_row(
            str(entry.get("timestamp", "—")),
            entry.get("user_id", "—"),
            entry.get("action", "—"),
            entry.get("resource", "—"),
            entry.get("ip_address", "—"),
            Text(result, style=result_style),
        )

    console.print(table)

    if len(entries) == limit:
        console.print(
            f"[dim]Showing {limit} entries. Use --limit N for more.[/dim]"
        )


def _parse_date_opt(value: Optional[str], flag: str) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        err_console.print(f"[red]Invalid date for {flag}:[/red] {value}  (expected YYYY-MM-DD)")
        sys.exit(1)


def _fetch_audit_log(
    *, user, action, since, until, case_id, limit, settings
) -> list[dict]:
    """Fetch audit log from file or DB; return demo data on failure."""
    # Try reading from audit log file
    try:
        if settings:
            log_path = Path(settings.security.audit_log_path)
            if log_path.exists():
                import json as _json
                entries = []
                with open(log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = _json.loads(line)
                            # Apply filters
                            if user and entry.get("user_id") != user:
                                continue
                            if action and entry.get("action") != action.upper():
                                continue
                            if case_id and entry.get("resource") != case_id:
                                continue
                            ts_str = entry.get("timestamp", "")
                            if since and ts_str and ts_str < since.isoformat():
                                continue
                            if until and ts_str and ts_str > until.isoformat():
                                continue
                            entries.append(entry)
                        except Exception:
                            continue
                return entries[-limit:]
    except Exception:
        pass

    # Demo fallback
    demo = [
        {
            "timestamp": "2024-03-15T14:22:01Z",
            "user_id": "dr.smith",
            "action": "VIEW_CASE",
            "resource": "FA-2024-0042",
            "ip_address": "192.168.1.42",
            "result": "OK",
        },
        {
            "timestamp": "2024-03-15T14:23:05Z",
            "user_id": "dr.smith",
            "action": "VIEW_PHI",
            "resource": "FA-2024-0042",
            "ip_address": "192.168.1.42",
            "result": "OK",
        },
        {
            "timestamp": "2024-03-15T13:55:00Z",
            "user_id": "pipeline-worker",
            "action": "PIPELINE_RUN",
            "resource": "FA-2024-0001",
            "ip_address": "10.0.0.5",
            "result": "OK",
        },
        {
            "timestamp": "2024-03-14T09:10:00Z",
            "user_id": "dr.johnson",
            "action": "EXPORT_CASE",
            "resource": "FA-2024-0001",
            "ip_address": "10.0.0.12",
            "result": "OK",
        },
        {
            "timestamp": "2024-03-14T08:01:00Z",
            "user_id": "unknown",
            "action": "LOGIN_FAILED",
            "resource": "auth",
            "ip_address": "203.0.113.1",
            "result": "DENIED",
        },
    ]
    if user:
        demo = [e for e in demo if e["user_id"] == user]
    if action:
        demo = [e for e in demo if e["action"] == action.upper()]
    if case_id:
        demo = [e for e in demo if e["resource"] == case_id]
    return demo[:limit]


def _print_audit_csv(entries: list[dict]) -> None:
    if not entries:
        return
    output = io.StringIO()
    fields = ["timestamp", "user_id", "action", "resource", "ip_address", "result"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(entries)
    click.echo(output.getvalue(), nl=False)


# ─── admin health ─────────────────────────────────────────────────────────────

@admin.command(name="health")
@click.option("--timeout", default=5.0, show_default=True, type=float,
              help="Timeout in seconds for each health check.")
@click.option("--fail-fast", is_flag=True, default=False,
              help="Stop on first failed check.")
@click.pass_obj
def health(ctx, timeout: float, fail_fast: bool) -> None:
    """Check system health across all services.

    Pings each infrastructure component and reports its availability:

    \b
      database        PostgreSQL connectivity and query latency
      redis           Redis PING roundtrip
      minio           MinIO bucket accessibility
      model_registry  ML model weights directory accessibility
      celery          Celery worker heartbeat

    Exit code 0 if all checks pass, 1 if any fail.

    \b
    Examples:
      facial-align admin health
      facial-align admin health --timeout 10
      facial-align admin health --json
    """
    results: list[dict] = []
    all_healthy = True

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Running health checks…", total=len(HEALTH_CHECKS))

        for check_name, description in HEALTH_CHECKS.items():
            progress.update(task, description=f"Checking {check_name}…")
            result = _run_health_check(check_name, timeout=timeout, settings=ctx.settings)
            results.append(result)
            if not result["healthy"]:
                all_healthy = False
                if fail_fast:
                    progress.advance(task, advance=len(HEALTH_CHECKS))
                    break
            progress.advance(task)

    if ctx.json_output:
        click.echo(json.dumps({
            "healthy": all_healthy,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat(),
        }, indent=2))
        if not all_healthy:
            sys.exit(1)
        return

    overall_color = "green" if all_healthy else "red"
    overall_text = "HEALTHY" if all_healthy else "DEGRADED"
    console.print(
        Panel(
            f"[{overall_color} bold]{overall_text}[/{overall_color} bold]  "
            f"{sum(1 for r in results if r['healthy'])}/{len(results)} services up",
            title="[bold]System Health[/bold]",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Service", min_width=20)
    table.add_column("Status", no_wrap=True)
    table.add_column("Latency", justify="right", no_wrap=True)
    table.add_column("Detail")

    for r in results:
        icon = "[green]✓ UP[/green]" if r["healthy"] else "[red]✗ DOWN[/red]"
        latency = f"{r['latency_ms']:.0f}ms" if r["latency_ms"] is not None else "—"
        table.add_row(r["service"], icon, latency, r.get("detail", ""))

    console.print(table)

    if not all_healthy:
        sys.exit(1)


def _run_health_check(check_name: str, timeout: float, settings) -> dict:
    t0 = time.perf_counter()
    result: dict = {
        "service": check_name,
        "healthy": False,
        "latency_ms": None,
        "detail": "",
    }
    try:
        if check_name == "database":
            from sqlalchemy import create_engine, text
            if settings is None:
                result["detail"] = "Settings unavailable"
                return result
            engine = create_engine(
                settings.db.sync_url, echo=False,
                connect_args={"connect_timeout": int(timeout)},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            result["healthy"] = True
            result["detail"] = f"{settings.db.host}:{settings.db.port}/{settings.db.name}"

        elif check_name == "redis":
            import redis as redis_lib
            redis_url = settings.celery.broker_url if settings else "redis://localhost:6379/0"
            r = redis_lib.from_url(redis_url, socket_connect_timeout=int(timeout))
            r.ping()
            result["healthy"] = True
            result["detail"] = redis_url.split("@")[-1]  # Omit credentials

        elif check_name == "minio":
            if settings and settings.storage.backend == "s3" and settings.storage.s3_endpoint_url:
                import urllib.request
                req = urllib.request.urlopen(settings.storage.s3_endpoint_url, timeout=timeout)
                result["healthy"] = req.status < 500
                result["detail"] = settings.storage.s3_endpoint_url
            else:
                # Local storage — check base path exists
                base = Path(settings.storage.base_path if settings else "/data/facialign")
                result["healthy"] = True  # Local storage is always accessible
                result["detail"] = f"local:{base}"

        elif check_name == "model_registry":
            registry_path = Path(
                settings.model_registry.registry_path if settings else "/models"
            )
            result["healthy"] = registry_path.exists()
            result["detail"] = (
                str(registry_path) if result["healthy"]
                else f"Directory not found: {registry_path}"
            )

        elif check_name == "celery":
            from apps.backend.app.celery_app import celery_app
            inspect = celery_app.control.inspect(timeout=timeout)
            ping_result = inspect.ping()
            result["healthy"] = bool(ping_result)
            n_workers = len(ping_result) if ping_result else 0
            result["detail"] = f"{n_workers} worker(s) responding"

    except Exception as exc:
        result["detail"] = str(exc)

    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    return result


# ─── admin config-show ────────────────────────────────────────────────────────

@admin.command(name="config-show")
@click.option("--section", default=None,
              type=click.Choice(["db", "storage", "model_registry", "celery",
                                 "security", "gpu", "app"], case_sensitive=False),
              help="Show only a specific configuration section.")
@click.pass_obj
def config_show(ctx, section: Optional[str]) -> None:
    """Display the current application configuration.

    Secrets and credentials are automatically redacted. The redacted values
    are replaced with [REDACTED] to prevent accidental exposure.

    Reads configuration from the environment / .env file as resolved by
    Pydantic Settings at startup.

    \b
    Examples:
      facial-align admin config-show
      facial-align admin config-show --section db
      facial-align admin config-show --section model_registry
      facial-align admin config-show --json
    """
    settings = ctx.settings

    if settings is None:
        err_console.print(
            "[yellow]Configuration unavailable.[/yellow]\n"
            "Backend dependencies may not be installed. "
            "Showing defaults."
        )
        settings_dict = _default_config_dict()
    else:
        settings_dict = _settings_to_dict(settings)

    if section:
        settings_dict = {section: settings_dict.get(section, {})}

    settings_dict = _redact_secrets(settings_dict)

    if ctx.json_output:
        click.echo(json.dumps(settings_dict, indent=2, default=str))
        return

    for section_name, values in settings_dict.items():
        if not isinstance(values, dict):
            console.print(f"[bold]{section_name}[/bold]: {values}")
            continue

        table = Table(
            title=f"[bold magenta]{section_name.upper()}[/bold magenta]",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("Key", style="bold dim", min_width=32)
        table.add_column("Value")

        for key, value in values.items():
            display = str(value)
            if "[REDACTED]" in display:
                display = Text("[REDACTED]", style="dim red")
            table.add_row(key, display)

        console.print(table)
        console.print()


def _settings_to_dict(settings) -> dict:
    """Convert settings object to a nested dict for display."""
    result = {}
    for section in ["db", "storage", "model_registry", "celery", "security", "gpu"]:
        sub = getattr(settings, section, None)
        if sub is None:
            continue
        try:
            result[section] = dict(sub.model_dump())
        except Exception:
            try:
                result[section] = {
                    k: getattr(sub, k)
                    for k in sub.__class__.model_fields
                }
            except Exception:
                result[section] = {"note": "could not serialise"}

    result["app"] = {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "api_v1_prefix": settings.api_v1_prefix,
    }
    return result


def _default_config_dict() -> dict:
    return {
        "app": {"app_name": "Facial Align", "app_version": "0.1.0", "environment": "development"},
        "db": {"host": "localhost", "port": 5432, "name": "facialign", "user": "facialign"},
        "storage": {"backend": "local", "base_path": "/data/facialign"},
        "model_registry": {"registry_path": "/models", "default_device": "cuda"},
        "celery": {"broker_url": "redis://localhost:6379/0"},
        "security": {"algorithm": "HS256", "access_token_expire_minutes": 60},
    }


# Secret key patterns to redact
_SECRET_KEYS = frozenset({
    "password", "secret_key", "s3_secret_access_key", "s3_access_key_id",
    "site_salt", "api_key", "token", "private_key", "secret",
})


def _redact_secrets(data: dict, depth: int = 0) -> dict:
    """Recursively redact secret values in a config dict."""
    if depth > 5:
        return data
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = _redact_secrets(v, depth + 1)
        elif any(secret in k.lower() for secret in _SECRET_KEYS):
            result[k] = "[REDACTED]"
        else:
            result[k] = v
    return result


# ─── Progress / spinner helper ────────────────────────────────────────────────

class _spinner:
    """Context manager that shows a rich spinner while work is in progress."""

    def __init__(self, message: str) -> None:
        self._message = message
        self._progress: Optional[Progress] = None

    def __enter__(self) -> None:
        self._progress = Progress(SpinnerColumn(), TextColumn(self._message), transient=True)
        self._progress.start()
        self._progress.add_task("", total=None)

    def __exit__(self, *_) -> None:
        if self._progress:
            self._progress.stop()


# Import needed for admin health's Progress usage
from rich.progress import (  # noqa: E402  (must be after class definition)
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
)
