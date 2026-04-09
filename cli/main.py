"""
Facial Align CLI — Main entry point.

Defines the top-level `facial-align` command group and shared context
that is propagated to all subcommands via Click's pass_context mechanism.

Usage:
    facial-align [OPTIONS] COMMAND [ARGS]...

    facial-align --help
    facial-align case list --status PLANNED
    facial-align model download totalsegmentator
    facial-align pipeline run FA-2024-0042
    facial-align admin health
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ─── Version ──────────────────────────────────────────────────────────────────

__version__ = "0.1.0"

# ─── Shared console ───────────────────────────────────────────────────────────

console = Console()
err_console = Console(stderr=True)


# ─── Context object passed to all subcommands ─────────────────────────────────

class AppContext:
    """Shared CLI context propagated via @click.pass_obj."""

    def __init__(
        self,
        config_path: Optional[Path],
        verbose: bool,
        quiet: bool,
        json_output: bool,
    ) -> None:
        self.config_path = config_path
        self.verbose = verbose
        self.quiet = quiet
        self.json_output = json_output
        self._settings = None

    @property
    def settings(self):
        """Lazy-load application settings, optionally overriding with --config."""
        if self._settings is None:
            try:
                # Allow --config to point at a custom .env file
                if self.config_path:
                    import os
                    os.environ.setdefault("ENV_FILE", str(self.config_path))
                from apps.backend.app.core.config import get_settings
                self._settings = get_settings()
            except Exception as exc:  # noqa: BLE001
                if not self.quiet:
                    err_console.print(
                        f"[yellow]Warning:[/yellow] Could not load app settings: {exc}\n"
                        "Some commands may be unavailable without a running backend.",
                    )
                self._settings = None
        return self._settings

    def log(self, message: str, level: str = "info") -> None:
        """Print a log message respecting --quiet / --verbose."""
        if self.quiet:
            return
        if level == "debug" and not self.verbose:
            return
        color = {"info": "cyan", "debug": "dim", "warning": "yellow", "error": "red"}.get(
            level, "white"
        )
        console.print(f"[{color}]{message}[/{color}]")

    def output(self, data: object, *, title: str = "") -> None:
        """Emit data as JSON (--json) or rich formatted output."""
        if self.json_output:
            if isinstance(data, str):
                click.echo(data)
            else:
                click.echo(json.dumps(data, indent=2, default=str))
        else:
            if title:
                console.print(Panel(str(data), title=title))
            else:
                console.print(data)


# ─── Root command group ───────────────────────────────────────────────────────

@click.group(
    name="facial-align",
    help=(
        "Facial Align — Craniofacial surgical planning platform CLI.\n\n"
        "Manage cases, ML models, pipeline execution, and system administration "
        "from the command line. All commands support --json for machine-readable "
        "output and integrate with the Facial Align backend services.\n\n"
        "Quick start:\n\n"
        "  facial-align case list\n\n"
        "  facial-align model download totalsegmentator\n\n"
        "  facial-align pipeline run FA-2024-0042\n\n"
        "  facial-align admin health"
    ),
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.version_option(version=__version__, prog_name="facial-align")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    envvar="FACIALIGN_CONFIG",
    help="Path to a .env configuration file. Overrides environment variables.",
    metavar="PATH",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    envvar="FACIALIGN_VERBOSE",
    help="Enable verbose debug output.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress non-essential output. Errors are always shown.",
)
@click.option(
    "--json-output",
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    envvar="FACIALIGN_JSON",
    help="Output results as JSON (useful for scripting and CI pipelines).",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config: Optional[Path],
    verbose: bool,
    quiet: bool,
    json_output: bool,
) -> None:
    """Facial Align surgical planning platform CLI."""
    ctx.ensure_object(dict)

    # Configure logging level based on flags
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    ctx.obj = AppContext(
        config_path=config,
        verbose=verbose,
        quiet=quiet,
        json_output=json_output,
    )

    if verbose and not json_output:
        console.print(
            f"[dim]facial-align v{__version__} | verbose mode | "
            f"config={config or 'default'}[/dim]"
        )


# ─── Subcommand registration ──────────────────────────────────────────────────

def _register_subcommands() -> None:
    """Import and attach all subcommand groups."""
    from cli.case_commands import case
    from cli.model_commands import model
    from cli.pipeline_commands import pipeline
    from cli.admin_commands import admin

    cli.add_command(case)
    cli.add_command(model)
    cli.add_command(pipeline)
    cli.add_command(admin)

    # Alias `export` at the top level for convenience
    from cli.case_commands import export_case
    cli.add_command(export_case, name="export")


_register_subcommands()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Console script entry point registered in pyproject.toml."""
    cli(prog_name="facial-align")


if __name__ == "__main__":
    main()
