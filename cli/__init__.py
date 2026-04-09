"""
Facial Align CLI — Command-line interface for the craniofacial surgical planning platform.

Provides commands for case management, model lifecycle, pipeline execution,
and administrative operations. All commands support --json output for
programmatic use and rich terminal output for interactive sessions.

Entry point: `facial-align` (registered via pyproject.toml console_scripts)
"""

from cli.main import cli

__all__ = ["cli"]
