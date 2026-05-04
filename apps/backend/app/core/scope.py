"""Helpers for normalizing institution scope values."""

from __future__ import annotations

from typing import Optional


def normalize_institution_code(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None
