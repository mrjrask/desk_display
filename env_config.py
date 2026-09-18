"""Safe parsing for numeric environment configuration."""

import logging
import math
import os
from typing import Optional


def env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    """Return an integer environment value, or *default* when it is unusable."""

    raw = os.environ.get(name)
    try:
        if raw is None or not raw.strip():
            raise ValueError("value is empty")
        value = int(raw)
        if minimum is not None and value < minimum:
            raise ValueError(f"value must be at least {minimum}")
    except (TypeError, ValueError) as exc:
        if raw is not None:
            logging.warning(
                "Ignoring %s=%r (%s); using default %s.", name, raw, exc, default
            )
        return default
    return value


def env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    """Return a finite float environment value, or *default* when unusable."""

    raw = os.environ.get(name)
    try:
        if raw is None or not raw.strip():
            raise ValueError("value is empty")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if minimum is not None and value < minimum:
            raise ValueError(f"value must be at least {minimum}")
    except (TypeError, ValueError) as exc:
        if raw is not None:
            logging.warning(
                "Ignoring %s=%r (%s); using default %s.", name, raw, exc, default
            )
        return default
    return value


def non_negative_env_int(name: str, default: int) -> int:
    """Return an integer environment value that is greater than or equal to zero."""

    return env_int(name, default, minimum=0)


def non_negative_env_float(name: str, default: float) -> float:
    """Return a float environment value that is greater than or equal to zero."""

    return env_float(name, default, minimum=0.0)
