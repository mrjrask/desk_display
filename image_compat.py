"""Shared compatibility helpers for Pillow image operations."""

from PIL import Image

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    LANCZOS = Image.LANCZOS
