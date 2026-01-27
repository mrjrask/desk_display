"""User-editable overrides for screenshot storage directories."""
from __future__ import annotations

from typing import Optional

# Set this to an absolute path (e.g. "/home/pi/desk_display_display_hat_mini/screenshots")
# to force the application to write screenshots there. Leave it as ``None`` to
# let the auto-detection logic choose a writable location.
SCREENSHOT_DIR: Optional[str] = None
