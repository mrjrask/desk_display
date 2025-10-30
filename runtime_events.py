"""Shared runtime events for long-running desk display services."""

from __future__ import annotations

import threading

# A shared shutdown event allows auxiliary entry points (main service, CLI
# preview, etc.) to cooperate when an exit is requested.  The event is intended
# to be imported as ``_shutdown_event`` in modules that already use that name to
# minimise code churn.
shutdown_event = threading.Event()

