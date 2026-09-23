"""Canonical identifiers shared by the Feed server and screenshot uploader."""


def sanitize_feed_id(value: str) -> str:
    """Return a stable, filesystem- and URL-safe Feed identifier."""

    safe = value.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "unknown"
