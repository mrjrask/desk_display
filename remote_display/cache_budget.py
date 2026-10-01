"""Size budgets for the render server's cache directories.

``images/cache/`` holds only files the server can download again (radar base
map tiles, auto-downloaded NCAA logos, MLB pitcher headshots), so
:func:`prune_to_budget` deletes its least recently used files until it fits.

``cache/`` also holds history files and databases that cannot be rebuilt, and
its two big parts (the artifact store and the local panel's client cache)
enforce their own limits, so :class:`BudgetWarning` only logs when it is over.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

from remote_display.resource_stats import directory_size

LOGGER = logging.getLogger(__name__)

# Files changed this recently may still be being written (or were just
# fetched for a render in progress), so pruning never touches them.
MIN_AGE_SECONDS = 3600.0


def _files(root: Path) -> list[tuple[float, int, Path]]:
    """``(last_used, size, path)`` for every regular file under *root*."""

    found: list[tuple[float, int, Path]] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            # atime is only coarsely updated (relatime), so
                            # the newer of the two is the best "last used".
                            found.append((max(stat.st_atime, stat.st_mtime), stat.st_size, Path(entry.path)))
                    except OSError:
                        continue
        except OSError:
            continue
    return found


def prune_to_budget(root: str | os.PathLike[str], max_bytes: int, *,
                    min_age_seconds: float = MIN_AGE_SECONDS,
                    clock: Callable[[], float] = time.time) -> list[Path]:
    """Delete the least recently used files under *root* until it fits *max_bytes*.

    Files used within *min_age_seconds* are kept even when that leaves the
    directory over budget. Returns the deleted paths.
    """

    path = Path(root)
    if not path.is_dir():
        return []
    files = _files(path)
    total = sum(size for _, size, _ in files)
    if total <= max_bytes:
        return []
    cutoff = clock() - min_age_seconds
    deleted: list[Path] = []
    for used, size, file in sorted(files, key=lambda item: item[0]):
        if total <= max_bytes:
            break
        if used > cutoff:
            break
        try:
            file.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            LOGGER.debug("Could not delete %s", file, exc_info=True)
            continue
        total -= size
        deleted.append(file)
    if deleted:
        LOGGER.info("Pruned %d file(s) from %s to keep it under %d MB", len(deleted), path, max_bytes >> 20)
    if total > max_bytes:
        LOGGER.warning("%s is %d MB, over its %d MB limit, but its remaining files were used in the last hour",
                       path, total >> 20, max_bytes >> 20)
    return deleted


class BudgetWarning:
    """Log once each time a directory goes over its limit (and again after it recovers)."""

    def __init__(self, root: str | os.PathLike[str], max_bytes: int, *, what: str) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.what = what
        self.over = False

    def check(self) -> int | None:
        """Return the directory's size in bytes, logging when it newly exceeds the limit."""

        size = directory_size(self.root)
        total = None if size is None else size["bytes"]
        over = total is not None and total > self.max_bytes
        if over and not self.over:
            LOGGER.warning("%s (%s) is %d MB, over its %d MB limit; see the Stats page for what uses it",
                           self.what, self.root, total >> 20, self.max_bytes >> 20)
        self.over = over
        return total
