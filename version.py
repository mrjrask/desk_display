"""Canonical Desk Display release version.

Protocol and persisted-data versions live in :mod:`protocol_versions`; keeping
that dependency-light module separate lets remote clients import them safely.
"""

from protocol_versions import APPLICATION_VERSION

__version__ = APPLICATION_VERSION
VERSION = __version__
