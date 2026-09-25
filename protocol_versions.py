"""Wire and document schema versions shared by Desk Display processes.

This module deliberately imports nothing from the application.  In particular,
importing it must not load :mod:`config`, Flask, Pillow, or a display driver.
Protocol and schema versions are integers: bump one only for an incompatible
change to the corresponding JSON contract.  ``APPLICATION_VERSION`` describes
the software release and is not used to decide wire compatibility.
"""

# The version of the Desk Display software release.
APPLICATION_VERSION = "0.1"

# Client/server registration and transport protocol.
NETWORK_PROTOCOL_VERSION = 1

# Server-produced manifest document.
MANIFEST_SCHEMA_VERSION = 1

# Downloadable package referenced by a manifest.
RENDER_PACKAGE_SCHEMA_VERSION = 1

# Playlist/schedule document.  Version 1 is the legacy flat sequence and
# version 2 is the named-playlist representation.
PLAYLIST_SCHEMA_VERSION = 2

# Server-owned and client-owned configuration documents respectively.
SERVER_CONFIG_SCHEMA_VERSION = 1
CLIENT_CONFIG_SCHEMA_VERSION = 1


__all__ = [
    "APPLICATION_VERSION",
    "CLIENT_CONFIG_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "NETWORK_PROTOCOL_VERSION",
    "PLAYLIST_SCHEMA_VERSION",
    "RENDER_PACKAGE_SCHEMA_VERSION",
    "SERVER_CONFIG_SCHEMA_VERSION",
]
