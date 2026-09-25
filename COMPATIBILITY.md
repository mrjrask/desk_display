# Versioning and compatibility

The canonical constants are in `protocol_versions.py`. That module must remain
dependency-light so registration can run before server configuration or client
display hardware is initialized. Application releases use a string; wire and
JSON schemas use independently bumped integers.

## Rules

* A server accepts only protocol versions listed in
  `SERVER_ACCEPTED_CLIENT_PROTOCOL_VERSIONS`. Software release strings do not
  affect compatibility.
* Registration contains `protocol_version` and `client_software_version`.
  A successful response identifies the server release and every current schema.
* An incompatible registration is rejected before content is returned. HTTP
  adapters should return status **409** and the structured
  `incompatible_protocol_version` response from `IncompatibleClientError`.
* A client accepts only explicitly listed manifest and render-package schemas.
  It must reject either document when its version is unsupported.
* Rejection by a newer server does not invalidate downloaded data. An older
  client may operate offline when its cached manifest/package are complete and
  their embedded schema versions remain supported; it may not fetch updates.
* Playlists and server/client configurations migrate one schema step at a time.
  Unknown future versions and missing migration steps fail closed rather than
  being guessed. Playlist v1 is converted to the named-playlist v2 structure.
* Standalone v0.1 server and client configuration files had no schema marker.
  They migrate losslessly to schema v1 by retaining their keys and adding
  `schema_version: 1` and `migrated_from: "standalone-v0.1"`.

Manifests carry the server software, manifest, render-package, playlist,
server-configuration, and client-configuration versions. A software upgrade
which does not change a wire/schema constant is therefore compatible.

## Remote display documents

`remote_display/models.py` defines the documents clients and the server
exchange. Each has an explicit wire envelope, `{"type": ..., "version": N,
...fields}`, and `from_wire` accepts only a known type and version with
exactly that version's fields. Every identifier, string, number and list is
bounded, and screen IDs and display profiles are checked against
`screens_catalog.py` and `display_profiles.py`.

| Type | Version | Purpose |
| --- | --- | --- |
| `client_capabilities` | 1 | Versions, stable client ID, display profile, canonical logical size, formats, colour modes, render-package versions, animation, touch and buttons, optional hardware description. |
| `client_demand` | 1 | Playlist revision, required and alternate screens, touch-expansion targets, package capabilities, sync interval. |
| `client_status` | 1 | Current screen and playlist, accepted revisions, sync and cache ages, playback state, diagnostic physical rotation, summarized recent errors. |
| `render_key` | 1 | Canonical artifact identity: screen, render profile, canonical size, colour mode, style, data and renderer revisions, and a client scope only for intentionally client-specific output. |

A render key never contains playlist order, playback position, physical
rotation or (unless the output is client-specific) the client ID, so clients
with equivalent demand share artifacts. `plan_renders` validates every
client's capabilities and demand before it returns any work, so an
incompatible client fails before anything is scheduled. Bump a document's
version, and keep parsing the old one where needed, for any incompatible
change to its fields.
