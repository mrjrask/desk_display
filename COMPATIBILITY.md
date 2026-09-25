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
