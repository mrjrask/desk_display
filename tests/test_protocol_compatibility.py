import pytest

import protocol
from protocol_versions import NETWORK_PROTOCOL_VERSION
from schema_migrations import migrate_client_config, migrate_server_config


def test_compatible_registration_advertises_all_versions():
    request = protocol.client_registration("display-1")
    response = protocol.registration_response(request)

    assert request["protocol_version"] == NETWORK_PROTOCOL_VERSION
    assert request["client_software_version"]
    assert response["accepted"] is True
    assert response["server_software_version"]
    assert all(
        response[key] == 1
        for key in (
            "manifest_schema_version",
            "render_package_schema_version",
            "server_config_schema_version",
            "client_config_schema_version",
        )
    )
    assert response["playlist_schema_version"] == 2


@pytest.mark.parametrize(
    "protocol_version", [NETWORK_PROTOCOL_VERSION - 1, NETWORK_PROTOCOL_VERSION + 1]
)
def test_unsupported_old_and_new_clients_are_rejected(protocol_version):
    registration = {
        "protocol_version": protocol_version,
        "client_software_version": "test",
    }
    with pytest.raises(protocol.IncompatibleClientError) as caught:
        protocol.registration_response(registration)

    assert caught.value.as_response()["error"] == "incompatible_protocol_version"
    assert caught.value.as_response()["accepted_protocol_versions"] == [NETWORK_PROTOCOL_VERSION]


def test_cached_offline_operation_survives_server_upgrade():
    cached = protocol.build_manifest(cache_complete=True, package_url="package.zip")
    cached["server_software_version"] = "99.0"

    assert protocol.cached_manifest_usable_offline(cached)
    assert not protocol.cached_manifest_usable_offline({**cached, "cache_complete": False})
    assert not protocol.cached_manifest_usable_offline(
        {**cached, "manifest_schema_version": 99}
    )


def test_standalone_v01_configs_migrate_losslessly():
    standalone = {"display": {"width": 320}, "server_url": "http://desk.local"}

    for migrated in (migrate_server_config(standalone), migrate_client_config(standalone)):
        assert migrated["display"] == standalone["display"]
        assert migrated["server_url"] == standalone["server_url"]
        assert migrated["schema_version"] == 1
        assert migrated["migrated_from"] == "standalone-v0.1"
