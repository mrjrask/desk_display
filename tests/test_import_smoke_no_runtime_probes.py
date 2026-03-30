import importlib
import socket
import subprocess
import sys

import pytest


def _guard(*_args, **_kwargs):
    raise AssertionError("runtime probe should not run during module import")


@pytest.mark.parametrize("module_name", ["config", "utils"])
def test_import_does_not_probe_runtime(monkeypatch, module_name):
    monkeypatch.setattr(subprocess, "run", _guard)
    monkeypatch.setattr(subprocess, "check_call", _guard)
    monkeypatch.setattr(subprocess, "check_output", _guard)
    monkeypatch.setattr(socket, "getaddrinfo", _guard)
    monkeypatch.setattr(socket, "create_connection", _guard)

    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
