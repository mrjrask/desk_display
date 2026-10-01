"""Update/restart commands queued for display clients (remote_display/client_commands.py)."""
from __future__ import annotations

import subprocess

import pytest

from remote_display import client_commands
from remote_display.client_commands import (
    COMMAND_TIMEOUT_SECONDS,
    CommandError,
    CommandPendingError,
    CommandRunner,
    CommandStore,
    clean_output,
    parse_heartbeat_commands,
)
from remote_display.models import ModelValidationError


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def store(tmp_path):
    return CommandStore(tmp_path / "commands.json", clock=Clock())


def test_queue_deliver_and_report(store):
    command = store.queue("office", "update", actor="jason")
    assert command["state"] == "pending" and command["requested_by"] == "jason"
    assert store.take_pending("den") == []
    assert store.take_pending("office") == [{"id": command["id"], "action": "update"}]
    assert store.take_pending("office") == []  # delivered once
    assert store.for_client("office")[0]["state"] == "delivered"
    store.record_results("office", [{"id": command["id"], "status": "succeeded", "exit_code": 0,
                                     "output": "Already up to date."}])
    done = store.for_client("office")[0]
    assert done["state"] == "succeeded" and done["output"] == "Already up to date."
    # A second report for the same command, or one for another client's, changes nothing.
    store.record_results("office", [{"id": command["id"], "status": "failed", "exit_code": 1, "output": "x"}])
    store.record_results("den", [{"id": command["id"], "status": "failed", "exit_code": 1, "output": "x"}])
    assert store.for_client("office")[0]["state"] == "succeeded"
    assert store.for_client("den") == []


def test_one_unfinished_command_per_action(store):
    store.queue("office", "update")
    with pytest.raises(CommandPendingError):
        store.queue("office", "update")
    store.queue("office", "restart")  # a different action is fine
    with pytest.raises(CommandError):
        store.queue("office", "rm -rf /")
    with pytest.raises(ModelValidationError):
        store.queue("../etc", "update")


def test_unanswered_commands_expire(store):
    queued = store.queue("office", "update")
    store._clock.now += COMMAND_TIMEOUT_SECONDS + 1
    expired = store.for_client("office")[0]
    assert expired["state"] == "expired" and "upgrade.sh" in expired["output"]
    assert store.take_pending("office") == []
    store.queue("office", "update")  # the expired one no longer blocks a retry
    # A late result still lands on the expired command.
    store.record_results("office", [{"id": queued["id"], "status": "succeeded", "exit_code": 0, "output": "ok"}])
    assert [c["state"] for c in store.for_client("office")] == ["pending", "succeeded"]


def test_history_is_bounded(store):
    for _ in range(client_commands.MAX_PER_CLIENT + 5):
        command = store.queue("office", "restart")
        store.take_pending("office")
        store.record_results("office", [{"id": command["id"], "status": "succeeded", "exit_code": 0,
                                         "output": None}])
    assert len(store.for_client("office")) == client_commands.MAX_PER_CLIENT


def test_heartbeat_document_validation():
    good = {"id": "0123456789abcdef", "status": "failed", "exit_code": 1, "output": "boom"}
    assert parse_heartbeat_commands({"version": 1, "results": [good]}) == [good]
    assert parse_heartbeat_commands({"version": 1}) == []
    for bad in ({"version": 2, "results": []}, {"version": 1, "extra": 1}, [],
                {"version": 1, "results": [{**good, "id": "nope"}]},
                {"version": 1, "results": [{**good, "status": "running"}]},
                {"version": 1, "results": [{**good, "exit_code": "1"}]},
                {"version": 1, "results": [{**good, "command": "ls"}]},
                {"version": 1, "results": [good] * 21}):
        with pytest.raises(ModelValidationError):
            parse_heartbeat_commands(bad)


def test_output_is_bounded_and_drops_url_credentials():
    assert clean_output("From https://user:ghp_secret@github.com/x/y") == "From https://github.com/x/y"
    long = clean_output("x" * 10_000)
    assert len(long) == client_commands.MAX_OUTPUT_CHARS and long.startswith("…")


class FakeGit:
    def __init__(self, heads=("abc1234", "def5678"), pull_code=0, pull_output="Fast-forward"):
        self.heads = list(heads)
        self.pull_code = pull_code
        self.pull_output = pull_output
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        if argv[3:] == ["rev-parse", "--short", "HEAD"]:
            return subprocess.CompletedProcess(argv, 0, self.heads.pop(0) + "\n", "")
        assert argv[3:] == ["pull", "--ff-only"]
        return subprocess.CompletedProcess(argv, self.pull_code, self.pull_output, "")


def runner(tmp_path, git, restart=None):
    return CommandRunner(tmp_path / "outbox.json", project_dir=tmp_path, restart=restart, run=git,
                         background=False)


def test_update_runs_git_pull_in_the_checkout(tmp_path):
    git = FakeGit()
    commands = runner(tmp_path, git)
    commands.handle([{"id": "0123456789abcdef", "action": "update"}])
    assert git.calls[1][0] == ["git", "-C", str(tmp_path), "pull", "--ff-only"]
    [result] = commands.results()
    assert result["status"] == "succeeded" and result["exit_code"] == 0
    assert result["output"].startswith("Updated abc1234 → def5678. Restart the client")
    # A redelivered command is never run twice, even after the result was sent.
    commands.acknowledge(["0123456789abcdef"])
    commands.handle([{"id": "0123456789abcdef", "action": "update"}])
    assert commands.results() == [] and len(git.calls) == 3


def test_update_reports_git_failures(tmp_path):
    git = FakeGit(heads=("abc1234",), pull_code=1, pull_output="fatal: Not possible to fast-forward")
    commands = runner(tmp_path, git)
    commands.handle([{"id": "0123456789abcdef", "action": "update"}])
    [result] = commands.results()
    assert result == {"id": "0123456789abcdef", "status": "failed", "exit_code": 1,
                      "output": "fatal: Not possible to fast-forward"}


def test_only_allow_listed_actions_run(tmp_path):
    git = FakeGit()
    commands = runner(tmp_path, git)
    commands.handle([{"id": "0123456789abcdef", "action": "shell", "argv": ["rm", "-rf", "/"]},
                     {"id": "bad id", "action": "update"}, "junk"])
    assert git.calls == []
    [result] = commands.results()
    assert result["status"] == "failed" and "does not know" in result["output"]


def test_restart_result_is_sent_by_the_next_process(tmp_path, monkeypatch):
    restarts = []
    commands = runner(tmp_path, FakeGit(), restart=lambda: restarts.append(True))
    commands.handle([{"id": "0123456789abcdef", "action": "restart"}])
    assert restarts == [True]
    assert commands.results() == []  # not from the process that is exiting
    monkeypatch.setattr(client_commands.os, "getpid", lambda: -1)
    restarted = runner(tmp_path, FakeGit())
    assert restarted.results() == [{"id": "0123456789abcdef", "status": "succeeded", "exit_code": 0,
                                    "output": "The client restarted."}]


def test_restart_without_a_service_fails_cleanly(tmp_path):
    commands = runner(tmp_path, FakeGit())
    commands.handle([{"id": "0123456789abcdef", "action": "restart"}])
    assert commands.results()[0]["status"] == "failed"


def test_unanswered_upgrades_wait_for_upgrade_sh(store):
    upgrade = store.queue("office", "upgrade")
    store.take_pending("office")
    store._clock.now += COMMAND_TIMEOUT_SECONDS + 1
    assert store.for_client("office")[0]["state"] == "delivered"  # pip on a Pi can take a while
    store._clock.now += client_commands.UPGRADE_TIMEOUT_SECONDS
    assert store.for_client("office")[0]["state"] == "expired"
    assert upgrade["action"] == "upgrade"


class FakeScripts:
    def __init__(self, code=0, stdout="", stderr=""):
        self.code, self.stdout, self.stderr = code, stdout, stderr
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, self.code, self.stdout, self.stderr)


def project(tmp_path, *scripts):
    root = tmp_path / "project"
    for script in scripts:
        (root / script).parent.mkdir(parents=True, exist_ok=True)
        (root / script).write_text("#!/usr/bin/env bash\n")
    return root


@pytest.mark.parametrize("action, script", sorted(client_commands.SCRIPTS.items()))
def test_maintenance_scripts_run_and_report_output(tmp_path, action, script):
    run = FakeScripts(stdout="Total freed: 12 MiB\n")
    root = project(tmp_path, script)
    commands = CommandRunner(tmp_path / "outbox.json", project_dir=root, run=run, background=False)
    commands.handle([{"id": "0123456789abcdef", "action": action}])
    [(argv, kwargs)] = run.calls
    assert argv == ["bash", str(root / script)]
    assert kwargs["cwd"] == str(root) and kwargs["stdin"] is subprocess.DEVNULL and kwargs["timeout"]
    assert commands.results() == [{"id": "0123456789abcdef", "status": "succeeded", "exit_code": 0,
                                   "output": "Total freed: 12 MiB"}]


def test_maintenance_script_failures_and_missing_scripts(tmp_path):
    run = FakeScripts(code=1, stderr="Permission denied")
    root = project(tmp_path, "scripts/reset_screenshots.sh")
    commands = CommandRunner(tmp_path / "outbox.json", project_dir=root, run=run, background=False)
    commands.handle([{"id": "0123456789abcdef", "action": "reset_screenshots"},
                     {"id": "1123456789abcdef", "action": "clear_caches"}])
    assert len(run.calls) == 1  # clear-caches.sh is missing in this checkout
    failed, missing = commands.results()
    assert failed["status"] == "failed" and failed["exit_code"] == 1 and failed["output"] == "Permission denied"
    assert missing["status"] == "failed" and "clear-caches.sh is missing" in missing["output"]


def upgrade_runner(tmp_path, run, clock=None):
    root = project(tmp_path, client_commands.UPGRADE_SCRIPT, client_commands.RUN_LOGGED_SCRIPT)
    return CommandRunner(tmp_path / "cache" / "outbox.json", project_dir=root, run=run, background=False,
                         clock=clock or Clock()), root


def test_upgrade_runs_outside_the_service_and_reports_from_its_log(tmp_path, monkeypatch):
    run = FakeScripts()
    commands, root = upgrade_runner(tmp_path, run)
    commands.handle([{"id": "0123456789abcdef", "action": "upgrade"}])
    [(argv, kwargs)] = run.calls
    assert argv[:3] == ["sudo", "-n", "systemd-run"] and "--collect" in argv
    assert "--unit=desk-display-upgrade-0123456789abcdef" in argv
    split = argv.index("--")
    log, status = tmp_path / "cache" / "command_jobs" / "upgrade-0123456789abcdef.log", \
        tmp_path / "cache" / "command_jobs" / "upgrade-0123456789abcdef.status"
    assert argv[split + 1:] == ["bash", str(root / client_commands.RUN_LOGGED_SCRIPT), str(log), str(status),
                                "bash", str(root / client_commands.UPGRADE_SCRIPT)]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert commands.results() == []  # still running

    # upgrade.sh restarts the client; the new process finds the exit code.
    log.write_text("Upgrading the client install\nUpgrade complete (client).\n")
    status.write_text("0\n")
    monkeypatch.setattr(client_commands.os, "getpid", lambda: -1)
    restarted = CommandRunner(tmp_path / "cache" / "outbox.json", project_dir=root, run=run, background=False)
    assert restarted.results() == [{"id": "0123456789abcdef", "status": "succeeded", "exit_code": 0,
                                    "output": "Upgrading the client install\nUpgrade complete (client)."}]
    assert not log.exists() and not status.exists()
    restarted.acknowledge(["0123456789abcdef"])
    assert restarted.results() == []


def test_upgrade_failures(tmp_path):
    clock = Clock()
    commands, _root = upgrade_runner(tmp_path, FakeScripts(code=1, stderr="sudo: a password is required"), clock)
    commands.handle([{"id": "0123456789abcdef", "action": "upgrade"}])
    [result] = commands.results()
    assert result["status"] == "failed" and "passwordless sudo" in result["output"]
    assert "a password is required" in result["output"]

    started, _root = upgrade_runner(tmp_path / "second", FakeScripts(), clock)
    started.handle([{"id": "1123456789abcdef", "action": "upgrade"}])
    jobs = tmp_path / "second" / "cache" / "command_jobs"
    (jobs / "upgrade-1123456789abcdef.log").write_text("pip install ...\n")
    assert started.results() == []
    clock.now += client_commands.UPGRADE_TIMEOUT_SECONDS + 1
    [overdue] = started.results()
    assert overdue["status"] == "failed" and overdue["exit_code"] is None
    assert overdue["output"].startswith("No result after 2 h") and "pip install" in overdue["output"]

    failed, _root = upgrade_runner(tmp_path / "third", FakeScripts(), clock)
    failed.handle([{"id": "2123456789abcdef", "action": "upgrade"}])
    jobs = tmp_path / "third" / "cache" / "command_jobs"
    (jobs / "upgrade-2123456789abcdef.log").write_text("error: externally-managed-environment\n")
    (jobs / "upgrade-2123456789abcdef.status").write_text("1\n")
    [result] = failed.results()
    assert result["status"] == "failed" and result["exit_code"] == 1


def test_run_logged_writes_output_and_exit_code(tmp_path):
    from pathlib import Path

    helper = Path(__file__).resolve().parents[1] / client_commands.RUN_LOGGED_SCRIPT
    log, status = tmp_path / "out.log", tmp_path / "out.status"
    result = subprocess.run(["bash", str(helper), str(log), str(status), "bash", "-c", "echo hi; echo err >&2; exit 3"],
                            check=False)
    assert result.returncode == 3
    assert log.read_text() == "hi\nerr\n" and status.read_text() == "3\n"
