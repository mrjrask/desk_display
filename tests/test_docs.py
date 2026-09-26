"""Phase 18: the operator documentation matches the implementation.

Checks every Markdown document in the repository root and ``docs/``:

* relative links point at files that exist, and ``#anchors`` at headings;
* every repository script or file a command names exists;
* service names are the ones the installers write;
* the three env examples are linked and each names its role.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import deployment_config as dc
import service_units as su

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")])
OPERATOR_DOCS = [ROOT / name for name in ("README.md", "OPERATIONS.md", "CONFIGURATION.md")]

_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_PATH = re.compile(r"(?<![\w./-])((?:\./)?(?:scripts|Installers|docs|requirements|tests)/[\w./-]+\.(?:sh|py|md|txt))")
_SERVICE = re.compile(r"\b([a-z_-]+\.service)\b")
# Units that are not the project's own, or that the add-on installers write.
_OTHER_SERVICES = {
    "display-manager.service", "waveshare-fbcp.service", "feed_server_desk_display.service",
    "screenshot_uploader_desk_display.service", "desk_display_adsb_collector.service",
    "airplay_desk_display.service", "desk_display_waveshare_oled.service", "desk_display-kernel.service",
    "shairport-sync.service", "nqptp.service", "NetworkManager.service", "lightdm.service",
    "getty.service", "ssh.service", "systemd-networkd-wait-online.service",
}


def slug(heading: str) -> str:
    text = re.sub(r"<[^>]+>", "", heading).strip().lower()
    text = re.sub(r"[`*_~]|\[|\]\([^)]*\)", "", text)
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = _CODE_FENCE.sub("", path.read_text(encoding="utf-8"))
    found: set[str] = set()
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            base = slug(match.group(1))
            n = counts.get(base, 0)
            counts[base] = n + 1
            found.add(base if n == 0 else f"{base}-{n}")
    found |= set(re.findall(r'<a (?:name|id)="([^"]+)"', text))
    return found


def links(path: Path) -> list[str]:
    return _LINK.findall(_CODE_FENCE.sub("", path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_and_anchors_resolve(doc):
    broken = []
    for target in links(doc):
        if re.match(r"^[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (doc.parent / file_part).resolve() if file_part else doc
        if not dest.exists():
            broken.append(target)
        elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
            broken.append(target)
    assert not broken, f"{doc.name}: broken links {broken}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_named_repository_files_exist(doc):
    missing = sorted({p for p in _PATH.findall(doc.read_text(encoding="utf-8"))
                      if not (ROOT / p.removeprefix("./")).exists()})
    assert not missing, f"{doc.name} names missing files: {missing}"


@pytest.mark.parametrize("doc", OPERATOR_DOCS + sorted((ROOT / "docs").glob("*.md")),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_service_names_match_the_installers(doc):
    known = set(su.ALL_SERVICES) | _OTHER_SERVICES
    unknown = sorted(set(_SERVICE.findall(doc.read_text(encoding="utf-8"))) - known)
    assert not unknown, f"{doc.name} names services no installer writes: {unknown}"


def test_the_env_examples_are_linked_and_name_their_roles():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for example, role in ((".env.example", "standalone"), (".env.server.example", "server"),
                          (".env.client.example", "client")):
        assert f"]({example})" in readme, f"README does not link {example}"
        text = (ROOT / example).read_text(encoding="utf-8")
        assert role in text.split("\n\n", 1)[0].lower(), f"{example} does not open by naming its role"
    for role in (dc.Role.SERVER, dc.Role.CLIENT):
        assert (ROOT / f".env.{role.value}.example").read_text(encoding="utf-8") == dc.render_example(role)


def test_every_install_mode_has_a_documented_command():
    ops = (ROOT / "OPERATIONS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for mode in ("server", "client", "combined"):
        command = f"install.sh --mode {mode}"
        assert command in ops and command in readme, command


def test_restoring_v0_1_is_documented_with_the_release_commit():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "9e193dc22ce0caa88a66493dafa42f90ae3b0db9" in changelog
    assert "lightweight tag" in changelog
    ops = (ROOT / "OPERATIONS.md").read_text(encoding="utf-8")
    assert "git switch -c restore-v0.1 v0.1" in ops and "install.sh --mode standalone" in ops


def _git(*args: str) -> str | None:
    import subprocess

    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def test_the_documented_v0_1_restore_works_on_the_release_tree():
    commit = _git("rev-parse", "v0.1^{commit}")
    if commit is None:
        pytest.skip("the v0.1 tag is not in this checkout (shallow clone)")
    assert commit.strip() == "9e193dc22ce0caa88a66493dafa42f90ae3b0db9"
    assert _git("cat-file", "-t", "v0.1").strip() == "commit"  # lightweight, as documented
    installer = _git("show", "v0.1:Installers/install.sh")
    # v0.1's installer takes the profile as its first argument, as step 5 says.
    assert installer and 'profile="${1:-}"' in installer and "--mode" not in installer
    # The rotation v0.1 plays is still where the current release leaves it.
    assert _git("show", "v0.1:screens_config.json") is not None
    assert (ROOT / "screens_config.json").exists()
