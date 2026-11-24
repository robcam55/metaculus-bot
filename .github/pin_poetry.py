"""Writes .github/poetry-requirements.txt: Poetry at one version and every package it
needs on the servers, each pinned to an exact version and to the sha256 hashes of its
wheels (a wheel is a package file that installs without running any build code). The
workflows install Poetry from that file with pip's --require-hashes, so a later release
of one of them can't reach a job that holds the keys.

Run by hand, never in a workflow:

    python .github/pin_poetry.py 2.5.1              (releases at least 14 days old)
    python .github/pin_poetry.py 2.5.1 2026-11-10   (releases uploaded before that day, UTC)

It needs a Python that has pip 26 or later, on any operating system. It reads pypi.org
and files.pythonhosted.org, uses no key, installs nothing and writes only that one file.

Why a date. A hash only proves a file is the one that was pinned; it says nothing about
a release that was bad from its first hour. So releases from the last 14 days are left
out, which gives a bad one time to be noticed and withdrawn. With the same date the
script writes the same pins again, unless a publisher has withdrawn a release since.

Why not plain pip. The servers are Linux x86_64 with CPython 3.11. pip can pick wheels
for another machine (--platform, --python-version), but it still decides which
dependencies apply (their "environment markers") for the machine it runs on. On Windows
that adds colorama and pywin32-ctypes and leaves out SecretStorage and jeepney, which
Linux needs. So this script has pip answer the markers as a server would, and then
checks the answer without pip: every dependency that applies on a server is in the set
at a version that fits, nothing else is, and each hash is one PyPI lists for a wheel a
server can install. Last, pip reads the finished file back in --require-hashes mode.
"""

import json
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pip import __version__ as pip_version
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.tags import compatible_tags, cpython_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename

OUT = Path(__file__).with_name("poetry-requirements.txt")

# What packaging.markers.default_environment() returns on the servers: ubuntu-latest
# with actions/setup-python "3.11". No marker in Poetry's tree reads the two kernel
# fields; the script stops if one ever does.
SERVER = {
    "implementation_name": "cpython",
    "implementation_version": "3.11.18",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "6.8.0",
    "platform_system": "Linux",
    "platform_version": "#1 SMP",
    "python_full_version": "3.11.18",
    "python_version": "3.11",
    "sys_platform": "linux",
}
KERNEL_FIELDS = ("platform_release", "platform_version")

# Versions are chosen among those with a wheel for glibc 2.28 or older (Ubuntu 20.04 has
# 2.31, 24.04 has 2.39), so an older or newer server image still finds one. Hashes are
# then listed for every wheel of that version that a glibc Linux x86_64 with CPython
# 3.11 could pick, whatever its glibc.
OLD_GLIBC = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
OLD_GLIBC += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]
ANY_GLIBC = [f"manylinux_2_{minor}_x86_64" for minor in range(60, 28, -1)] + OLD_GLIBC
CROSS = ["--only-binary=:all:", "--python-version=3.11", "--implementation=cp", "--abi=cp311"]
CROSS += [f"--platform={platform}" for platform in OLD_GLIBC]

# pip, with the markers answered for a server
SHIM = (
    "import json, sys\n"
    "from pip._vendor.packaging import markers\n"
    "server = json.loads(sys.argv[1])\n"
    "markers.default_environment = lambda: dict(server)\n"
    "from pip._internal.cli.main import main\n"
    "sys.exit(main(sys.argv[2:]))\n"
)


def pip_as_server(*args: str) -> None:
    command = [sys.executable, "-c", SHIM, json.dumps(SERVER), *args]
    subprocess.run(command, check=True)


def resolve(version: str, before: str) -> list[dict]:
    """pip's choice of versions for poetry==version on a server, among the files uploaded
    before that day, as its report lists them."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        pip_as_server(
            "install", "--dry-run", "--ignore-installed", "--no-cache-dir", "--quiet",
            "--disable-pip-version-check", *CROSS, f"--uploaded-prior-to={before}T00:00:00Z",
            f"--target={Path(tmp) / 'target'}", f"--report={report}", f"poetry=={version}",
        )  # fmt: skip
        return json.loads(report.read_text(encoding="utf-8"))["install"]


def applies(requirement: Requirement, extras: frozenset[str]) -> bool:
    """Whether a dependency applies on a server. Every field is given, so the answer
    doesn't depend on the machine this runs on, or on the shim having worked."""
    marker = requirement.marker
    if marker is None:
        return True
    if any(field in str(marker) for field in KERNEL_FIELDS):
        raise SystemExit(f"{requirement} reads the kernel's version, which this script only guesses")
    return any(marker.evaluate({**SERVER, "extra": extra}) for extra in ("", *sorted(extras)))


def check_closure(root: str, pins: dict[str, str], requires: dict[str, list[Requirement]]) -> None:
    """Walks the dependencies from Poetry down, as they apply on a server."""
    seen: set[tuple[str, frozenset[str]]] = set()
    queue = [(root, frozenset())]
    while queue:
        name, extras = queue.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        for requirement in requires[name]:
            if not applies(requirement, extras):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in pins:
                raise SystemExit(f"{name} needs {requirement} on a server, and pip left it out")
            if not requirement.specifier.contains(pins[dependency], prereleases=True):
                raise SystemExit(f"{name} needs {requirement}; pip chose {pins[dependency]}")
            queue.append((dependency, frozenset(requirement.extras)))
    unneeded = sorted(set(pins) - {name for name, _ in seen})
    if unneeded:
        raise SystemExit(f"pip chose packages that nothing needs on a server: {unneeded}")


def wheels_for_a_server(name: str, version: str, before: str) -> list[dict]:
    """PyPI's wheels of this release that CPython 3.11 on glibc Linux x86_64 can install,
    uploaded before that day."""
    fits = set(cpython_tags((3, 11), ["cp311"], ANY_GLIBC)) | set(compatible_tags((3, 11), "cp311", ANY_GLIBC))
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as reply:
        files = json.load(reply)["urls"]
    wheels = [
        f
        for f in files
        if f["packagetype"] == "bdist_wheel"
        and parse_wheel_filename(f["filename"])[3] & fits
        and f["upload_time_iso_8601"][:10] < before
    ]
    for wheel in wheels:
        if wheel.get("yanked"):
            raise SystemExit(f"{wheel['filename']} was withdrawn by its publisher: {wheel.get('yanked_reason')}")
    return wheels


def main(version: str, before: str) -> None:
    datetime.strptime(before, "%Y-%m-%d")  # stops on anything but a date
    chosen = resolve(version, before)
    pins: dict[str, str] = {}
    requires: dict[str, list[Requirement]] = {}
    pip_hash: dict[str, str] = {}
    for item in chosen:
        metadata = item["metadata"]
        name = canonicalize_name(metadata["name"])
        pins[name] = metadata["version"]
        requires[name] = [Requirement(text) for text in metadata.get("requires_dist") or []]
        pip_hash[name] = item["download_info"]["archive_info"]["hashes"]["sha256"]
        needs_python = metadata.get("requires_python")
        if needs_python and not SpecifierSet(needs_python).contains(SERVER["python_full_version"]):
            raise SystemExit(f"{name} {pins[name]} needs Python {needs_python}")
        if item.get("is_yanked"):
            raise SystemExit(f"{name} {pins[name]} was withdrawn by its publisher")
    if pins.get("poetry") != version:
        raise SystemExit(f"pip did not choose poetry {version}: {pins.get('poetry')}")
    check_closure("poetry", pins, requires)

    entries = []
    hash_count = 0
    print(f"{'package':<22}{'version':<16}{'uploaded':<12}hashes")
    for name in sorted(pins):
        wheels = wheels_for_a_server(name, pins[name], before)
        hashes = sorted({wheel["digests"]["sha256"] for wheel in wheels})
        if pip_hash[name] not in hashes:
            raise SystemExit(f"{name} {pins[name]}: the file pip chose is not among PyPI's wheels for a server")
        uploaded = max(wheel["upload_time_iso_8601"] for wheel in wheels)[:10]
        print(f"{name:<22}{pins[name]:<16}{uploaded:<12}{len(hashes)}")
        lines = [f"{name}=={pins[name]}"] + [f"    --hash=sha256:{digest}" for digest in hashes]
        entries.append(" \\\n".join(lines))
        hash_count += len(hashes)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = f"""\
# Poetry {version} and the packages it needs on the servers (Linux x86_64, CPython 3.11),
# each pinned to one version and to the sha256 hashes of its wheels. Every workflow
# installs Poetry from this file with pip's --require-hashes, so a later release of one
# of them can't run in a job (README, "Why it's built this way").
#
# Written by a program; don't edit it by hand. These exact pins come from
#
#     python .github/pin_poetry.py {version} {before}
#
# and without the date the same command takes releases at least 14 days old (to move
# Poetry to another version, change the version in tests/check_rc_bot.py too). It runs
# pip's own resolver for poetry=={version}, as
#     pip install --dry-run --ignore-installed --only-binary=:all: --python-version=3.11
#         --implementation=cp --abi=cp311 --platform=manylinux_2_28_x86_64 (and older)
#         --uploaded-prior-to={before}T00:00:00Z --report=report.json poetry=={version}
# with the dependencies' environment markers answered for Linux and CPython 3.11, not
# for the machine it runs on, and takes each wheel's hashes from pypi.org.
#
# Written {today} (UTC) with pip {pip_version}, from releases uploaded before {before}:
# {len(pins)} packages, {hash_count} hashes.
"""
    OUT.write_text(header + "\n".join(entries) + "\n", encoding="utf-8", newline="\n")

    # pip reads the file back as a server would: it stops if a package that applies there
    # is missing, unpinned or has no hash
    with tempfile.TemporaryDirectory() as tmp:
        pip_as_server(
            "install", "--dry-run", "--ignore-installed", "--no-cache-dir", "--quiet",
            "--disable-pip-version-check", "--require-hashes", *CROSS,
            f"--target={Path(tmp) / 'target'}", "-r", str(OUT),
        )  # fmt: skip
    print(f"wrote {OUT.name}: {len(pins)} packages, {hash_count} hashes")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python .github/pin_poetry.py <poetry version, e.g. 2.5.1> [<day, e.g. 2026-11-10>]")
    two_weeks_ago = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else two_weeks_ago)
