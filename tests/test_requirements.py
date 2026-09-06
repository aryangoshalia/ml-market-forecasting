"""Keep pyproject.toml and the requirements files from drifting apart.

pyproject declares supported ranges for packaging; requirements.txt pins the exact
versions the project was tested against. Both are real, and a mismatch between them is
the kind of thing that only shows up on someone else's machine.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from market_forecast.config import project_root

RUNTIME = project_root() / "requirements.txt"
DEV = project_root() / "requirements-dev.txt"
PYPROJECT = project_root() / "pyproject.toml"


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-r"):
            continue
        name, _, pinned = line.partition("==")
        pins[canonical(name)] = pinned
    return pins


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.split("[")[0]).strip().lower()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runtime_pins() -> dict[str, str]:
    return read_pins(RUNTIME)


@pytest.fixture(scope="module")
def dev_pins() -> dict[str, str]:
    return read_pins(DEV)


class TestRequirementsFiles:
    def test_both_files_exist(self):
        assert RUNTIME.exists() and DEV.exists()

    def test_dev_includes_the_runtime_set(self):
        assert "-r requirements.txt" in DEV.read_text(encoding="utf-8")

    def test_every_runtime_entry_is_pinned_exactly(self, runtime_pins):
        for name, pinned in runtime_pins.items():
            assert pinned, f"{name} is not pinned to an exact version"
            Version(pinned)

    def test_every_dev_entry_is_pinned_exactly(self, dev_pins):
        for name, pinned in dev_pins.items():
            assert pinned, f"{name} is not pinned to an exact version"
            Version(pinned)

    def test_numpy_stays_below_2(self, runtime_pins):
        # shap and hmmlearn still have rough edges against numpy 2.x
        assert Version(runtime_pins["numpy"]) < Version("2.0")


class TestMatchesPyproject:
    def test_runtime_covers_every_declared_dependency(self, pyproject, runtime_pins):
        declared = {canonical(Requirement(d).name) for d in pyproject["project"]["dependencies"]}
        assert declared <= set(runtime_pins), declared - set(runtime_pins)

    def test_runtime_declares_nothing_extra(self, pyproject, runtime_pins):
        declared = {canonical(Requirement(d).name) for d in pyproject["project"]["dependencies"]}
        assert set(runtime_pins) <= declared, set(runtime_pins) - declared

    def test_dev_covers_every_declared_dev_dependency(self, pyproject, dev_pins):
        declared = {
            canonical(Requirement(d).name)
            for d in pyproject["project"]["optional-dependencies"]["dev"]
        }
        assert declared <= set(dev_pins), declared - set(dev_pins)

    def test_pins_satisfy_the_declared_ranges(self, pyproject, runtime_pins, dev_pins):
        pins = {**runtime_pins, **dev_pins}
        declared = (
            pyproject["project"]["dependencies"]
            + pyproject["project"]["optional-dependencies"]["dev"]
        )
        for entry in declared:
            requirement = Requirement(entry)
            pinned = pins.get(canonical(requirement.name))
            if pinned and requirement.specifier:
                assert requirement.specifier.contains(pinned), (
                    f"{requirement.name}=={pinned} violates {requirement.specifier}"
                )


class TestInstalledEnvironmentMatchesPins:
    def test_the_running_environment_uses_the_pinned_versions(self, runtime_pins):
        from importlib.metadata import PackageNotFoundError, version

        mismatched = []
        for name, pinned in runtime_pins.items():
            try:
                installed = version(name)
            except PackageNotFoundError:
                mismatched.append(f"{name} is not installed")
                continue
            if installed != pinned:
                mismatched.append(f"{name}: pinned {pinned}, installed {installed}")
        assert not mismatched, mismatched
