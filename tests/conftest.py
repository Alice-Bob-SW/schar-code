import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# Resolved from this file, not from Path.cwd(), so tests run from any directory.
DATA_DIR = Path(__file__).parent / "data"
RSA100_REFERENCE_DIR = DATA_DIR / "rsa100_reference"

MBU_VERSIONS = ("delayed_mbu", "immediate_mbu")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(params=MBU_VERSIONS)
def mbu_version(request) -> ModuleType:
    """The `scharcode.bloqs.factoring.<version>` subpackage, once per version.

    For tests that hold for both versions (classical action, call graphs, qubit counts).
    The two versions export the same public classes, e.g. `mbu_version.ComputeResidue`.

    The delayed version's Phaseups are cost-only placeholders: phase checks must use
    `immediate_mbu` directly, not this fixture.
    """
    return importlib.import_module(f"scharcode.bloqs.factoring.{request.param}")


@dataclass(frozen=True)
class Rsa100Reference:
    """RSA-100 reference data from Gidney's code (see `data/rsa100_reference/PROVENANCE.md`)."""

    metadata: dict
    arrays: dict[str, np.ndarray]


@pytest.fixture(scope="session")
def rsa100_reference() -> Rsa100Reference:
    """Load the fixture on first use only, so tests that don't need it don't pay for it."""
    with open(RSA100_REFERENCE_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    with np.load(RSA100_REFERENCE_DIR / "execution_fixture.npz") as npz:
        arrays = {key: npz[key] for key in npz.files}
    return Rsa100Reference(metadata=metadata, arrays=arrays)
