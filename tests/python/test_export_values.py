"""
Test the plugin forces through the Python API against values of the InteraxFF export (tests/data, written by
generate_export_values.py), on every available platform. This is the Python counterpart of
tests/TestInteraxExportValues.h and needs neither InteraxFF nor its model.

Usage: python -m pytest test_export_values.py      (INTERAX_PLUGIN_PREFIX: where the plugin is installed)
"""
import os
import sys

import numpy as np
import pytest

PLUGIN_PREFIX = os.environ.get("INTERAX_PLUGIN_PREFIX", "/shared/joaomorado/opt/openmm-interax")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
sys.path.insert(0, f"{PLUGIN_PREFIX}/python")

import openmm  # noqa: E402

openmm.Platform.loadPluginsFromDirectory(f"{PLUGIN_PREFIX}/lib/plugins")
import interaxplugin  # noqa: E402,F401  (registers the serialization proxies)

SYSTEMS = ["cluster", "methanol_box"]
# (platform, precision, whether it computes the interactions in double precision)
PLATFORMS = [("Reference", None, True), ("CPU", None, False), ("CUDA", "mixed", False), ("CUDA", "double", True)]
AVAILABLE = {openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())}


def load(name):
    """The plugin System and the export values (box, positions, and energy and forces of each term)."""
    with open(os.path.join(DATA, name, "system.xml")) as file:
        system = openmm.XmlSerializer.deserialize(file.read())
    box, positions, terms = None, None, []
    with open(os.path.join(DATA, name, "reference.txt")) as file:
        lines = [line for line in file if line.strip() and not line.startswith("#")]
    k = 0
    while k < len(lines):
        words = lines[k].split()
        if words[0] == "box":
            box = None if words[1] == "none" else np.array(words[1:], dtype=float).reshape(3, 3)
            k += 1
        elif words[0] == "positions":
            n = int(words[1])
            positions = np.loadtxt(lines[k + 1:k + 1 + n])
            k += 1 + n
        elif words[0] == "term":
            n = len(positions)
            terms.append((words[1], int(words[2]), float(words[3]), np.loadtxt(lines[k + 1:k + 1 + n])))
            k += 1 + n
    return system, box, positions, terms


def set_overlap_method(system, exact):
    for force in system.getForces():
        for cls in (interaxplugin.InteraxExchangeForce, interaxplugin.InteraxShortRangeForce):
            if exact and cls.isinstance(force):
                cls.cast(force).setOverlapMethod(cls.Exact)


@pytest.mark.parametrize("name", SYSTEMS)
@pytest.mark.parametrize("platform,precision,double", PLATFORMS)
@pytest.mark.parametrize("exact", [False, True])
def test_export_values(name, platform, precision, double, exact):
    if platform not in AVAILABLE:
        pytest.skip(f"the {platform} platform is not available")
    if not double and not exact:
        pytest.skip("in single precision the InteraxFF order 2 scheme of the export loses precision (docs/SPEC.md)")
    energy_tolerance, force_tolerance = (1e-8, 1e-7) if double and not exact else (1e-6, 1e-5) if double else (2e-6, 5e-5)
    system, box, positions, terms = load(name)
    set_overlap_method(system, exact)
    properties = {"Precision": precision} if precision else {}
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform), properties)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    for term, group, energy, forces in terms:
        state = context.getState(getEnergy=True, getForces=True, groups={group})
        e = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        f = state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        rms = np.sqrt((forces ** 2).mean())
        assert abs(e - energy) <= energy_tolerance * abs(energy), f"{term}: energy {e} vs export {energy}"
        assert np.abs(f - forces).max() <= force_tolerance * rms, f"{term}: forces differ from the export by {np.abs(f - forces).max() / rms:.1e} of the RMS force"
