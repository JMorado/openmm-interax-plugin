"""
Test InteraxThreeBodyForce on its own, without InteraxFF: random sites in a periodic box, in the molecular form (one
site per molecule, the centroid of several particles, no damping) and the atomic form (one site per atom, triples
within a molecule excluded, Tang-Toennies damping), each also with the centroid damping 1 - exp(-(r/R0)^6). The energy is compared with an independent NumPy evaluation, the
Reference forces with finite differences of the energy, and every other platform with Reference.

Usage: python -m pytest test_three_body.py      (INTERAX_PLUGIN_PREFIX: where the plugin is installed)
"""
import itertools
import math
import os
import sys

import numpy as np
import pytest


import openmm  # noqa: E402

import plugin_path  # noqa: E402  (INTERAX_PLUGIN_PREFIX)
import interaxplugin as ip  # noqa: E402

BOX = 2.2           # nm
CUTOFF = 1.0        # nm, more than a third of the box: triples need not close into triangles
NUM_MOLECULES = 24  # of three particles each
PLATFORMS = [("CPU", None), ("CUDA", "mixed"), ("CUDA", "double")]
AVAILABLE = plugin_path.available_platforms()


# (atomic, centroid damping radius R0 in nm): R0 = 0.5 nm damps a large part of the triples of this box
VARIANTS = [(False, 0.0), (True, 0.0), (False, 0.5), (True, 0.5)]


def make_system(atomic, seed=1, centroid_r0=0.0):
    """A box of triatomic molecules and an InteraxThreeBodyForce on them, molecular or atomic."""
    rng = np.random.default_rng(seed)
    centres = rng.random((NUM_MOLECULES, 3)) * BOX
    positions = np.concatenate([c + 0.1 * rng.normal(size=(3, 3)) for c in centres])
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*np.eye(3) * BOX)
    for _ in positions:
        system.addParticle(12.0)
    force = ip.InteraxThreeBodyForce()
    force.setCutoffDistance(CUTOFF)
    force.setUsesPeriodicBoundaryConditions(True)
    force.setScaleFactor(1.3)
    force.setCentroidDampingRadius(centroid_r0)
    if atomic:
        force.setDampingMethod(ip.InteraxThreeBodyForce.TangToennies)
        force.setDampingScale(0.8)
        for m in range(NUM_MOLECULES):
            for i in range(3 * m, 3 * m + 3):
                force.addSite([i], rng.uniform(5e-4, 2e-3), rng.uniform(800, 1500), rng.uniform(0.02, 0.04), m)
    else:
        for m in range(NUM_MOLECULES):
            force.addSite([3 * m, 3 * m + 1, 3 * m + 2], rng.uniform(2e-3, 5e-3), rng.uniform(800, 1500))
    system.addForce(force)
    return system, force, positions


def tang_toennies_3(x):
    return 1 - math.exp(-x) * sum(x ** k / math.factorial(k) for k in range(4))


def numpy_energy(force, positions):
    """
    The energy of the force from its definition, by brute force over all triples of sites, and the sum of the
    magnitudes of the triple energies (the scale of the rounding errors, as the triples partly cancel).
    """
    sites, params = [], []
    for i in range(force.getNumSites()):
        particles, alpha, frequency, width, molecule, _ = force.getSiteParameters(i)
        sites.append(positions[list(particles)].mean(axis=0))
        params.append((alpha, frequency, width, molecule if molecule >= 0 else -1 - i))
    sites = np.array(sites)
    damped = force.getDampingMethod() == ip.InteraxThreeBodyForce.TangToennies
    beta, scale, cutoff = force.getDampingScale(), force.getScaleFactor(), force.getCutoffDistance()
    r0 = force.getCentroidDampingRadius()

    def delta(i, j):
        d = sites[j] - sites[i]
        return d - BOX * np.round(d / BOX)

    energy = magnitude = 0.0
    for i, j, k in itertools.combinations(range(len(sites)), 3):
        if len({params[i][3], params[j][3], params[k][3]}) < 3:
            continue
        a, b, c = delta(i, j), delta(i, k), delta(j, k)
        ra, rb, rc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        if max(ra, rb, rc) >= cutoff:
            continue
        (a1, w1, s1, _), (a2, w2, s2, _), (a3, w3, s3, _) = params[i], params[j], params[k]
        c9 = scale * 1.5 * a1 * a2 * a3 * w1 * w2 * w3 * (w1 + w2 + w3) / ((w1 + w2) * (w2 + w3) * (w3 + w1))
        cosines = (a @ b) * (-a @ c) * (b @ c) / (ra * rb * ra * rc * rb * rc)  # angles at A, B and C
        e = c9 * (1 + 3 * cosines) / (ra * rb * rc) ** 3
        if damped:
            e *= tang_toennies_3(2 * beta * ra / (s1 + s2)) * tang_toennies_3(2 * beta * rb / (s1 + s3)) * tang_toennies_3(2 * beta * rc / (s2 + s3))
        if r0 > 0:
            e *= -math.expm1(-((ra / r0) ** 6)) * -math.expm1(-((rb / r0) ** 6)) * -math.expm1(-((rc / r0) ** 6))
        energy += e
        magnitude += abs(e)
    return energy, magnitude


def evaluate(system, positions, platform, precision=None):
    properties = {"Precision": precision} if precision else {}
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform), properties)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer))


@pytest.mark.parametrize("atomic,centroid_r0", VARIANTS)
def test_reference_energy_and_forces(atomic, centroid_r0):
    system, force, positions = make_system(atomic, centroid_r0=centroid_r0)
    energy, forces = evaluate(system, positions, "Reference")
    expected, _ = numpy_energy(force, positions)
    assert expected != 0
    assert abs(energy - expected) <= 1e-10 * abs(expected), (energy, expected)

    # Finite differences of the energy along the forces (central, step 1e-5 nm).
    direction = forces / np.linalg.norm(forces)
    step = 1e-5
    e_plus, _ = evaluate(system, positions + step * direction, "Reference")
    e_minus, _ = evaluate(system, positions - step * direction, "Reference")
    numeric = -(e_plus - e_minus) / (2 * step)
    analytic = np.sum(forces * direction)
    assert abs(numeric - analytic) <= 1e-5 * abs(analytic), (numeric, analytic)


@pytest.mark.parametrize("atomic,centroid_r0", VARIANTS)
@pytest.mark.parametrize("platform,precision", PLATFORMS)
def test_platform_matches_reference(atomic, centroid_r0, platform, precision):
    if platform not in AVAILABLE:
        pytest.skip(f"the {platform} platform is not available")
    system, force, positions = make_system(atomic, centroid_r0=centroid_r0)
    energy, forces = evaluate(system, positions, platform, precision)
    reference_energy, reference_forces = evaluate(system, positions, "Reference")
    _, magnitude = numpy_energy(force, positions)
    tolerance = 1e-10 if precision == "double" else 2e-5
    assert abs(energy - reference_energy) <= tolerance * magnitude, (energy, reference_energy, magnitude)
    rms = np.sqrt((reference_forces ** 2).mean())
    # CUDA accumulates forces in 2^-32 fixed point: an absolute floor of ~4 quanta (the damped variants have small forces)
    assert np.abs(forces - reference_forces).max() <= max(10 * tolerance * rms, 1e-9)


def test_neighbor_list_growth():
    """Dense atomic sites overflow the initial CUDA neighbor lists, which must grow without changing the result."""
    if "CUDA" not in AVAILABLE:
        pytest.skip("the CUDA platform is not available")
    rng = np.random.default_rng(3)
    positions = rng.random((600, 3)) * BOX
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*np.eye(3) * BOX)
    force = ip.InteraxThreeBodyForce()
    force.setCutoffDistance(0.6)
    force.setUsesPeriodicBoundaryConditions(True)
    for i in range(len(positions)):
        system.addParticle(1.0)
        force.addSite([i], 1e-3, 1000.0)
    system.addForce(force)
    energy, forces = evaluate(system, positions, "CUDA", "double")
    reference_energy, reference_forces = evaluate(system, positions, "Reference")
    assert abs(energy - reference_energy) <= 1e-10 * abs(reference_energy)
    assert np.abs(forces - reference_forces).max() <= 1e-8 * np.sqrt((reference_forces ** 2).mean())


def test_centroid_damping_off_is_unchanged():
    """R0 = 0 is exactly the undamped force, and a very small R0 (g = 1 to rounding at these distances) too."""
    system, force, positions = make_system(False)
    energy, forces = evaluate(system, positions, "Reference")
    force.setCentroidDampingRadius(1e-4)
    energy2, forces2 = evaluate(system, positions, "Reference")
    assert energy2 == energy and np.array_equal(forces2, forces)


@pytest.mark.parametrize("centroid_r0", [0.0, 0.5])
def test_serialization(centroid_r0):
    system, force, positions = make_system(True, centroid_r0=centroid_r0)
    xml = openmm.XmlSerializer.serialize(force)
    # Without centroid damping the version 2 format is written (readable by older builds)
    assert ('version="2"' in xml) == (centroid_r0 == 0.0) and ("centroidDampingRadius" in xml) == (centroid_r0 != 0.0)
    deserialized = openmm.XmlSerializer.deserialize(xml)
    copy = ip.InteraxThreeBodyForce.cast(deserialized)  # deserialized keeps ownership of the force
    assert copy.getNumSites() == force.getNumSites()
    assert copy.getDampingMethod() == force.getDampingMethod() and copy.getDampingScale() == force.getDampingScale()
    assert copy.getCentroidDampingRadius() == force.getCentroidDampingRadius()
    for i in range(force.getNumSites()):
        assert copy.getSiteParameters(i) == force.getSiteParameters(i)
