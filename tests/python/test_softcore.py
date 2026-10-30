"""
Test the soft-core distance of the alchemical pairs of the plugin pair forces (InteraxPairForce.setSoftcoreRadius):
a pair with exactly one alchemical particle at distance r is evaluated at r_eff = (r^6 + (1 - lambda) R^6)^(1/6) and
scaled by lambda. Checked: no change at lambda = 1; one pair at r equals lambda times the same pair without soft core
at r_eff; forces against finite differences; every platform against Reference.

Usage: python -m pytest test_softcore.py      (INTERAX_PLUGIN_PREFIX: where the plugin is installed)
"""
import os
import sys

import numpy as np
import pytest


import openmm  # noqa: E402

import plugin_path  # noqa: E402  (INTERAX_PLUGIN_PREFIX)
import interaxplugin as ip  # noqa: E402

RADIUS = 0.3
PLATFORMS = [("CPU", None), ("CUDA", "mixed"), ("CUDA", "double")]
AVAILABLE = plugin_path.available_platforms()


def make_forces(rng, n, softcore):
    """The three pair forces with random parameters for n particles."""
    dispersion = ip.InteraxDispersionForce()
    exchange = ip.InteraxExchangeForce()
    short_range = ip.InteraxShortRangeForce()
    for _ in range(n):
        dispersion.addParticle(rng.uniform(1e-3, 3e-3), rng.uniform(1e-5, 3e-5), rng.uniform(1e-7, 3e-7), rng.uniform(5e-4, 2e-3),
                               rng.uniform(0.02, 0.04))
        exchange.addParticle(rng.uniform(5, 15), rng.uniform(0.02, 0.04))
        short_range.addParticle(rng.uniform(-30, -5), rng.uniform(-10, 10), rng.uniform(0.02, 0.04))
    for force in (dispersion, exchange, short_range):
        force.setSoftcoreRadius(softcore)
    return dispersion, exchange, short_range


def system_with(forces, n, solute):
    system = openmm.System()
    for _ in range(n):
        system.addParticle(12.0)
    for group, force in enumerate(forces):
        for i in range(n):
            values = list(force.getParticleParameters(i))[:-1]
            force.setParticleParameters(i, *values, i in solute)
        force.setForceGroup(group)
        system.addForce(force)
    return system


def evaluate(system, positions, lam, platform="Reference", precision=None, group=None):
    properties = {"Precision": precision} if precision else {}
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform), properties)
    context.setPositions(positions)
    if "lambda_sterics" in context.getParameters():
        context.setParameter("lambda_sterics", lam)
    state = context.getState(getEnergy=True, getForces=True, **({"groups": {group}} if group is not None else {}))
    return (state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer))


def clusters(seed=2):
    """Two clusters of four particles; the first one is the alchemical solute."""
    rng = np.random.default_rng(seed)
    positions = np.concatenate([0.12 * rng.normal(size=(4, 3)), 0.12 * rng.normal(size=(4, 3)) + [0.25, 0, 0]])
    return rng, positions, set(range(4))


def test_no_change_at_lambda_one():
    rng, positions, solute = clusters()
    state = rng.bit_generator.state
    plain = system_with(make_forces(rng, 8, 0.0), 8, solute)
    rng.bit_generator.state = state
    soft = system_with(make_forces(rng, 8, RADIUS), 8, solute)
    e0, f0 = evaluate(plain, positions, 1.0)
    e1, f1 = evaluate(soft, positions, 1.0)
    assert abs(e1 - e0) <= 1e-12 * abs(e0)
    assert np.abs(f1 - f0).max() <= 1e-10 * np.abs(f0).max()


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.9])
def test_single_pair_at_soft_core_distance(lam):
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    soft = system_with(make_forces(rng, 2, RADIUS), 2, {0})
    rng.bit_generator.state = state
    plain = system_with(make_forces(rng, 2, 0.0), 2, set())
    for r in (0.02, 0.1, 0.2, 0.35):
        r_eff = (r ** 6 + (1 - lam) * RADIUS ** 6) ** (1 / 6)
        for group in range(3):
            e_soft, _ = evaluate(soft, [[0, 0, 0], [r, 0, 0]], lam, group=group)
            e_plain, _ = evaluate(plain, [[0, 0, 0], [r_eff, 0, 0]], 1.0, group=group)
            assert abs(e_soft - lam * e_plain) <= 1e-10 * max(abs(e_plain), 1e-6), (group, r, e_soft, lam * e_plain)


def test_forces_against_finite_differences():
    rng, positions, solute = clusters()
    system = system_with(make_forces(rng, 8, RADIUS), 8, solute)
    energy, forces = evaluate(system, positions, 0.4)
    direction = forces / np.linalg.norm(forces)
    step = 1e-5
    e_plus, _ = evaluate(system, positions + step * direction, 0.4)
    e_minus, _ = evaluate(system, positions - step * direction, 0.4)
    numeric = -(e_plus - e_minus) / (2 * step)
    analytic = np.sum(forces * direction)
    assert abs(numeric - analytic) <= 1e-6 * abs(analytic), (numeric, analytic)


@pytest.mark.parametrize("platform,precision", PLATFORMS)
def test_platform_matches_reference(platform, precision):
    if platform not in AVAILABLE:
        pytest.skip(f"the {platform} platform is not available")
    rng, positions, solute = clusters()
    system = system_with(make_forces(rng, 8, RADIUS), 8, solute)
    for lam in (0.0, 0.4, 1.0):
        energy, forces = evaluate(system, positions, lam, platform, precision)
        reference_energy, reference_forces = evaluate(system, positions, lam)
        tolerance = 1e-9 if precision == "double" else 1e-4
        assert abs(energy - reference_energy) <= tolerance * abs(reference_energy), (lam, energy, reference_energy)
        assert np.abs(forces - reference_forces).max() <= tolerance * 10 * np.abs(reference_forces).max()
