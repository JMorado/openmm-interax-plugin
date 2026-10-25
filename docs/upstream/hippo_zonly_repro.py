"""
HippoNonbondedForce, Reference/CPU platforms: forces on ZOnly particles are wrong (reads particleData[-1]).

Two rigid "molecules" A-B with a ZOnly frame on A (z axis particle B, no x particle) carrying a dipole and a
quadrupole, plus a polarizable partner. Analytic forces are compared with central finite differences of the energy.
Before the comparison, garbage is planted in front of the particle data by allocating other Contexts, which changes
what particleData[-1] reads.
Usage: python hippo_zonly_repro.py [Reference|CPU]
"""
import sys

import numpy as np
import openmm
from openmm import unit

platform = sys.argv[1] if len(sys.argv) > 1 else "Reference"
ZONLY = openmm.HippoNonbondedForce.ZOnly
NOAXIS = openmm.HippoNonbondedForce.NoAxisType


def make_system():
    system = openmm.System()
    force = openmm.HippoNonbondedForce()
    force.setNonbondedMethod(openmm.HippoNonbondedForce.NoCutoff)
    force.setExtrapolationCoefficients([0.0, 0.0, 1.0])
    quad = [0.002, 0, 0, 0, 0.002, 0, 0, 0, -0.004]          # cylindrically symmetric about z (e nm^2)
    for mol in range(2):
        system.addParticle(12.0)
        system.addParticle(16.0)
        a, b = 2 * mol, 2 * mol + 1
        # particle a: ZOnly frame, z axis towards b, no x particle (-1)
        force.addParticle(-0.3, [0, 0, 0.01], quad, 4.0, 40.0, 0, 1, 0, 0, 0, 1, 0.001, ZONLY, b, -1, -1)
        force.addParticle(0.3, [0, 0, 0], [0] * 9, 6.0, 45.0, 0, 1, 0, 0, 0, 1, 0.001, NOAXIS, -1, -1, -1)
        force.addException(a, b, 0, 0, 0, 0, 0, 0)
    system.addForce(force)
    return system


positions = np.array([[0.0, 0.0, 0.0], [0.13, 0.02, 0.01], [0.31, 0.12, -0.05], [0.42, 0.21, -0.01]])
system = make_system()
keep = [openmm.Context(make_system(), openmm.VerletIntegrator(1e-3), openmm.Platform.getPlatformByName(platform)) for _ in range(20)]
context = openmm.Context(system, openmm.VerletIntegrator(1e-3), openmm.Platform.getPlatformByName(platform))
context.setPositions(positions)
forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
h = 1e-5
fd = np.zeros_like(forces)
for i in range(len(positions)):
    for k in range(3):
        e = []
        for s in (1, -1):
            x = positions.copy()
            x[i, k] += s * h
            context.setPositions(x)
            e.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        fd[i, k] = -(e[0] - e[1]) / (2 * h)
print(f"OpenMM {openmm.__version__}, {platform} platform")
for i in range(len(positions)):
    print(f"particle {i}: analytic {np.round(forces[i], 5)}  finite difference {np.round(fd[i], 5)}")
print(f"max |analytic - finite difference| = {np.abs(forces - fd).max():.3e} kJ/mol/nm")
