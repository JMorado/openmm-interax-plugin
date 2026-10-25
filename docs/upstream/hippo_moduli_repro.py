"""
HippoNonbondedForce, CUDA/OpenCL: the PME convolution uses the x B-spline moduli for the z axis.

A box of point charges with dipoles, PME, the same system on Reference and on CUDA. With a cubic box (equal grid
sizes) the two agree; with a box longer in z (different grid sizes) the CUDA energy is wrong.
Usage: python hippo_moduli_repro.py
"""
import numpy as np
import openmm
from openmm import unit


def energy(box, platform, seed=1):
    rng = np.random.default_rng(seed)
    n = 300
    pos = rng.uniform(0, 1, size=(n, 3)) * np.array(box)
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(box[0], 0, 0), openmm.Vec3(0, box[1], 0), openmm.Vec3(0, 0, box[2]))
    force = openmm.HippoNonbondedForce()
    force.setNonbondedMethod(openmm.HippoNonbondedForce.PME)
    force.setCutoffDistance(0.7)
    force.setSwitchingDistance(0.6)
    force.setPMEParameters(3.0, 24, 24, int(round(24 * box[2] / box[0])))
    force.setDPMEParameters(3.0, 24, 24, int(round(24 * box[2] / box[0])))
    force.setExtrapolationCoefficients([1.0])
    for i in range(n):
        system.addParticle(1.0)
        q = 0.4 if i % 2 == 0 else -0.4
        force.addParticle(q, [0, 0, 0], [0] * 9, 0.0, 40.0, 0, 1, 0, 0, 0, 1, 1e-3, openmm.HippoNonbondedForce.NoAxisType, -1, -1, -1)
    system.addForce(force)
    context = openmm.Context(system, openmm.VerletIntegrator(1e-3), openmm.Platform.getPlatformByName(platform),
                             {"Precision": "double"} if platform == "CUDA" else {})
    context.setPositions(pos)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


print(f"OpenMM {openmm.__version__}")
for box in ([3.0, 3.0, 3.0], [3.0, 3.0, 6.0]):
    e_ref, e_cuda = energy(box, "Reference"), energy(box, "CUDA")
    print(f"box {box}: Reference {e_ref:.6f}  CUDA {e_cuda:.6f}  difference {e_cuda - e_ref:+.3e} kJ/mol")
