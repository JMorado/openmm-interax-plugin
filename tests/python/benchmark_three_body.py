"""
Benchmark the atomic three-body dispersion against the molecular one of InteraxFF.

For each system (benchmark.py), the plugin conversion of the export is run with its molecular ATM, and with the
molecular ATM replaced by an atomic one (interaxplugin_convert.atomic_three_body: one site per atom, Tang-Toennies
damping, intramolecular triples excluded) at several cutoffs. Reported per variant: the ATM energy, the time of a
molecular dynamics step with the three-body force alone and with the complete system, and the number of triples
inside the cutoff.

Usage: python benchmark_three_body.py [--platform CUDA] [--precision mixed] [--systems methanol hfe] [--cutoffs 0.6 0.8 1.2]
"""
import argparse
import time

import numpy as np
from benchmark import build, mark_terms
from compare_export import COMPONENTS, interaxplugin, interaxplugin_convert, load_model, openmm, unit

from interax.openmm import create_system_from_topology


def count_triples(force, positions, box):
    """The number of triples of sites of different molecules whose three distances are within the cutoff."""
    sites, molecules = [], []
    for i in range(force.getNumSites()):
        particles, _, _, _, molecule, _ = force.getSiteParameters(i)
        sites.append(positions[list(particles)].mean(axis=0))
        molecules.append(molecule if molecule >= 0 else -1 - i)
    sites, molecules = np.array(sites), np.array(molecules)
    d = sites[None] - sites[:, None]
    d -= box * np.round(d / box)
    neighbor = (np.linalg.norm(d, axis=-1) < force.getCutoffDistance()) & (molecules[None] != molecules[:, None])
    upper = np.triu(neighbor, 1).astype(np.float64)
    # Triples A < B < C with all three pairs neighbors: (U U)[A, C] counts the B between them, masked by U[A, C].
    return int(np.sum((upper @ upper) * upper))


def ms_per_step(system, positions, platform, properties, steps=200):
    integrator = openmm.LangevinMiddleIntegrator(298.15, 1.0, 0.001)
    context = openmm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    energy = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    integrator.step(20)
    context.getState(getEnergy=True)
    start = time.perf_counter()
    integrator.step(steps)
    context.getState(getEnergy=True)
    return energy, 1e3 * (time.perf_counter() - start) / steps


def with_three_body(system, three_body):
    """A copy of a plugin System with its three-body force replaced (in group 0, every other force in group 1)."""
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for i in reversed(range(copy.getNumForces())):
        if interaxplugin.InteraxThreeBodyForce.isinstance(copy.getForce(i)):
            copy.removeForce(i)
    for force in copy.getForces():
        force.setForceGroup(1)
    three_body = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(three_body))
    three_body.setForceGroup(0)
    copy.addForce(three_body)
    return copy


def alone(system, three_body):
    copy = openmm.System()
    for i in range(system.getNumParticles()):
        copy.addParticle(system.getParticleMass(i))
    copy.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    copy.addForce(openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(three_body)))
    copy.getForce(0).setForceGroup(0)
    return copy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--systems", nargs="+", default=["methanol", "hfe"])
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[0.6, 0.8, 1.2])
    args = parser.parse_args()
    platform = openmm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL") else {}
    ff, nagl = load_model()
    for name in args.systems:
        topology, positions = build(name)
        export = create_system_from_topology(topology, ff, nagl, electrostatics="hippo", nonbonded_cutoff=0.9, components=COMPONENTS)
        dispersion = next(f for f in export.getForces()
                          if isinstance(f, openmm.CustomNonbondedForce) and interaxplugin_convert._classify(f) == "dispersion")
        plugin = interaxplugin_convert.convert_system(export, copy=False, overlap="exact")
        mark_terms(plugin, True)
        box = np.diag(np.array([v.value_in_unit(unit.nanometer) for v in plugin.getDefaultPeriodicBoxVectors()]))
        molecular = next(f for f in plugin.getForces() if interaxplugin.InteraxThreeBodyForce.isinstance(f))
        molecular = interaxplugin.InteraxThreeBodyForce.cast(molecular)
        variants = [(f"molecular {molecular.getCutoffDistance():.1f} nm", molecular)]
        for cutoff in args.cutoffs:
            variants.append((f"atomic    {cutoff:.1f} nm", interaxplugin_convert.atomic_three_body(dispersion, cutoff=cutoff)))
        print(f"\n{name}: {plugin.getNumParticles()} particles, {args.platform} ({args.precision})", flush=True)
        print(f"  {'three-body':<20} {'sites':>6} {'triples':>12} {'E (kJ/mol)':>12} {'alone ms/step':>14} {'full ms/step':>13} {'ns/day (1 fs)':>14}")
        _, baseline = ms_per_step(alone(plugin, interaxplugin.InteraxThreeBodyForce()), positions, platform, properties)
        for label, force in variants:
            triples = count_triples(force, positions, box)
            energy, alone_ms = ms_per_step(alone(plugin, force), positions, platform, properties)
            _, full_ms = ms_per_step(with_three_body(plugin, force), positions, platform, properties)
            print(f"  {label:<20} {force.getNumSites():>6} {triples:>12,} {energy:>12.3f} {alone_ms - baseline:>14.3f} {full_ms:>13.3f} "
                  f"{86.4 / full_ms:>14.1f}", flush=True)
        print(f"  (alone: the step time of the three-body force by itself minus that of an empty system, {baseline:.3f} ms)")


if __name__ == "__main__":
    main()
