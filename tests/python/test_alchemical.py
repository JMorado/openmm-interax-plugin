"""
Check the alchemical scaling of the plugin forces against a decoupled reference.

The first molecule of each system is flagged alchemical. Its interactions with the rest are then scaled by
lambda_sterics, so for every term and every lambda

    E(lambda) = (1 - lambda) E_off + lambda E_on

where E_on is the energy with the molecule fully coupled (no alchemical flags) and E_off the energy with the molecule
removed from the term (zero polarizability or width): every pair and triple involving it vanishes, as at lambda = 0.
The same holds for the forces. The dispersion long range correction is the one exception, by a constant: it counts
every pair of particles, those within the alchemical molecule included (as the export's does), and these pairs, with
two alchemical particles, are not scaled. So in a periodic box E_dispersion(lambda) = (1 - lambda) (E_off + c) +
lambda E_on, where c, the correction of the pairs within the molecule, is the same at every lambda and cancels in
free energy differences. The test reports c and checks the linearity in lambda.

Usage: python test_alchemical.py [--platform Reference] [--precision double]
"""
import argparse

import numpy as np

from compare_export import (COMPONENTS, box_topology, cluster_topology, interaxplugin, interaxplugin_convert, load_model,
                            openmm, terms_by_kind, unit)
from interax.openmm import create_system_from_topology

LAMBDAS = (0.0, 0.3, 0.75, 1.0)


def first_molecule(topology):
    return list(range(topology.molecule(0).n_atoms))


def modify(force, solute, mode):
    """Flag the solute alchemical (mode 'alchemical') or remove it from the term (mode 'off')."""
    alchemical = mode == "alchemical"
    if interaxplugin.InteraxThreeBodyForce.isinstance(force):
        force = interaxplugin.InteraxThreeBodyForce.cast(force)
        for m in range(force.getNumSites()):
            particles, alpha, frequency, width, molecule, _ = force.getSiteParameters(m)
            if set(particles) <= set(solute):
                force.setSiteParameters(m, particles, 0.0 if mode == "off" else alpha, frequency, width, molecule, alchemical)
        return
    for cls, gate in ((interaxplugin.InteraxDispersionForce, 3), (interaxplugin.InteraxExchangeForce, 1),
                      (interaxplugin.InteraxShortRangeForce, 2)):
        if cls.isinstance(force):
            force = cls.cast(force)
            for i in solute:
                values = list(force.getParticleParameters(i))[:-1]
                if mode == "off":
                    values[gate] = 0.0
                force.setParticleParameters(i, *values, alchemical)


def energies(system, positions, box, platform, precision, indices, lambdas):
    """Energy and forces of each term (one force group each) at each lambda."""
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(31)
    for group, i in enumerate(indices.values()):
        system.getForce(i).setForceGroup(group)
    properties = {"Precision": precision} if platform in ("CUDA", "OpenCL") else {}
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform), properties)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    results = {}
    for lam in lambdas:
        if "lambda_sterics" in context.getParameters():
            context.setParameter("lambda_sterics", lam)
        for group, kind in enumerate(indices):
            state = context.getState(getEnergy=True, getForces=True, groups={group})
            results[kind, lam] = (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                                  state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    return results


def check(name, topology, positions, periodic, platform, precision, ff, nagl):
    export = create_system_from_topology(topology, ff, nagl, electrostatics="hippo", nonbonded_cutoff=0.9, components=COMPONENTS)
    box = export.getDefaultPeriodicBoxVectors() if periodic else None
    solute = first_molecule(topology)
    systems = {}
    for mode in ("on", "off", "alchemical"):
        system = interaxplugin_convert.convert_system(export)
        indices = terms_by_kind(system, True)
        if mode != "on":
            for i in indices.values():
                modify(system.getForce(i), solute, mode)
        systems[mode] = (system, indices)
    on = energies(systems["on"][0], positions, box, platform, precision, systems["on"][1], [1.0])
    off = energies(systems["off"][0], positions, box, platform, precision, systems["off"][1], [1.0])
    alchemical = energies(systems["alchemical"][0], positions, box, platform, precision, systems["alchemical"][1], LAMBDAS)
    print(f"\n{name}: alchemical molecule of {len(solute)} particles, {platform}")
    print(f"  {'term':<12} {'lambda':>6} {'E expected':>14} {'E plugin':>14} {'abs dE':>9} {'max |dF|':>10}")
    worst = 0.0
    for kind in systems["on"][1]:
        e_on, f_on = on[kind, 1.0]
        e_off, f_off = off[kind, 1.0]
        constant = alchemical[kind, 0.0][0] - e_off  # the intramolecular long range correction (zero but in that case)
        if kind != "dispersion" or not periodic:
            assert abs(constant) <= 1e-9 * max(abs(e_on), 1.0), (kind, constant)
        else:
            print(f"  {kind}: long range correction of the pairs within the alchemical molecule c = {constant:.6e} kJ/mol")
        for lam in LAMBDAS:
            e, f = alchemical[kind, lam]
            e_expected = (1 - lam) * (e_off + constant) + lam * e_on
            f_expected = (1 - lam) * f_off + lam * f_on
            de, df = abs(e - e_expected), np.abs(f - f_expected).max()
            scale = max(abs(e_on - e_off), 1e-12)
            worst = max(worst, de / scale)
            print(f"  {kind:<12} {lam:6.2f} {e_expected:14.6f} {e:14.6f} {de:9.1e} {df:10.2e}")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--precision", default="double")
    args = parser.parse_args()
    ff, nagl = load_model()
    worst = check("cluster", *cluster_topology(), False, args.platform, args.precision, ff, nagl)
    worst = max(worst, check("methanol box", *box_topology(), True, args.platform, args.precision, ff, nagl))
    print(f"\nlargest difference relative to the decoupling energy: {worst:.1e}")


if __name__ == "__main__":
    main()
