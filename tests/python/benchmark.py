"""
Benchmark the plugin forces against the InteraxFF export they replace.

For each system and platform, two measurements:

  terms   time of one evaluation (energy and forces) of the replaced terms only: dispersion, exchange repulsion,
          short-range correction and molecular three-body dispersion, export (Custom forces) vs plugin
  md      speed of molecular dynamics with the complete system (HIPPO electrostatics and polarization, valence terms,
          ...), export vs plugin; only on the CUDA and OpenCL platforms, where it takes seconds

Systems (model E1, 0.9 nm cutoff, as studies/hfe/run_hfe.py and the liquid simulations):

  methanol   256 molecules of liquid methanol
  hfe        one ethanol in 500 waters, the hydration free energy box

Usage: python benchmark.py [--platforms Reference CPU CUDA] [--systems methanol hfe] [--precision mixed]
Run with the interax environment; the plugin is loaded from PLUGIN_PREFIX.
"""
import argparse
import time

from compare_export import COMPONENTS, interaxplugin, interaxplugin_convert, load_model, openmm

from openff.interchange.components._packmol import UNIT_CUBE, pack_box
from openff.toolkit import Molecule
from openff.units import unit as off_unit

from interax.openmm import create_system_from_topology

PLUGIN = ("InteraxDispersionForce", "InteraxExchangeForce", "InteraxShortRangeForce", "InteraxThreeBodyForce")


def build(name):
    """The topology and positions (nm) of a benchmark system."""
    if name == "methanol":
        molecules, counts, density = [Molecule.from_smiles("CO")], [256], 0.79
    else:
        molecules, counts, density = [Molecule.from_smiles("CCO"), Molecule.from_smiles("O")], [1, 500], 1.0
    for molecule in molecules:
        molecule.generate_conformers(n_conformers=1)
    topology = pack_box(molecules=molecules, number_of_copies=counts, target_density=density * off_unit.gram / off_unit.milliliter,
                        box_shape=UNIT_CUBE)
    return topology, topology.get_positions().m_as(off_unit.nanometer)


def _plugin_classes():
    return [getattr(interaxplugin, name) for name in PLUGIN]


def mark_terms(system, plugin):
    """Group 0: the InteraxFF terms that the plugin replaces (export or plugin version); group 1: everything else."""
    count = 0
    for force in system.getForces():
        if plugin:
            is_term = any(cls.isinstance(force) for cls in _plugin_classes())
        else:
            is_term = (isinstance(force, openmm.CustomNonbondedForce) and interaxplugin_convert._classify(force) is not None) or (
                isinstance(force, openmm.CustomCentroidBondForce) and "C9" in force.getEnergyFunction())
        force.setForceGroup(0 if is_term else 1)
        count += is_term
    return count


def make_context(system, positions, platform, precision, integrator=None):
    properties = {"Precision": precision} if platform in ("CUDA", "OpenCL") else {}
    integrator = integrator or openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName(platform), properties)
    context.setPositions(positions)
    return context


def time_terms(context, min_time=3.0, max_evaluations=200):
    """Seconds per evaluation of the energy and forces of group 0."""
    context.getState(getEnergy=True, getForces=True, groups={0})  # compile kernels, build neighbor lists
    count, start = 0, time.perf_counter()
    while count < max_evaluations and (count == 0 or time.perf_counter() - start < min_time):
        context.getState(getEnergy=True, getForces=True, groups={0})
        count += 1
    return (time.perf_counter() - start) / count


def time_md(context, timestep_fs=1.0, steps=500):
    """ns/day of dynamics with the complete system."""
    integrator = context.getIntegrator()
    integrator.step(20)
    context.getState(getEnergy=True)
    start = time.perf_counter()
    integrator.step(steps)
    context.getState(getEnergy=True)  # wait for the GPU to finish
    elapsed = time.perf_counter() - start
    return steps * timestep_fs * 1e-6 / (elapsed / 86400)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platforms", nargs="+", default=["Reference", "CPU", "CUDA"])
    parser.add_argument("--systems", nargs="+", default=["methanol", "hfe"])
    parser.add_argument("--precision", default="mixed", help="CUDA/OpenCL precision")
    parser.add_argument("--md-steps", type=int, default=500)
    parser.add_argument("--skip-reference-export", action="store_true", help="the export on Reference takes minutes per evaluation")
    args = parser.parse_args()
    ff, nagl = load_model()
    rows = []
    for name in args.systems:
        topology, positions = build(name)
        start = time.perf_counter()
        export = create_system_from_topology(topology, ff, nagl, electrostatics="hippo", nonbonded_cutoff=0.9, components=COMPONENTS)
        build_export = time.perf_counter() - start
        triples = sum(f.getNumBonds() for f in export.getForces() if isinstance(f, openmm.CustomCentroidBondForce))
        mark_terms(export, False)
        # The plugin system is converted from a second copy of the export, built the same way (copying the export
        # through XML would serialize all its triples).
        plugin = create_system_from_topology(topology, ff, nagl, electrostatics="hippo", nonbonded_cutoff=0.9, components=COMPONENTS)
        start = time.perf_counter()
        plugin = interaxplugin_convert.convert_system(plugin, copy=False)
        convert = time.perf_counter() - start
        mark_terms(plugin, True)
        print(f"\n{name}: {export.getNumParticles()} particles, {triples} stored triples; "
              f"export built in {build_export:.0f} s, converted in {convert:.1f} s", flush=True)
        for platform in args.platforms:
            for variant, system in (("export", export), ("plugin", plugin)):
                if platform == "Reference" and variant == "export" and args.skip_reference_export:
                    continue
                start = time.perf_counter()
                integrator = openmm.LangevinMiddleIntegrator(298.15, 1.0, 0.001)
                context = make_context(system, positions, platform, args.precision, integrator)
                setup = time.perf_counter() - start
                terms = time_terms(context, max_evaluations=200 if platform != "Reference" else 3)
                md = time_md(context, steps=args.md_steps) if platform in ("CUDA", "OpenCL") else float("nan")
                rows.append((name, platform, variant, setup, terms, md))
                print(f"  {platform:<9} {variant:<6} context {setup:7.1f} s   terms {1e3 * terms:10.2f} ms   md {md:8.2f} ns/day", flush=True)
                del context, integrator
    print(f"\n{'system':<9} {'platform':<9} {'terms export':>13} {'terms plugin':>13} {'speedup':>8} {'md export':>10} {'md plugin':>10} {'speedup':>8}")
    for name in args.systems:
        for platform in args.platforms:
            r = {v: (t, m) for n, p, v, _, t, m in rows if n == name and p == platform}
            if "plugin" not in r:
                continue
            te, me = r.get("export", (float("nan"), float("nan")))
            tp, mp = r["plugin"]
            print(f"{name:<9} {platform:<9} {1e3 * te:10.2f} ms {1e3 * tp:10.2f} ms {te / tp:8.1f} {me:10.2f} {mp:10.2f} {mp / me:8.2f}")
    return rows


if __name__ == "__main__":
    main()
