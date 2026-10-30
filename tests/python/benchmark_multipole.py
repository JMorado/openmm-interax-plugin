"""
Benchmark InteraxMultipoleForce against the HippoNonbondedForce it replaces, on InteraxFF systems with electrostatics
and polarization only (bare forces), and with the full force field (plugin pair forces, electrostatics wrapped in a
CustomCVForce as the InteraxFF export uses it) for molecular dynamics.

  accuracy  energy and forces in mixed precision vs double precision (Reference platform): HIPPO with the export's
            merged exponents (its only safe mode), HIPPO with exact exponents, and InteraxMultipoleForce with exact
            exponents (its references: HIPPO merged and InteraxMultipoleForce exact on Reference)
  time      one evaluation of energy and forces of the electrostatics alone
  md        ns/day of molecular dynamics with the complete system (2 fs, Langevin)
  hfe       the hydration free energy box with the solute decoupled the way absolv does it: lambda * E_on + (1 - lambda)
            * E_off with two HIPPO copies (one without the solute), vs one InteraxMultipoleForce with alchemical solute

Systems (0.9 nm cutoff, PME): a box of liquid methanol and one ethanol in water (the HFE box).
Usage: python benchmark_multipole.py [--systems methanol hfe] [--platform CUDA]      (model: INTERAX_MODEL, default MQ)
"""
import argparse
import time

from compare_export import COMPONENTS, interaxplugin, interaxplugin_convert, load_model, np, openmm, unit  # noqa: E402
from plugin_path import model  # noqa: E402

from openff.interchange.components._packmol import UNIT_CUBE, pack_box  # noqa: E402
from openff.toolkit import Molecule  # noqa: E402
from openff.units import unit as off_unit  # noqa: E402

from interax.openmm import create_system_from_topology  # noqa: E402

MODEL = model("models_v3R4bfdE1MQ")

KJ = unit.kilojoule_per_mole


def build(name):
    if name == "methanol":
        molecules, counts, density = [Molecule.from_smiles("CO")], [256], 0.79
    else:
        molecules, counts, density = [Molecule.from_smiles("CCO"), Molecule.from_smiles("O")], [1, 500], 1.0
    for molecule in molecules:
        molecule.generate_conformers(n_conformers=1)
    topology = pack_box(molecules=molecules, number_of_copies=counts, target_density=density * off_unit.gram / off_unit.milliliter,
                        box_shape=UNIT_CUBE)
    return topology, topology.get_positions().m_as(off_unit.nanometer)


def wrapped(force):
    wrapper = openmm.CustomCVForce("E_hippo")
    wrapper.addCollectiveVariable("E_hippo", force)
    return wrapper


def electrostatics_system(system, force):
    s = openmm.System()
    s.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    for i in range(system.getNumParticles()):
        s.addParticle(system.getParticleMass(i))
    s.addForce(force)
    return s


def context(system, positions, platform, precision):
    properties = {"Precision": precision} if platform in ("CUDA", "OpenCL") else {}
    c = openmm.Context(system, openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds),
                       openmm.Platform.getPlatformByName(platform), properties)
    c.setPositions(positions)
    return c


def energy_forces(c):
    state = c.getState(getEnergy=True, getForces=True)
    return state.getPotentialEnergy().value_in_unit(KJ), state.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)


def time_evaluation(c, n=50):
    c.getState(getEnergy=True, getForces=True)
    start = time.perf_counter()
    for _ in range(n):
        c.getState(getEnergy=True, getForces=True)
    return 1000 * (time.perf_counter() - start) / n


def time_md(c, steps=2000):
    c.getIntegrator().step(100)
    c.getState(getEnergy=True)
    start = time.perf_counter()
    c.getIntegrator().step(steps)
    c.getState(getEnergy=True)
    return steps * 0.002 / 1000 / ((time.perf_counter() - start) / 86400)


def replace_electrostatics(system, force):
    """A copy of system with its wrapped HIPPO force replaced by force (wrapped)."""
    s = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for i in range(s.getNumForces()):
        if interaxplugin_convert.wrapped_hippo(s.getForce(i)) is not None:
            group = s.getForce(i).getForceGroup()
            s.removeForce(i)
            w = wrapped(force)
            w.setForceGroup(group)
            s.addForce(w)
            return s
    raise ValueError("no HIPPO force")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=["methanol", "hfe"])
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--skip-md", action="store_true")
    args = parser.parse_args()
    ff, nagl = load_model(MODEL)
    for name in args.systems:
        topology, positions = build(name)
        full = {}
        for exact in (False, True):
            # the HIPPO export, with merged (default) or exact exponents
            import interax.openmm as iom
            snap = iom._snap_close_values
            if exact:
                iom._snap_close_values = lambda v, rel_tol: np.asarray(v, dtype=np.float64)
            try:
                full[exact] = create_system_from_topology(topology, ff, nagl, electrostatics="interax", nonbonded_cutoff=0.9, components=COMPONENTS)
            finally:
                iom._snap_close_values = snap
        hippo = {k: interaxplugin_convert.wrapped_hippo(next(f for f in s.getForces() if interaxplugin_convert.wrapped_hippo(f) is not None))
                 for k, s in full.items()}
        # Bare forces: wrapped in a CustomCVForce, OpenMM 8.1.2 gives wrong PME results on CUDA in some boxes (this HFE box:
        # +2071 kJ/mol for HIPPO and InteraxMultipoleForce alike; the production HFE and liquid boxes are not affected)
        variants = {
            "HIPPO (merged exponents)": openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(hippo[False])),
            "HIPPO (exact exponents)": openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(hippo[True])),
            "InteraxMultipoleForce": interaxplugin_convert.convert_multipoles(hippo[True]),
        }
        references = {"HIPPO (merged exponents)": "HIPPO (merged exponents)", "HIPPO (exact exponents)": "InteraxMultipoleForce",
                      "InteraxMultipoleForce": "InteraxMultipoleForce"}
        print(f"\n### {name}: {full[False].getNumParticles()} particles, electrostatics + polarization ({args.platform})")
        ref = {k: energy_forces(context(electrostatics_system(full[False], openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(v))),
                                        positions, "Reference", "double"))
               for k, v in variants.items() if k in set(references.values())}
        print(f"| force | E (mixed) | dE vs double | max |dF| / rms F | ms / evaluation (mixed) | ms (double) |")
        print("|---|---|---|---|---|---|")
        for label, force in variants.items():
            row = {}
            for precision in ("mixed", "double"):
                c = context(electrostatics_system(full[False], openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force))),
                            positions, args.platform, precision)
                row[precision] = (energy_forces(c), time_evaluation(c))
            (e, f), _ = row["mixed"]
            e_ref, f_ref = ref[references[label]]
            print(f"| {label} | {e:.4f} | {e - e_ref:+.2e} | {np.abs(f - f_ref).max() / np.sqrt((f_ref ** 2).mean()):.1e} | "
                  f"{row['mixed'][1]:.2f} | {row['double'][1]:.2f} |")

        # Molecular dynamics with the complete force field (plugin pair forces + electrostatics)
        print(f"\nMD, complete force field, mixed precision ({args.platform}):")
        base = interaxplugin_convert.convert_system(full[False], overlap="exact")
        base_exact = interaxplugin_convert.convert_system(full[True], overlap="exact")
        md = {"HIPPO (merged exponents)": base,
              "InteraxMultipoleForce": replace_electrostatics(base_exact, interaxplugin_convert.convert_multipoles(hippo[True]))}
        if args.skip_md:
            md = {}
        for label, s in md.items():
            c = context(s, positions, args.platform, "mixed")
            openmm.LocalEnergyMinimizer.minimize(c, 10.0, 200)
            c.setVelocitiesToTemperature(300 * unit.kelvin)
            print(f"  {label:28s} {time_md(c):8.2f} ns/day")

        if name == "hfe":
            # absolv's decoupling: lambda E_on + (1 - lambda) E_off (two HIPPO copies) vs one alchemical InteraxMultipoleForce
            solute = range(9)  # ethanol is the first molecule
            on = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(hippo[False]))
            off = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(hippo[False]))
            for i in solute:
                p = list(off.getParticleParameters(i))
                p[0], p[1], p[2], p[3], p[11] = 0.0, [0.0] * 3, [0.0] * 9, 0.0, 1e-20
                off.setParticleParameters(i, *p)
            # absolv gives each end state its own CustomCVForce (two HIPPO forces cannot share a GPU context)
            two = electrostatics_system(full[False], openmm.CustomCVForce("lambda_electrostatics*E_on"))
            two.getForce(0).addGlobalParameter("lambda_electrostatics", 0.5)
            two.getForce(0).addCollectiveVariable("E_on", on)
            off_force = openmm.CustomCVForce("(1-lambda_electrostatics)*E_off")
            off_force.addGlobalParameter("lambda_electrostatics", 0.5)
            off_force.addCollectiveVariable("E_off", off)
            two.addForce(off_force)
            native = interaxplugin_convert.convert_multipoles(hippo[True], alchemical_particles=solute)
            one = electrostatics_system(full[False], wrapped(native))
            print(f"\nHFE box at lambda_electrostatics = 0.5, electrostatics only, mixed precision ({args.platform}):")
            for label, s in (("absolv: two HIPPO copies", two), ("one alchemical InteraxMultipoleForce", one)):
                c = context(s, positions, args.platform, "mixed")
                c.setParameter("lambda_electrostatics", 0.5)
                print(f"  {label:38s} {time_evaluation(c):.2f} ms / evaluation")


if __name__ == "__main__":
    main()
