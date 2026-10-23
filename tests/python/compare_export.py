"""
Compare the plugin forces with the InteraxFF export they replace, term by term, on real InteraxFF systems.

Builds the systems with interax.openmm.create_system_from_topology (model E1, examples/models_v3R4bfdE1), converts
them with interaxplugin_convert.convert_system, and compares the energy and the forces of each term:

  cluster   a non-periodic cluster of H/C/N/O/S/Cl molecules (no cutoff)
  box       a periodic box of liquid methanol (0.9 nm cutoff, switching from 0.8 nm, dispersion long range correction)

Usage: python compare_export.py [--platform Reference] [--systems cluster box]
Run with the interax environment; the plugin is loaded from PLUGIN_PREFIX.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
PLUGIN_PREFIX = os.environ.get("INTERAX_PLUGIN_PREFIX", "/shared/joaomorado/opt/openmm-interax")
INTERAX_REPO = "/shared/joaomorado/repos/InteraxFF"
MODEL = f"{INTERAX_REPO}/examples/models_v3R4bfdE1"
sys.path.insert(0, f"{PLUGIN_PREFIX}/python")
sys.path.insert(0, INTERAX_REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402,F401  (load the env's libstdc++ before anything else)
import openmm  # noqa: E402
import openmm.unit as unit  # noqa: E402

openmm.Platform.loadPluginsFromDirectory(f"{PLUGIN_PREFIX}/lib/plugins")
import interaxplugin  # noqa: E402
import interaxplugin_convert  # noqa: E402
from openff.interchange.components._packmol import UNIT_CUBE, pack_box  # noqa: E402
from openff.toolkit import Molecule, Topology  # noqa: E402
from openff.units import unit as off_unit  # noqa: E402

from interax.force_field import ForceField  # noqa: E402
from interax.models import NAGLModel, PerAtomicNumberModel  # noqa: E402
from interax.openmm import create_system_from_topology  # noqa: E402

COMPONENTS = ["E_static", "E_ind", "E_disp", "E_exrep", "E_src", "E_exind", "E_atm"]
KINDS = {
    "dispersion": (lambda f: isinstance(f, openmm.CustomNonbondedForce) and "m1" in _names(f), interaxplugin.InteraxDispersionForce),
    "exchange": (lambda f: isinstance(f, openmm.CustomNonbondedForce) and "A" in _names(f) and _nonzero_a(f), interaxplugin.InteraxExchangeForce),
    "short_range": (lambda f: isinstance(f, openmm.CustomNonbondedForce) and "K" in _names(f) and "L" in _names(f), interaxplugin.InteraxShortRangeForce),
    "three_body": (lambda f: isinstance(f, openmm.CustomCentroidBondForce), interaxplugin.InteraxThreeBodyForce),
}


def _names(force):
    return {force.getPerParticleParameterName(k) for k in range(force.getNumPerParticleParameters())}


def _nonzero_a(force):
    k = [force.getPerParticleParameterName(j) for j in range(force.getNumPerParticleParameters())].index("A")
    return any(force.getParticleParameters(i)[k] != 0 for i in range(force.getNumParticles()))


def load_model():
    nagl = NAGLModel.load(f"{MODEL}/nagl_model.pt")
    ff = ForceField(nagl, PerAtomicNumberModel.load(f"{MODEL}/per_atomic_number_model.pt", device="cpu"))
    ff.load_state_dict(torch.load(f"{MODEL}/force_field_state.pt", map_location="cpu"), strict=False)
    ff.eval()
    return ff, nagl


def cluster_topology():
    """Six different molecules on a 2 x 3 grid 0.45 nm apart, randomly rotated (non-periodic)."""
    rng = np.random.default_rng(7)
    molecules, positions = [], []
    for k, smiles in enumerate(["CO", "O", "CC(C)=O", "c1ccncc1", "CSC", "ClC(Cl)Cl"]):
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        xyz = mol.conformers[0].m_as(off_unit.nanometer)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        centre = np.array([0.45 * (k % 3), 0.45 * (k // 3), 0.1 * rng.normal()])
        positions.append((xyz - xyz.mean(axis=0)) @ q.T + centre)
        molecules.append(mol)
    return Topology.from_molecules(molecules), np.vstack(positions)


def box_topology():
    mol = Molecule.from_smiles("CO")
    mol.generate_conformers(n_conformers=1)
    topology = pack_box(molecules=[mol], number_of_copies=[140], target_density=0.79 * off_unit.gram / off_unit.milliliter,
                        box_shape=UNIT_CUBE)
    return topology, topology.get_positions().m_as(off_unit.nanometer)


def terms_by_kind(system, plugin):
    """Map each InteraxFF term to the index of its force (export or plugin version)."""
    found = {}
    for i, force in enumerate(system.getForces()):
        for kind, (is_export, plugin_class) in KINDS.items():
            if (plugin and plugin_class.isinstance(force)) or (not plugin and is_export(force)):
                found[kind] = i
    return found


def evaluate(system, positions, box, platform, indices):
    """Energy (kJ/mol) and forces (kJ/mol/nm) of each force in indices, each evaluated in its own group."""
    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(31)
    for group, i in enumerate(indices.values()):
        system.getForce(i).setForceGroup(group)
    properties = {"Precision": "double"} if platform in ("CUDA", "OpenCL") else {}
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform), properties)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    results = {}
    for group, kind in enumerate(indices):
        state = context.getState(getEnergy=True, getForces=True, groups={group})
        results[kind] = (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                         state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    return results


def compare(name, topology, positions, periodic, platform, ff, nagl):
    system = create_system_from_topology(topology, ff, nagl, electrostatics="hippo", nonbonded_cutoff=0.9, components=COMPONENTS)
    box = system.getDefaultPeriodicBoxVectors() if periodic else None
    converted = interaxplugin_convert.convert_system(system)
    export_terms, plugin_terms = terms_by_kind(system, False), terms_by_kind(converted, True)
    assert set(export_terms) == set(plugin_terms), (export_terms, plugin_terms)
    reference = evaluate(system, positions, box, "Reference", export_terms)
    test = evaluate(converted, positions, box, platform, plugin_terms)
    print(f"\n{name}: {system.getNumParticles()} particles, plugin on {platform} vs export on Reference")
    print(f"  {'term':<12} {'E export':>14} {'E plugin':>14} {'rel dE':>9} {'max |dF|':>10} {'rms F':>10}")
    worst = 0.0
    for kind in export_terms:
        e_ref, f_ref = reference[kind]
        e_test, f_test = test[kind]
        rel = abs(e_test - e_ref) / max(abs(e_ref), 1e-12)
        df = np.abs(f_test - f_ref).max()
        rms = np.sqrt((f_ref ** 2).mean())
        worst = max(worst, rel, df / max(rms, 1e-12))
        print(f"  {kind:<12} {e_ref:14.6f} {e_test:14.6f} {rel:9.1e} {df:10.2e} {rms:10.2e}")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--systems", nargs="+", default=["cluster", "box"])
    args = parser.parse_args()
    ff, nagl = load_model()
    worst = 0.0
    if "cluster" in args.systems:
        worst = max(worst, compare("cluster", *cluster_topology(), False, args.platform, ff, nagl))
    if "box" in args.systems:
        worst = max(worst, compare("methanol box", *box_topology(), True, args.platform, ff, nagl))
    print(f"\nlargest relative difference: {worst:.1e}")


if __name__ == "__main__":
    main()
