"""
Compare the plugin forces with the InteraxFF export they replace, term by term, on real InteraxFF systems.

Builds the systems with interax.openmm.create_system_from_topology, converts them with
interaxplugin_convert.convert_system, and compares the energy and the forces of each term:

  cluster   a non-periodic cluster of H/C/N/O/S/Cl molecules (no cutoff)
  box       a periodic box of liquid methanol (0.9 nm cutoff, switching from 0.8 nm, dispersion long range correction)

Usage: python compare_export.py [--platform Reference] [--precision double] [--overlap export] [--systems cluster box]

By default the Slater overlap follows the Taylor scheme of the model (interax.openmm.CONVERT_OVERLAP[ff.slater_taylor]:
"export" for order2, "exact" for order6). With --overlap export the plugin uses the method that reproduces the Taylor
scheme of the export, and must reproduce it to rounding; with --overlap exact it uses its accurate scheme, which differs
from the order-2 export by the export's truncation error and equals the order-6 export to rounding. The export builds
the three-body dispersion as an InteraxThreeBodyForce already, which the conversion keeps: its row compares the platform
with the Reference platform.
Run in a Python environment with OpenMM, the plugin and InteraxFF's dependencies; the plugin, InteraxFF and the model
(INTERAX_MODEL) are found as plugin_path.py describes. The base force field's charges (intramolecular terms only) are
AM1-BCC when AmberTools or OpenEye can compute them, else Gasteiger (INTERAX_BASE_CHARGES=am1bcc|gasteiger forces one);
packmol is looked for on PATH, then next to the Python interpreter.
"""
import argparse
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402  (load the env's libstdc++ before anything else)
import openmm  # noqa: E402
import openmm.unit as unit  # noqa: E402

from plugin_path import INTERAX_MODEL  # noqa: E402  (INTERAX_PLUGIN_PREFIX, INTERAX_REPO, INTERAX_MODEL)

import interaxplugin  # noqa: E402
import interaxplugin_convert  # noqa: E402
from openff.interchange.components._packmol import UNIT_CUBE, pack_box  # noqa: E402
from openff.toolkit import Molecule, Topology  # noqa: E402
from openff.units import unit as off_unit  # noqa: E402

from interax.force_field import ForceField  # noqa: E402
from interax.models import NAGLModel, PerAtomicNumberModel  # noqa: E402
from interax.openmm import COMPONENTS, CONVERT_OVERLAP, create_system_from_topology  # noqa: E402

# packmol (pack_box) from the environment of this interpreter when it is not on PATH
if shutil.which("packmol") is None:
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")


def _base_charges():
    """'am1bcc' or 'gasteiger' (INTERAX_BASE_CHARGES, else AM1-BCC if a toolkit can compute it)."""
    choice = os.environ.get("INTERAX_BASE_CHARGES", "auto").lower()
    if choice != "auto":
        return choice
    from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY, AmberToolsToolkitWrapper, OpenEyeToolkitWrapper

    # the toolkits the registry holds, not just installed binaries: it is built at import, before PATH changes
    registered = GLOBAL_TOOLKIT_REGISTRY.registered_toolkits
    return "am1bcc" if any(isinstance(t, (AmberToolsToolkitWrapper, OpenEyeToolkitWrapper)) for t in registered) else "gasteiger"


if _base_charges() == "gasteiger":
    # The base force field's charges only enter the intramolecular terms, which every comparison here leaves unchanged
    from openff.interchange import Interchange  # noqa: E402

    _from_smirnoff = Interchange.from_smirnoff.__func__

    def _with_gasteiger(cls, force_field, topology, **kwargs):
        molecules = []
        for molecule in topology.unique_molecules:
            molecule = Molecule(molecule)
            molecule.assign_partial_charges("gasteiger")
            molecules.append(molecule)
        return _from_smirnoff(cls, force_field, topology, charge_from_molecules=molecules, **kwargs)

    Interchange.from_smirnoff = classmethod(_with_gasteiger)

# Each term: the per-particle parameters of its CustomNonbondedForce in the export, and its plugin force
KINDS = {
    "dispersion": ({"m1", "m2", "m3", "alpha", "s"}, interaxplugin.InteraxDispersionForce),
    "exchange": ({"A", "s"}, interaxplugin.InteraxExchangeForce),
    "short_range": ({"K", "L", "s"}, interaxplugin.InteraxShortRangeForce),
    "three_body": (None, interaxplugin.InteraxThreeBodyForce),
}


def _names(force):
    return {force.getPerParticleParameterName(k) for k in range(force.getNumPerParticleParameters())}


def load_model(model=None):
    """The force field in the directory model (default INTERAX_MODEL)."""
    model = model or INTERAX_MODEL
    if model is None:
        raise RuntimeError("no InteraxFF model: set INTERAX_MODEL")
    ff = ForceField(NAGLModel.load(f"{model}/nagl_model.pt"), PerAtomicNumberModel.load(f"{model}/per_atomic_number_model.pt", device="cpu"))
    ff.load_state_dict(torch.load(f"{model}/force_field_state.pt", map_location="cpu"))
    return ff.eval()


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
    # packmol works in a temporary directory, removed afterwards even if it fails
    with tempfile.TemporaryDirectory() as directory:
        topology = pack_box(molecules=[mol], number_of_copies=[140], target_density=0.79 * off_unit.gram / off_unit.milliliter,
                            box_shape=UNIT_CUBE,
                            working_directory=directory)
    return topology, topology.get_positions().m_as(off_unit.nanometer)


def terms_by_kind(system):
    """Map each InteraxFF term to the index of its force, in an export or a converted System."""
    found = {}
    for i, force in enumerate(system.getForces()):
        for kind, (names, plugin_class) in KINDS.items():
            exported = names is not None and isinstance(force, openmm.CustomNonbondedForce) and names <= _names(force)
            if exported or plugin_class.isinstance(force):
                found[kind] = i
    return found


def evaluate(system, positions, box, platform, indices, precision="double"):
    """Energy (kJ/mol) and forces (kJ/mol/nm) of each force in indices, each evaluated in its own group."""
    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for force in system.getForces():
        force.setForceGroup(31)
    for group, i in enumerate(indices.values()):
        system.getForce(i).setForceGroup(group)
    properties = {"Precision": precision} if platform == "CUDA" else {}
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


def compare(name, topology, positions, periodic, platform, precision, overlap, ff):
    system = create_system_from_topology(topology, ff, nonbonded_cutoff=0.9, components=COMPONENTS)
    box = system.getDefaultPeriodicBoxVectors() if periodic else None
    converted = interaxplugin_convert.convert_system(system, overlap=overlap)
    export_terms, plugin_terms = terms_by_kind(system), terms_by_kind(converted)
    assert set(export_terms) == set(plugin_terms) == set(KINDS), (export_terms, plugin_terms)
    reference = evaluate(system, positions, box, "Reference", export_terms)
    test = evaluate(converted, positions, box, platform, plugin_terms, precision)
    print(f"\n{name}: {system.getNumParticles()} particles, plugin on {platform} ({precision}, {overlap} overlap) "
          f"vs {ff.slater_taylor} export on Reference")
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
    parser.add_argument("--precision", default="double", help="CUDA precision")
    parser.add_argument("--overlap", choices=["export", "exact"], help="default: the scheme of the model (CONVERT_OVERLAP)")
    parser.add_argument("--systems", nargs="+", default=["cluster", "box"])
    args = parser.parse_args()
    ff = load_model()
    overlap = args.overlap or CONVERT_OVERLAP[ff.slater_taylor]
    worst = 0.0
    if "cluster" in args.systems:
        worst = max(worst, compare("cluster", *cluster_topology(), False, args.platform, args.precision, overlap, ff))
    if "box" in args.systems:
        worst = max(worst, compare("methanol box", *box_topology(), True, args.platform, args.precision, overlap, ff))
    print(f"\nlargest relative difference: {worst:.1e}")


if __name__ == "__main__":
    main()
