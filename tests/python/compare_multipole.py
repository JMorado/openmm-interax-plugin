"""
Compare InteraxMultipoleForce with the HippoNonbondedForce it replaces, on real InteraxFF systems.

Builds the systems with interax.openmm.create_system_from_topology (electrostatics and polarization only), converts
the HIPPO force with interaxplugin_convert.convert_multipoles (HIPPO's physics: no core damping, kappa = 1), and
compares the energy, the forces and the induced dipoles:

  cluster   a non-periodic cluster of H/C/N/O/S/Cl molecules (no cutoff)
  box       a periodic box of liquid methanol (PME, 0.9 nm cutoff)

Usage: python compare_multipole.py [--platform Reference] [--precision double] [--systems cluster box]
The model (INTERAX_MODEL) should have quadrupoles and mutual polarization to test everything, e.g. models_v3R4bfdE1MQ.
"""
import argparse

from compare_export import box_topology, cluster_topology, interaxplugin, interaxplugin_convert, load_model, np, openmm, unit  # noqa: E402
from plugin_path import model  # noqa: E402

from interax.openmm import create_system_from_topology, extend_positions_for_sites  # noqa: E402

MODEL = model("models_v3R4bfdE1MQ")


def split(system, core_damping=False, kappa=1.0):
    """Two Systems with the particles of system: one with its HIPPO force, one with the converted InteraxMultipoleForce."""
    hippo_system, plugin_system = openmm.System(), openmm.System()
    for s in (hippo_system, plugin_system):
        s.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        for i in range(system.getNumParticles()):
            s.addParticle(system.getParticleMass(i))
            if system.isVirtualSite(i):
                raise NotImplementedError("models with virtual sites are not supported by this comparison")
    hippo = next(interaxplugin_convert.wrapped_hippo(f) for f in system.getForces() if interaxplugin_convert.wrapped_hippo(f) is not None)
    hippo_system.addForce(openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(hippo)))
    plugin_system.addForce(interaxplugin_convert.convert_multipoles(hippo, core_damping=core_damping, mutual_damping_scale=kappa))
    return hippo_system, plugin_system


def evaluate(system, positions, box, platform, precision):
    properties = {"Precision": precision} if platform in ("CUDA", "OpenCL") else {}
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform), properties)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    force = system.getForce(0)
    if isinstance(force, openmm.HippoNonbondedForce):
        dipoles = np.array([[v.value_in_unit(unit.nanometer) if hasattr(v, "unit") else v for v in d] for d in
                            force.getInducedDipoles(context)])
    else:
        dipoles = np.array(interaxplugin.InteraxMultipoleForce.cast(force).getInducedDipolesList(context)).reshape(-1, 3)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer), dipoles)


def finite_difference_error(system, positions, box, forces, atoms, h=1e-5):
    """Largest |F - F_fd| over the given atoms, relative to the rms force (central differences, Reference platform)."""
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    x = np.array(positions.value_in_unit(unit.nanometer) if hasattr(positions, "unit") else positions, dtype=float)
    worst = 0.0
    for a in atoms:
        for k in range(3):
            e = []
            for sign in (1, -1):
                y = x.copy()
                y[a, k] += sign * h
                context.setPositions(y)
                e.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            worst = max(worst, abs(forces[a, k] + (e[0] - e[1]) / (2 * h)))
    return worst / np.sqrt((forces ** 2).mean())


def compare(name, topology, positions, periodic, platform, precision, ff, nagl, core_damping=False, kappa=1.0):
    system = create_system_from_topology(topology, ff, nagl, electrostatics="interax", nonbonded_cutoff=0.9, components=["E_static", "E_ind"])
    positions = extend_positions_for_sites(positions, system)
    box = system.getDefaultPeriodicBoxVectors() if periodic else None
    hippo_system, plugin_system = split(system, core_damping, kappa)
    # With HIPPO's physics the reference is HIPPO on the Reference platform, otherwise InteraxMultipoleForce on Reference
    reference_system = hippo_system if (not core_damping and kappa == 1.0) else plugin_system
    e_ref, f_ref, d_ref = evaluate(reference_system, positions, box, "Reference", "double")
    e, f, d = evaluate(plugin_system, positions, box, platform, precision)
    rms = np.sqrt((f_ref ** 2).mean())
    rel_e = abs(e - e_ref) / abs(e_ref)
    rel_f = np.abs(f - f_ref).max() / rms
    rel_d = np.abs(d - d_ref).max() / np.abs(d_ref).max()
    reference = "HIPPO" if reference_system is hippo_system else f"InteraxMultipoleForce (core damping {core_damping}, kappa {kappa})"
    print(f"{name}: {system.getNumParticles()} particles, InteraxMultipoleForce on {platform} ({precision}) vs {reference} on Reference")
    print(f"  energy {e_ref:.8f} vs {e:.8f} kJ/mol (rel {rel_e:.1e}); max |dF| / rms F {rel_f:.1e}; max |d mu| / max mu {rel_d:.1e}")
    # HIPPO's reference forces on ZOnly atoms are undefined (they read the position of particle -1), so the forces are
    # also checked against finite differences of the energy, on a sample of atoms that includes every frame type
    force = interaxplugin.InteraxMultipoleForce.cast(plugin_system.getForce(0))
    axis = np.array([force.getParticleParametersList(i)[16] for i in range(force.getNumParticles())])
    atoms = sorted({int(np.nonzero(axis == t)[0][0]) for t in np.unique(axis)} | set(range(0, len(axis), max(1, len(axis) // 6))))
    rel_fd = finite_difference_error(plugin_system, positions, box, f, atoms)
    print(f"  forces vs finite differences on {len(atoms)} atoms (frame types {sorted(set(axis[atoms].astype(int)))}): max |dF| / rms F {rel_fd:.1e}")
    return max(rel_e, rel_d, rel_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--precision", default="double", help="CUDA/OpenCL precision")
    parser.add_argument("--systems", nargs="+", default=["cluster", "box"])
    parser.add_argument("--core-damping", action="store_true")
    parser.add_argument("--kappa", type=float, default=1.0, help="mutual damping scale")
    parser.add_argument("--no-fd", action="store_true", help="skip the finite-difference check")
    args = parser.parse_args()
    ff, nagl = load_model(MODEL)
    worst = 0.0
    if "cluster" in args.systems:
        worst = max(worst, compare("cluster", *cluster_topology(), False, args.platform, args.precision, ff, nagl, args.core_damping, args.kappa))
    if "box" in args.systems:
        worst = max(worst, compare("methanol box", *box_topology(), True, args.platform, args.precision, ff, nagl, args.core_damping, args.kappa))
    print(f"largest relative difference: {worst:.1e}")


if __name__ == "__main__":
    main()
