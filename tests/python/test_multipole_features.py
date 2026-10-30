"""
The InteraxFF options of InteraxMultipoleForce on a GPU platform against the Reference platform, on real InteraxFF
systems (electrostatics and polarization only):

  options        core damping, its range eta and the mutual damping scale kappa, together
  intramolecular polarization="mutual-intra": the induced dipoles of a molecule couple (exceptions with dipole-dipole
                 scale 1, 0 for bonded pairs)
  lambda         the first molecule alchemical, lambda_electrostatics = 0.3 and the end states (lambda = 0 against the
                 system without its multipoles and polarizabilities)
  serialization  an XML round trip of the force gives the same energy
  own kernel     setUseOwnKernel(True): the force alone, and in one context with a NonbondedForce (Lennard-Jones, same
                 exclusions) that uses the default nonbonded kernel, against the Reference platform

Systems: the non-periodic cluster and the periodic methanol box of compare_export.py.
Usage: python test_multipole_features.py [--platform CUDA] [--precision double], or python -m pytest
test_multipole_features.py (CUDA in double precision; skipped without a CUDA device).
"""
import argparse
import os

os.environ.setdefault("INTERAX_PLUGIN_PREFIX", "/shared/joaomorado/opt/openmm-interax-v8")
os.environ.setdefault("INTERAX_MODEL", "/shared/joaomorado/repos/InteraxFF/examples/models_v3R4bfdE1MQ")

from compare_export import box_topology, cluster_topology, interaxplugin, interaxplugin_convert, load_model, np, openmm, unit  # noqa: E402

from interax.openmm import create_system_from_topology  # noqa: E402


def multipoles_of(system):
    """The InteraxMultipoleForce of an export (electrostatics="interax", wrapped in a CustomCVForce "E_hippo")."""
    for f in system.getForces():
        if isinstance(f, openmm.CustomCVForce):
            for k in range(f.getNumCollectiveVariables()):
                cv = f.getCollectiveVariable(k)
                if interaxplugin.InteraxMultipoleForce.isinstance(cv):
                    return cv
    raise ValueError("no InteraxMultipoleForce in the system")


def variant(base, core_damping=False, core_damping_scale=1.0, mutual_damping_scale=1.0, alchemical_particles=()):
    """A copy of an exported InteraxMultipoleForce with the given options (the defaults are HIPPO's), as
    convert_multipoles made from the HippoNonbondedForce before the export's HIPPO path was removed (2026-10-30).
    Returns (force, typed view): add the force to a System, change it through the view."""
    force = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(base))
    typed = interaxplugin.InteraxMultipoleForce.cast(force)
    typed.setCoreDamping(core_damping)
    typed.setCoreDampingScale(core_damping_scale)
    typed.setMutualDampingScale(mutual_damping_scale)
    alchemical = set(alchemical_particles)
    for i in range(typed.getNumParticles()):
        p = typed.getParticleParametersList(i)
        typed.setParticleParameters(i, p[0], p[1:4], p[4:13], p[13], p[14], p[15], int(p[16]), int(p[17]), int(p[18]), int(p[19]),
                                    i in alchemical)
    return force, typed


def single(system, force):
    s = openmm.System()
    s.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    for i in range(system.getNumParticles()):
        s.addParticle(system.getParticleMass(i))
    s.addForce(force)
    return s


def evaluate(system, positions, platform, precision, parameters=None):
    properties = {"Precision": precision} if platform in ("CUDA", "OpenCL") else {}
    c = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform), properties)
    c.setPositions(positions)
    for name, value in (parameters or {}).items():
        c.setParameter(name, value)
    state = c.getState(getEnergy=True, getForces=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


def report(label, test, ref, tolerance):
    (e, f), (e_ref, f_ref) = test, ref
    rel_e = abs(e - e_ref) / max(abs(e_ref), 1e-12)
    rel_f = np.abs(f - f_ref).max() / max(np.sqrt((f_ref ** 2).mean()), 1e-12)
    ok = rel_e < tolerance and rel_f < 100 * tolerance
    print(f"  {'ok  ' if ok else 'FAIL'} {label:44s} E {e:16.6f} vs {e_ref:16.6f} (rel {rel_e:.1e}), max |dF| / rms F {rel_f:.1e}")
    return ok


def run(name, topology, positions, platform, precision, ff, nagl, tolerance):
    print(f"{name}: {topology.n_atoms} atoms, {platform} ({precision}) vs Reference")
    ok = True
    system = create_system_from_topology(topology, ff, nagl, electrostatics="interax", nonbonded_cutoff=0.9, components=["E_static", "E_ind"])
    base = multipoles_of(system)
    na = topology.molecule(0).n_atoms

    # core damping, eta and kappa
    force, _ = variant(base, core_damping=True, core_damping_scale=0.6, mutual_damping_scale=0.8)
    s = single(system, force)
    ok &= report("core damping, eta 0.6, kappa 0.8", evaluate(s, positions, platform, precision), evaluate(s, positions, "Reference", "double"), tolerance)

    # intramolecular coupling of the induced dipoles
    intra = create_system_from_topology(topology, ff, nagl, electrostatics="interax", nonbonded_cutoff=0.9, components=["E_static", "E_ind"],
                                        polarization="mutual-intra")
    s = single(system, variant(multipoles_of(intra), core_damping=True)[0])
    ok &= report("mutual-intra, core damping", evaluate(s, positions, platform, precision), evaluate(s, positions, "Reference", "double"), tolerance)

    # alchemical scaling
    alchemical = single(system, variant(base, core_damping=True, alchemical_particles=range(na))[0])
    lam = {"lambda_electrostatics": 0.3}
    ok &= report("lambda_electrostatics 0.3", evaluate(alchemical, positions, platform, precision, lam),
                 evaluate(alchemical, positions, "Reference", "double", lam), tolerance)
    off_force, off = variant(base, core_damping=True)
    for i in range(na):
        p = off.getParticleParametersList(i)
        off.setParticleParameters(i, 0.0, [0, 0, 0], [0] * 9, 0.0, p[14], 0.0, int(p[16]), int(p[17]), int(p[18]), int(p[19]), False)
    ok &= report("lambda_electrostatics 0 = solute removed", evaluate(alchemical, positions, platform, precision, {"lambda_electrostatics": 0.0}),
                 evaluate(single(system, off_force), positions, "Reference", "double"), tolerance)

    # serialization (of a force not yet owned by a System, which the comparison then owns)
    force, _ = variant(base, core_damping=True, core_damping_scale=0.6, mutual_damping_scale=0.8, alchemical_particles=range(na))
    xml = openmm.XmlSerializer.serialize(force)
    reference = single(system, force)
    clone = single(system, openmm.XmlSerializer.deserialize(xml))
    # two GPU contexts in single or mixed precision differ by rounding (the order of the atomic sums)
    ok &= report("XML round trip", evaluate(clone, positions, platform, precision, lam),
                 evaluate(reference, positions, platform, precision, lam), 1e-12 if precision == "double" else 1e-7)

    # own kernel: alone, and sharing the context (and neighbor list) with a NonbondedForce
    force, typed = variant(base, core_damping=True, core_damping_scale=0.6, mutual_damping_scale=0.8)
    typed.setUseOwnKernel(True)
    if not interaxplugin.InteraxMultipoleForce.cast(openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force))).getUseOwnKernel():
        print("  FAIL useOwnKernel lost in the XML round trip")
        ok = False
    own = single(system, force)
    ok &= report("own kernel", evaluate(own, positions, platform, precision), evaluate(own, positions, "Reference", "double"), tolerance)
    lj = openmm.NonbondedForce()
    periodic = typed.getNonbondedMethod() == interaxplugin.InteraxMultipoleForce.PME
    lj.setNonbondedMethod(openmm.NonbondedForce.PME if periodic else openmm.NonbondedForce.NoCutoff)
    lj.setCutoffDistance(typed.getCutoffDistance())
    lj.setUseDispersionCorrection(False)
    for i in range(typed.getNumParticles()):
        lj.addParticle(0.0, 0.3, 0.2)
    offset = 0  # the exceptions of the export: every intramolecular pair
    for molecule in topology.molecules:
        for i in range(offset, offset + molecule.n_atoms):
            for j in range(i + 1, offset + molecule.n_atoms):
                lj.addException(i, j, 0.0, 1.0, 0.0)
        offset += molecule.n_atoms
    assert lj.getNumExceptions() == typed.getNumExceptions()
    lj.setForceGroup(1)
    own.addForce(lj)
    ok &= report("own kernel + NonbondedForce, one context", evaluate(own, positions, platform, precision),
                 evaluate(own, positions, "Reference", "double"), tolerance)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="double")
    args = parser.parse_args()
    ff, nagl = load_model()
    tolerance = 1e-5 if args.precision == "double" else 2e-4
    ok = run("cluster", *cluster_topology(), args.platform, args.precision, ff, nagl, tolerance)
    ok &= run("methanol box", *box_topology(), args.platform, args.precision, ff, nagl, tolerance)
    print("all passed" if ok else "SOME FAILED")


def test_multipole_features():
    import pytest

    try:
        openmm.Context(openmm.System(), openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("CUDA"))
    except Exception:
        pytest.skip("no usable CUDA platform")
    ff, nagl = load_model()
    ok = run("cluster", *cluster_topology(), "CUDA", "double", ff, nagl, 1e-5)
    ok &= run("methanol box", *box_topology(), "CUDA", "double", ff, nagl, 1e-5)
    assert ok


if __name__ == "__main__":
    main()
