"""HIPPO and InteraxMultipoleForce on CUDA, bare and wrapped in a CustomCVForce, against the Reference platform (PME).

Historical: needs the export's electrostatics="hippo" path, removed from InteraxFF on 2026-10-30 (it now raises)."""
import os
os.environ.setdefault("INTERAX_PLUGIN_PREFIX", "/shared/joaomorado/opt/openmm-interax-v3")
from benchmark_multipole import COMPONENTS, build, context, electrostatics_system, energy_forces, interaxplugin_convert, load_model, np, openmm, wrapped
from interax.openmm import create_system_from_topology
ff, nagl = load_model()
for name in ("methanol", "hfe"):
    top, x = build(name)
    s = create_system_from_topology(top, ff, nagl, electrostatics="hippo", nonbonded_cutoff=0.9, components=COMPONENTS)
    h = interaxplugin_convert.wrapped_hippo(next(f for f in s.getForces() if interaxplugin_convert.wrapped_hippo(f) is not None))
    clone = lambda f: openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(f))  # noqa: E731
    e_ref, f_ref = energy_forces(context(electrostatics_system(s, clone(h)), x, "Reference", "double"))
    print(f"{name}: Reference {e_ref:.4f}")
    for label, force in (("HIPPO bare", clone(h)), ("HIPPO wrapped", wrapped(clone(h))),
                         ("Interax bare", interaxplugin_convert.convert_multipoles(h)), ("Interax wrapped", wrapped(interaxplugin_convert.convert_multipoles(h)))):
        e, f = energy_forces(context(electrostatics_system(s, force), x, "CUDA", "double"))
        print(f"  {label:16s} {e:.4f}  dE {e - e_ref:+.3e}  max|dF|/rms {np.abs(f - f_ref).max() / np.sqrt((f_ref**2).mean()):.1e}")
