"""
The end states of the alchemical scaling of InteraxMultipoleForce (Reference platform, model INTERAX_MODEL or MQ): with
the first molecule (ethanol, in three waters) alchemical, lambda_electrostatics = 1 must equal the full force and 0 the
force without the molecule's multipoles and polarizabilities, with and without core damping, also through a
CustomCVForce (which passes the parameter on to its inner context).
"""
import sys
sys.path.insert(0, "/shared/joaomorado/opt/openmm-interax-v3/python")
sys.path.insert(0, "/shared/joaomorado/repos/InteraxFF")
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np, openmm
from openmm import unit
openmm.Platform.loadPluginsFromDirectory("/shared/joaomorado/opt/openmm-interax-v3/lib/plugins")
import interaxplugin, interaxplugin_convert
from openff.toolkit import Molecule, Topology
from interax.force_field import ForceField
from interax.models import NAGLModel, PerAtomicNumberModel
from interax.openmm import create_system_from_topology
M = "/shared/joaomorado/repos/InteraxFF/examples/models_v3R4bfdE1MQ"
nagl = NAGLModel.load(f"{M}/nagl_model.pt"); ff = ForceField(nagl, PerAtomicNumberModel.load(f"{M}/per_atomic_number_model.pt", device="cpu"))
ff.load_state_dict(torch.load(f"{M}/force_field_state.pt", map_location="cpu"), strict=False); ff.eval()
mols = [Molecule.from_smiles(s) for s in ("CCO", "O", "O", "O")]
X = []
for k, m in enumerate(mols):
    m.generate_conformers(n_conformers=1); X.append(m.conformers[0].m_as("nanometer") + np.array([0.3*k, 0.07*k, -0.05*k]))
X = np.vstack(X); na = mols[0].n_atoms
system = create_system_from_topology(Topology.from_molecules(mols), ff, nagl, components=["E_static", "E_ind"], electrostatics="interax")
hippo = next(interaxplugin_convert.wrapped_hippo(f) for f in system.getForces() if interaxplugin_convert.wrapped_hippo(f) is not None)
def ctx(force):
    s = openmm.System()
    for i in range(system.getNumParticles()): s.addParticle(1.0)
    s.addForce(force)
    c = openmm.Context(s, openmm.VerletIntegrator(1e-3), openmm.Platform.getPlatformByName("Reference")); c.setPositions(X); return c
E = lambda c: c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
for core in (False, True):
    alch = interaxplugin_convert.convert_multipoles(hippo, core_damping=core, alchemical_particles=range(na))
    c = ctx(alch)
    e1 = E(c); c.setParameter("lambda_electrostatics", 0.0); e0 = E(c); c.setParameter("lambda_electrostatics", 0.5); eh = E(c)
    full = ctx(interaxplugin_convert.convert_multipoles(hippo, core_damping=core))
    off = interaxplugin_convert.convert_multipoles(hippo, core_damping=core)
    for i in range(na):
        q = off.getParticleParametersList(i)
        off.setParticleParameters(i, 0.0, [0, 0, 0], [0]*9, 0.0, q[14], 0.0, int(q[16]), int(q[17]), int(q[18]), int(q[19]), False)
    print(f"core {core}: lambda=1 {e1:.8f} vs full {E(full):.8f} | lambda=0 {e0:.8f} vs solute off {E(ctx(off)):.8f} | lambda=0.5 {eh:.6f}")
