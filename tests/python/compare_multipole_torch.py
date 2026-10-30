"""InteraxMultipoleForce (exact exponents) vs torch: E_static + E_ind of a dimer, for polarization options.
Usage: compare_multipole_torch.py SMILES MODEL_DIR MUTUAL_ORDER KAPPA CORE SHIFT   (needs INTERAX_REPO: studies/audit/_common.py)"""
import os, sys
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np, openmm
from openmm import unit
from plugin_path import INTERAX_REPO
sys.path.insert(0, os.path.join(INTERAX_REPO, "studies", "audit"))
import interaxplugin_convert
from openff.toolkit import Molecule, Topology
from interax.force_field import ForceField
from interax.models import NAGLModel, PerAtomicNumberModel
from interax.openmm import create_system_from_topology
from _common import batch
smiles, model, order, kappa, core, shift = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "1", float(sys.argv[6])
M = model
nagl = NAGLModel.load(f"{M}/nagl_model.pt"); ff = ForceField(nagl, PerAtomicNumberModel.load(f"{M}/per_atomic_number_model.pt", device="cpu"))
ff.load_state_dict(torch.load(f"{M}/force_field_state.pt", map_location="cpu"), strict=False); ff.eval()
pol = ff._induction
mols = [Molecule.from_smiles(smiles) for _ in range(2)]
for m in mols: m.generate_conformers(n_conformers=1)
xa = mols[0].conformers[0].m_as("nanometer"); B = 18.8972612
xb = xa + np.array([shift, 0.05, 0.02])
with torch.no_grad():
    pol.mutual_order.fill_(order); pol.mutual_damping_scale.fill_(1.0); pol.core_damping.fill_(False)
    pol.mutual_lambda.fill_(1.0)
system = create_system_from_topology(Topology.from_molecules(mols), ff, nagl, components=["E_static", "E_ind"], electrostatics="interax",
                                     polarization="mutual" if order > 0 else "direct")
hippo = next(interaxplugin_convert.wrapped_hippo(f) for f in system.getForces() if interaxplugin_convert.wrapped_hippo(f) is not None)
with torch.no_grad():
    pol.mutual_damping_scale.fill_(kappa); pol.core_damping.fill_(core)
    o = ff(batch(mols[0], nagl, xa*B, dtype=torch.float32), batch(mols[1], nagl, xb*B, dtype=torch.float32), components=["E_static", "E_ind"])
    s_exact = np.concatenate([ff.get_parameters(batch(m, nagl, xa*B, dtype=torch.float32))["s"][0].double().numpy() for m in mols]) * 0.0529177210903
t = {k: float(v[0])*2625.4996394799 for k, v in o.items()}
f = interaxplugin_convert.convert_multipoles(hippo, core_damping=core, mutual_damping_scale=kappa)
for i in range(f.getNumParticles()):
    q = f.getParticleParametersList(i)
    f.setParticleParameters(i, q[0], q[1:4], q[4:13], q[13], 1/s_exact[i], q[15], int(q[16]), int(q[17]), int(q[18]), int(q[19]), bool(q[20]))
s = openmm.System()
for i in range(system.getNumParticles()): s.addParticle(1.0)
s.addForce(f)
c = openmm.Context(s, openmm.VerletIntegrator(1e-3), openmm.Platform.getPlatformByName("Reference")); c.setPositions(np.vstack([xa, xb]))
e = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
tt = t["E_static"] + t["E_ind"]
print(f"{smiles:10s} order {order} kappa {kappa} core {core!s:5s} shift {shift}: torch {tt:.6f} (ind {t['E_ind']:.6f}) plugin {e:.6f} diff {tt-e:+.2e}")
