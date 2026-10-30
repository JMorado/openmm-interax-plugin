"""
The ffxml generators of interaxplugin_forcefield: a water force field written as ffxml, used through
openmm.app.ForceField.createSystem(), gives the same forces and energies as the plugin forces built directly with the
API, with and without PME. Also checks that a force tag can be split over two files (parameters merged, settings
checked).

Usage: python -m pytest test_forcefield.py      (INTERAX_PLUGIN_PREFIX: where the plugin is installed)
"""
import io
import os
import sys

import numpy as np
import pytest

PLUGIN_PREFIX = os.environ.get("INTERAX_PLUGIN_PREFIX", "/shared/joaomorado/opt/openmm-interax")
sys.path.insert(0, f"{PLUGIN_PREFIX}/python")
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

import openmm  # noqa: E402
from openmm import app, unit  # noqa: E402

openmm.Platform.loadPluginsFromDirectory(f"{PLUGIN_PREFIX}/lib/plugins")
import interaxplugin as ip  # noqa: E402
import interaxplugin_forcefield  # noqa: E402,F401

QS = 1e-4
O = dict(charge=-0.4, core=6.0, alpha=40.0, pol=0.0009, dipole=(0.0, 0.0, 0.003), quad=(QS, 0, 0, 0, -0.6 * QS, 0, 0, 0, -0.4 * QS),
         m1=0.12, m2=0.02, m3=0.005, adisp=0.0012, s=0.03, A=40.0, K=-3.0, L=2.0, w=1.1e3)
H = dict(charge=0.2, core=1.0, alpha=45.0, pol=0.0003, dipole=(0.0, 0.0, -0.002), quad=(0,) * 9,
         m1=0.03, m2=0.004, m3=0.001, adisp=0.0003, s=0.025, A=15.0, K=-1.0, L=1.5, w=1.4e3)


def ffxml(multipole_settings='extrapolationCoefficients="0.0,0.5,0.5"', split=False):
    def atom(t, p, axis, frame):
        q = " ".join(f'q{a + 1}{b + 1}="{p["quad"][3 * a + b]}"' for a in range(3) for b in range(3))
        return (f'<Atom type="{t}" charge="{p["charge"]}" coreCharge="{p["core"]}" alpha="{p["alpha"]}" polarizability="{p["pol"]}" '
                f'axisType="{axis}" d0="{p["dipole"][0]}" d1="{p["dipole"][1]}" d2="{p["dipole"][2]}" {q} {frame}/>')
    multipole_o = atom("w-O", O, 1, 'zAtomType="w-H" xAtomType="w-H"')
    multipole_h = atom("w-H", H, 0, 'zAtomType="w-O" xAtomType="w-H"')
    head = """<ForceField>
 <AtomTypes><Type name="w-O" class="w-O" element="O" mass="15.999"/><Type name="w-H" class="w-H" element="H" mass="1.008"/></AtomTypes>
 <Residues><Residue name="HOH"><Atom name="O" type="w-O"/><Atom name="H1" type="w-H"/><Atom name="H2" type="w-H"/>
  <Bond atomName1="O" atomName2="H1"/><Bond atomName1="O" atomName2="H2"/></Residue></Residues>"""
    body_o = f"""
 <InteraxMultipoleForce {multipole_settings}>{multipole_o}</InteraxMultipoleForce>
 <InteraxDispersionForce dampingScale="1.1"><Atom type="w-O" m1="{O['m1']}" m2="{O['m2']}" m3="{O['m3']}" alpha="{O['adisp']}" width="{O['s']}"/></InteraxDispersionForce>
 <InteraxExchangeForce><Atom type="w-O" amplitude="{O['A']}" width="{O['s']}"/></InteraxExchangeForce>
 <InteraxShortRangeForce><Atom type="w-O" k="{O['K']}" l="{O['L']}" width="{O['s']}"/></InteraxShortRangeForce>
 <InteraxThreeBodyForce scaleFactor="0.8" centroidDampingRadius="0.25"><Atom type="w-O" alpha="{O['adisp']}" frequency="{O['w']}"/></InteraxThreeBodyForce>"""
    body_h = f"""
 <InteraxMultipoleForce {multipole_settings}>{multipole_h}</InteraxMultipoleForce>
 <InteraxDispersionForce dampingScale="1.1"><Atom type="w-H" m1="{H['m1']}" m2="{H['m2']}" m3="{H['m3']}" alpha="{H['adisp']}" width="{H['s']}"/></InteraxDispersionForce>
 <InteraxExchangeForce><Atom type="w-H" amplitude="{H['A']}" width="{H['s']}"/></InteraxExchangeForce>
 <InteraxShortRangeForce><Atom type="w-H" k="{H['K']}" l="{H['L']}" width="{H['s']}"/></InteraxShortRangeForce>
 <InteraxThreeBodyForce scaleFactor="0.8" centroidDampingRadius="0.25"><Atom type="w-H" alpha="{H['adisp']}" frequency="{H['w']}"/></InteraxThreeBodyForce>"""
    if split:
        return [head + body_o + "\n</ForceField>", "<ForceField>" + body_h + "\n</ForceField>"]
    return [head + body_o + body_h + "\n</ForceField>"]


def water_box(n=12, box=2.0, seed=3):
    rng = np.random.default_rng(seed)
    top = app.Topology()
    chain = top.addChain()
    pos = []
    for k in range(n):
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", app.element.oxygen, res)
        h1 = top.addAtom("H1", app.element.hydrogen, res)
        h2 = top.addAtom("H2", app.element.hydrogen, res)
        top.addBond(o, h1)
        top.addBond(o, h2)
        c = rng.uniform(0.2, box - 0.2, 3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pos += [c, c + q @ [0.0957, 0, 0], c + q @ [-0.024, 0.0927, 0]]
    top.setPeriodicBoxVectors(np.eye(3) * box * unit.nanometer)
    return top, np.array(pos)


def direct_system(top, periodic, cutoff=0.9):
    """The same forces built with the plugin API."""
    n = top.getNumAtoms()
    system = openmm.System()
    for a in top.atoms():
        system.addParticle(15.999 if a.element.symbol == "O" else 1.008)
    if periodic:
        system.setDefaultPeriodicBoxVectors(*top.getPeriodicBoxVectors())
    molecules = [[3 * k, 3 * k + 1, 3 * k + 2] for k in range(n // 3)]
    mp = ip.InteraxMultipoleForce()
    disp, exch, src = ip.InteraxDispersionForce(), ip.InteraxExchangeForce(), ip.InteraxShortRangeForce()
    disp.setDampingScale(1.1)
    for m in molecules:
        o, h1, h2 = m
        for i, p, axis, z, x in ((o, O, 1, h1, h2), (h1, H, 0, o, h2), (h2, H, 0, o, h1)):
            mp.addParticle(p["charge"], list(p["dipole"]), list(p["quad"]), p["core"], p["alpha"], p["pol"], axis, z, x, -1)
            disp.addParticle(p["m1"], p["m2"], p["m3"], p["adisp"], p["s"])
            exch.addParticle(p["A"], p["s"])
            src.addParticle(p["K"], p["L"], p["s"])
    for m in molecules:
        for a in range(3):
            for b in range(a + 1, 3):
                mp.addException(m[a], m[b], 0.0, 0.0, 0.0)
                for f in (disp, exch, src):
                    f.addExclusion(m[a], m[b])
    mp.setExtrapolationCoefficients([0.0, 0.5, 0.5])
    mp.setNonbondedMethod(ip.InteraxMultipoleForce.PME if periodic else ip.InteraxMultipoleForce.NoCutoff)
    mp.setCutoffDistance(cutoff)
    mp.setUseOwnKernel(True)
    for f in (disp, exch, src):
        if periodic:
            f.setNonbondedMethod(ip.InteraxPairForce.CutoffPeriodic)
            f.setCutoffDistance(cutoff)
            f.setUseSwitchingFunction(True)
            f.setSwitchingDistance(cutoff - 0.1)
    disp.setUseLongRangeCorrection(periodic)
    atm = ip.InteraxThreeBodyForce()
    atm.setScaleFactor(0.8)
    atm.setCentroidDampingRadius(0.25)
    atm.setUsesPeriodicBoundaryConditions(periodic)
    a_mol = O["adisp"] + 2 * H["adisp"]
    alphas, ws = [O["adisp"], H["adisp"], H["adisp"]], [O["w"], H["w"], H["w"]]
    c6 = sum(1.5 * alphas[i] * alphas[j] * ws[i] * ws[j] / (ws[i] + ws[j]) for i in range(3) for j in range(3))
    for m in molecules:
        atm.addSite([m[0]], a_mol, 4 * c6 / (3 * a_mol**2))
    for f in (mp, disp, exch, src, atm):
        system.addForce(f)
    return system


def energy_forces(system, positions, platform="Reference"):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform))
    c.setPositions(positions)
    s = c.getState(getEnergy=True, getForces=True)
    return s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_ffxml_equals_direct_construction(periodic, split):
    top, pos = water_box()
    ff = app.ForceField(*[io.StringIO(text) for text in ffxml(split=split)])
    method = app.PME if periodic else app.NoCutoff
    system = ff.createSystem(top, nonbondedMethod=method, nonbondedCutoff=0.9 * unit.nanometer, removeCMMotion=False)
    expected = direct_system(top, periodic)
    e1, f1 = energy_forces(system, pos)
    e2, f2 = energy_forces(expected, pos)
    assert abs(e1 - e2) < 1e-8 * max(1.0, abs(e2))
    assert np.abs(f1 - f2).max() < 1e-6 * max(1.0, np.abs(f2).max())


def test_inconsistent_settings_are_refused():
    texts = ffxml(split=True)
    texts[1] = texts[1].replace('extrapolationCoefficients="0.0,0.5,0.5"', 'extrapolationCoefficients="1.0"')
    with pytest.raises(ValueError, match="settings"):
        app.ForceField(*[io.StringIO(text) for text in texts])
