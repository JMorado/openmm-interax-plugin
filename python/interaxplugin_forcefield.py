"""
OpenMM ForceField (ffxml) support for the InteraxFF plugin forces: build Systems from scratch with
openmm.app.ForceField.createSystem(), as for any other OpenMM force field, instead of converting an exported System.

Importing this module registers generators for these ffxml tags (in openmm.app.forcefield.parsers):

    <InteraxMultipoleForce>    <Atom type charge coreCharge alpha polarizability axisType d0 d1 d2 q11 ... q33
                                     [zAtomType xAtomType yAtomType]/>
    <InteraxDispersionForce>   <Atom type m1 m2 m3 alpha width/>
    <InteraxExchangeForce>     <Atom type amplitude width/>
    <InteraxShortRangeForce>   <Atom type k l width/>
    <InteraxThreeBodyForce>    <Atom type alpha frequency/>     (one site per molecule, London combination)

Parameters are per atom type, in OpenMM units, with the meanings of the corresponding addParticle() methods. The
force-level settings are attributes of the tags (below). A tag may appear in several files (for instance one file per
molecule, written by a residue template generator): the parameters are merged into one generator, whose settings must
agree.

InteraxFF interactions are intermolecular. Every force excludes all pairs within a molecule (a connected component of
the bonds, virtual sites included). In InteraxMultipoleForce those pairs are exceptions with zero multipole and
dipole-multipole scales and the dipole-dipole scale intramolecularDipoleDipoleScale (0, or 1 for intramolecular
mutual polarization), zero for bonded pairs and for virtual sites.

The local frame of a multipole is given, as in OpenMM's AMOEBA and HIPPO force fields, by the atom types of its z, x
and y atoms (zAtomType, xAtomType, yAtomType). They are looked up in the particle's molecule: among its bonded atoms
first, then the atoms two bonds away, then the whole molecule, where the type must be unique. The last case allows
frames whose axis atom is not a neighbour (the bond multipole sites of InteraxFF, which sit on their host atom).

Force-level attributes (defaults in brackets):

    InteraxMultipoleForce   extrapolationCoefficients [1.0]; coreDamping [false]; coreDampingScale [1];
                            mutualDampingScale [1]; chiralityMode ("none" | "amoeba") [amoeba];
                            intramolecularDipoleDipoleScale [0]; useOwnKernel [true]
    InteraxDispersionForce  dampingScale [1]; useLongRangeCorrection [true, periodic only]
    InteraxExchangeForce,
    InteraxShortRangeForce  overlapMethod ("exact" | "order2" | "order6") [exact]
    InteraxThreeBodyForce   scaleFactor [1]; cutoff (nm) [1.2]; centroidDampingRadius (nm) [0]

The pair forces use the createSystem() cutoff with periodic boxes (switching from switchDistance, else from
max(cutoff - 0.1, cutoff / 2) nm) and no cutoff otherwise; InteraxMultipoleForce uses PME or NoCutoff.
"""

import openmm
from openmm.app import forcefield as _ff

import interaxplugin as ip

_TRUE = ("true", "1", "yes")


def _bool(value):
    return str(value).lower() in _TRUE


def _settings(element, defaults):
    """The force-level attributes of an element, with defaults, as strings."""
    return {key: element.attrib.get(key, default) for key, default in defaults.items()}


def _get_generator(cls, ff, element, defaults):
    """The generator of this class already registered with ff (its settings must match), or a new one."""
    settings = _settings(element, defaults)
    for generator in ff.getGenerators():
        if isinstance(generator, cls):
            if generator.settings != settings:
                raise ValueError(f"{cls.tag}: settings {settings} differ from those of an earlier <{cls.tag}> ({generator.settings})")
            return generator
    generator = cls(ff, settings)
    ff.registerGenerator(generator)
    return generator


def _molecules(sys, data):
    """(molecule index of every particle, list of the particles of every molecule): connected components of the bonds."""
    bonds = list(_ff._findBondsForExclusions(data, sys))
    # A virtual site belongs to the molecule of its parent (the atom it shares exclusions with, else its first atom)
    for atom, (site, atoms, exclude_with) in data.virtualSites.items():
        bonds.append((atom.index, exclude_with if exclude_with is not None else atoms[0]))
    bonded = [[] for _ in data.atoms]
    for i, j in bonds:
        bonded[i].append(j)
        bonded[j].append(i)
    group = _ff._findGroups(bonded)
    molecules = [[] for _ in range(max(group) + 1 if group else 0)]
    for i, g in enumerate(group):
        molecules[g].append(i)
    return group, molecules


def _is_site(data, atom):
    return atom.element is None or atom in data.virtualSites


def _periodic(nonbondedMethod):
    return nonbondedMethod in (_ff.PME, _ff.CutoffPeriodic, _ff.LJPME, _ff.Ewald)


def _setup_pair_force(force, nonbondedMethod, nonbondedCutoff, args, molecules):
    """Cutoff, switching and the exclusion of every intramolecular pair, as the InteraxFF export sets them."""
    cutoff = nonbondedCutoff.value_in_unit(openmm.unit.nanometer) if hasattr(nonbondedCutoff, "value_in_unit") else float(nonbondedCutoff)
    if _periodic(nonbondedMethod):
        force.setNonbondedMethod(ip.InteraxPairForce.CutoffPeriodic)
        force.setCutoffDistance(cutoff)
        switch = args.get("switchDistance", None)
        if switch is None:
            switch = max(cutoff - 0.1, 0.5 * cutoff)
        elif hasattr(switch, "value_in_unit"):
            switch = switch.value_in_unit(openmm.unit.nanometer)
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(switch)
    elif nonbondedMethod == _ff.NoCutoff:
        force.setNonbondedMethod(ip.InteraxPairForce.NoCutoff)
    else:
        raise ValueError(f"The InteraxFF pair forces support NoCutoff and periodic methods, not {nonbondedMethod}")
    for particles in molecules:
        for a in range(len(particles)):
            for b in range(a + 1, len(particles)):
                force.addExclusion(particles[a], particles[b])


class _PairGenerator:
    tag = None
    paramNames = ()
    defaults = {}

    def __init__(self, ff, settings):
        self.ff = ff
        self.settings = settings
        self.params = _ff.ForceField._AtomTypeParameters(ff, self.tag, "Atom", self.paramNames)

    @classmethod
    def parseElement(cls, element, ff):
        generator = _get_generator(cls, ff, element, cls.defaults)
        generator.params.parseDefinitions(element)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        force = self.makeForce()
        for atom in data.atoms:
            self.addParticle(force, [float(v) for v in self.params.getAtomParameters(atom, data)])
        _, molecules = _molecules(sys, data)
        _setup_pair_force(force, nonbondedMethod, nonbondedCutoff, args, molecules)
        self.finish(force, nonbondedMethod)
        force.setName(type(force).__name__)
        sys.addForce(force)

    def finish(self, force, nonbondedMethod):
        pass


_OVERLAP = {"exact": "Exact", "order2": "InteraxOrder2", "order6": "InteraxOrder6"}


class InteraxDispersionGenerator(_PairGenerator):
    """Constructs an InteraxDispersionForce (Tang-Toennies damped C6, C8, C10 dispersion)."""

    tag = "InteraxDispersionForce"
    paramNames = ("m1", "m2", "m3", "alpha", "width")
    defaults = {"dampingScale": "1", "useLongRangeCorrection": "true"}

    def makeForce(self):
        force = ip.InteraxDispersionForce()
        force.setDampingScale(float(self.settings["dampingScale"]))
        return force

    def addParticle(self, force, p):
        force.addParticle(*p)

    def finish(self, force, nonbondedMethod):
        force.setUseLongRangeCorrection(_periodic(nonbondedMethod) and _bool(self.settings["useLongRangeCorrection"]))


class InteraxExchangeGenerator(_PairGenerator):
    """Constructs an InteraxExchangeForce (Slater overlap exchange repulsion)."""

    tag = "InteraxExchangeForce"
    paramNames = ("amplitude", "width")
    defaults = {"overlapMethod": "exact"}

    def makeForce(self):
        force = ip.InteraxExchangeForce()
        force.setOverlapMethod(getattr(ip.InteraxExchangeForce, _OVERLAP[self.settings["overlapMethod"]]))
        return force

    def addParticle(self, force, p):
        force.addParticle(*p)


class InteraxShortRangeGenerator(_PairGenerator):
    """Constructs an InteraxShortRangeForce (short-range correction)."""

    tag = "InteraxShortRangeForce"
    paramNames = ("k", "l", "width")
    defaults = {"overlapMethod": "exact"}

    def makeForce(self):
        force = ip.InteraxShortRangeForce()
        force.setOverlapMethod(getattr(ip.InteraxShortRangeForce, _OVERLAP[self.settings["overlapMethod"]]))
        return force

    def addParticle(self, force, p):
        force.addParticle(*p)


class InteraxThreeBodyGenerator:
    """
    Constructs an InteraxThreeBodyForce with one site per molecule: the centroid of its heavy atoms (of all its atoms
    if it has none), with the molecular polarizability sum_i alpha_i and the frequency of the London combination of
    the atomic C6, w = 4 C6_mol / (3 alpha_mol^2), C6_mol = sum_ij 3/2 alpha_i alpha_j w_i w_j / (w_i + w_j).
    """

    tag = "InteraxThreeBodyForce"
    paramNames = ("alpha", "frequency")
    defaults = {"scaleFactor": "1", "cutoff": "1.2", "centroidDampingRadius": "0"}

    def __init__(self, ff, settings):
        self.ff = ff
        self.settings = settings
        self.params = _ff.ForceField._AtomTypeParameters(ff, self.tag, "Atom", self.paramNames)

    @classmethod
    def parseElement(cls, element, ff):
        generator = _get_generator(cls, ff, element, cls.defaults)
        generator.params.parseDefinitions(element)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        alpha, frequency = [], []
        for atom in data.atoms:
            a, w = (float(v) for v in self.params.getAtomParameters(atom, data))
            alpha.append(a)
            frequency.append(w)
        _, molecules = _molecules(sys, data)
        force = ip.InteraxThreeBodyForce()
        force.setCutoffDistance(float(self.settings["cutoff"]))
        force.setUsesPeriodicBoundaryConditions(_periodic(nonbondedMethod))
        force.setScaleFactor(float(self.settings["scaleFactor"]))
        r0 = float(self.settings["centroidDampingRadius"])
        if r0 > 0:
            force.setCentroidDampingRadius(r0)
        for particles in molecules:
            a_mol = sum(alpha[i] for i in particles)
            c6 = sum(1.5 * alpha[i] * alpha[j] * frequency[i] * frequency[j] / max(frequency[i] + frequency[j], 1e-30)
                     for i in particles for j in particles)
            w_mol = 4 * c6 / (3 * a_mol * a_mol) if a_mol > 0 else 0.0
            real = [i for i in particles if not _is_site(data, data.atoms[i])]
            heavy = [i for i in real if sys.getParticleMass(i).value_in_unit(openmm.unit.dalton) > 1.5]
            force.addSite(heavy or real, a_mol, w_mol)
        force.setName(type(force).__name__)
        sys.addForce(force)


class InteraxMultipoleGenerator:
    """Constructs an InteraxMultipoleForce (multipole electrostatics and polarization)."""

    tag = "InteraxMultipoleForce"
    paramNames = ("charge", "coreCharge", "alpha", "polarizability", "axisType", "d0", "d1", "d2",
                  "q11", "q12", "q13", "q21", "q22", "q23", "q31", "q32", "q33")
    defaults = {"extrapolationCoefficients": "1.0", "coreDamping": "false", "coreDampingScale": "1",
                "mutualDampingScale": "1", "chiralityMode": "amoeba", "intramolecularDipoleDipoleScale": "0",
                "useOwnKernel": "true"}

    def __init__(self, ff, settings):
        self.ff = ff
        self.settings = settings
        self.params = _ff.ForceField._AtomTypeParameters(ff, self.tag, "Atom", self.paramNames)

    @classmethod
    def parseElement(cls, element, ff):
        generator = _get_generator(cls, ff, element, cls.defaults)
        generator.params.parseDefinitions(element)

    def _axis_atom(self, name, extra, atom, data, group, molecules, bonded12, bonded13, exclude):
        if name not in extra:
            return -1
        wanted = extra[name]
        for candidates in (bonded12[atom.index], bonded13[atom.index]):
            for j in sorted(candidates):
                if data.atomType[data.atoms[j]] == wanted and j not in exclude:
                    return j
        matches = [j for j in molecules[group[atom.index]] if data.atomType[data.atoms[j]] == wanted and j not in exclude and j != atom.index]
        if len(matches) != 1:
            raise ValueError(f"{self.tag}: {'no' if not matches else 'more than one'} atom of type {wanted} in the molecule "
                             f"of atom {atom.index} ({name})")
        return matches[0]

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        methods = {_ff.NoCutoff: ip.InteraxMultipoleForce.NoCutoff, _ff.PME: ip.InteraxMultipoleForce.PME}
        if nonbondedMethod not in methods:
            raise ValueError("InteraxMultipoleForce supports NoCutoff and PME")
        group, molecules = _molecules(sys, data)
        n = len(data.atoms)
        bonded12 = [set() for _ in range(n)]
        for bond in data.bonds:
            bonded12[bond.atom1].add(bond.atom2)
            bonded12[bond.atom2].add(bond.atom1)
        bonded13 = [set().union(*(bonded12[j] for j in bonded12[i])) - bonded12[i] - {i} for i in range(n)]

        force = ip.InteraxMultipoleForce()
        for atom in data.atoms:
            v = [float(x) for x in self.params.getAtomParameters(atom, data)]
            charge, core, alpha, polarizability, axis = v[0], v[1], v[2], v[3], int(v[4])
            dipole, quadrupole = v[5:8], v[8:17]
            extra = self.params.getExtraParameters(atom, data)
            args_ = (atom, data, group, molecules, bonded12, bonded13)
            z = self._axis_atom("zAtomType", extra, *args_, [])
            x = self._axis_atom("xAtomType", extra, *args_, [z])
            y = self._axis_atom("yAtomType", extra, *args_, [z, x])
            force.addParticle(charge, dipole, quadrupole, core, alpha, polarizability, axis, z, x, y)
        intra = float(self.settings["intramolecularDipoleDipoleScale"])
        for particles in molecules:
            for a in range(len(particles)):
                for b in range(a + 1, len(particles)):
                    i, j = particles[a], particles[b]
                    site = _is_site(data, data.atoms[i]) or _is_site(data, data.atoms[j])
                    dd = 0.0 if (site or j in bonded12[i]) else intra
                    force.addException(i, j, 0.0, 0.0, dd)
        force.setNonbondedMethod(methods[nonbondedMethod])
        cutoff = nonbondedCutoff.value_in_unit(openmm.unit.nanometer) if hasattr(nonbondedCutoff, "value_in_unit") else float(nonbondedCutoff)
        force.setCutoffDistance(cutoff)
        if "ewaldErrorTolerance" in args:
            force.setEwaldErrorTolerance(args["ewaldErrorTolerance"])
        force.setExtrapolationCoefficients([float(c) for c in self.settings["extrapolationCoefficients"].split(",")])
        force.setCoreDamping(_bool(self.settings["coreDamping"]))
        force.setCoreDampingScale(float(self.settings["coreDampingScale"]))
        force.setMutualDampingScale(float(self.settings["mutualDampingScale"]))
        mode = self.settings["chiralityMode"].lower()
        if mode not in ("none", "amoeba"):
            raise ValueError(f"{self.tag}: chiralityMode must be 'none' or 'amoeba', not {mode!r}")
        force.setChiralityMode(ip.InteraxMultipoleForce.NoChirality if mode == "none" else ip.InteraxMultipoleForce.AmoebaChirality)
        force.setName(type(force).__name__)
        if _bool(self.settings["useOwnKernel"]):
            # Its own GPU kernel: it can share the context with the other nonbonded forces (same exclusions, same cutoff)
            force.setUseOwnKernel(True)
            sys.addForce(force)
        else:
            # Like HIPPO it then replaces the default nonbonded kernel on the GPU: evaluate it in its own context
            wrapper = openmm.CustomCVForce("E_hippo")
            wrapper.addCollectiveVariable("E_hippo", force)
            sys.addForce(wrapper)


for _generator in (InteraxMultipoleGenerator, InteraxDispersionGenerator, InteraxExchangeGenerator, InteraxShortRangeGenerator,
                   InteraxThreeBodyGenerator):
    _ff.parsers[_generator.tag] = _generator.parseElement
