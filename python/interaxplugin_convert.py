"""
Convert an OpenMM System built by the InteraxFF export (interax.openmm.create_system_from_topology) to the plugin
forces: the CustomNonbondedForces of the dispersion, exchange repulsion and short-range correction become
InteraxDispersionForce, InteraxExchangeForce and InteraxShortRangeForce, and the molecular three-body
CustomCentroidBondForce becomes InteraxThreeBodyForce. Every other force (valence terms, HIPPO electrostatics and
polarization, intramolecular nonbonded terms) is kept as it is.

The converted forces reproduce the export to the precision of its Slater integrals (docs/SPEC.md).
"""

import math

import openmm

import interaxplugin as ip

# InteraxFF unit conversions (interax/openmm.py)
HARTREE_TO_KJ_MOL = 2625.4996394799
BOHR_TO_NM = 0.0529177210903
K_A = HARTREE_TO_KJ_MOL * BOHR_TO_NM**3


def _per_particle_names(force):
    return [force.getPerParticleParameterName(k) for k in range(force.getNumPerParticleParameters())]


def _global_parameters(force):
    return {force.getGlobalParameterName(k): force.getGlobalParameterDefaultValue(k) for k in range(force.getNumGlobalParameters())}


def _classify(force):
    """The InteraxFF term of an exported CustomNonbondedForce: 'dispersion', 'exchange', 'short_range' or None."""
    names = set(_per_particle_names(force))
    expression = force.getEnergyFunction()
    if {"m1", "m2", "m3", "alpha", "s"} <= names:
        return "dispersion"
    if {"K", "L", "s"} <= names and "St" in expression:
        return "short_range"
    if {"A", "s"} <= names and "K_A" in expression:
        return "exchange"
    return None


def _copy_pair_settings(source, target):
    """Copy the cutoff, switching function and exclusions of a CustomNonbondedForce to a plugin pair force."""
    methods = {
        openmm.CustomNonbondedForce.NoCutoff: ip.InteraxPairForce.NoCutoff,
        openmm.CustomNonbondedForce.CutoffNonPeriodic: ip.InteraxPairForce.CutoffNonPeriodic,
        openmm.CustomNonbondedForce.CutoffPeriodic: ip.InteraxPairForce.CutoffPeriodic,
    }
    target.setNonbondedMethod(methods[source.getNonbondedMethod()])
    target.setCutoffDistance(source.getCutoffDistance())
    target.setUseSwitchingFunction(source.getUseSwitchingFunction())
    target.setSwitchingDistance(source.getSwitchingDistance())
    for k in range(source.getNumExclusions()):
        target.addExclusion(*source.getExclusionParticles(k))
    if source.getNumInteractionGroups() > 0:
        raise ValueError("Interaction groups are not supported: use the alchemical flags of the plugin forces instead.")
    target.setForceGroup(source.getForceGroup())
    target.setName(source.getName())


def _particle_values(force, index):
    return dict(zip(_per_particle_names(force), force.getParticleParameters(index)))


def _is_active(values):
    """False for a lone-pair site, which the export keeps in the pair forces with lp_on = 0."""
    return values.get("lp_on", 1.0) != 0.0


def convert_dispersion(force):
    globals_ = _global_parameters(force)
    if "tt_scale" not in globals_:
        raise ValueError("Only the Tang-Toennies dispersion can be converted (the force has no tt_scale).")
    if not all(f"E{n}" in force.getEnergyFunction() for n in (6, 8, 10)):
        raise ValueError("The dispersion must include the C6, C8 and C10 terms.")
    plugin = ip.InteraxDispersionForce()
    plugin.setDampingScale(globals_["tt_scale"])
    plugin.setUseLongRangeCorrection(force.getUseLongRangeCorrection())
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        alpha = v["alpha"] if _is_active(v) else 0.0
        plugin.addParticle(v["m1"], v["m2"], v["m3"], alpha, v["s"])
    _copy_pair_settings(force, plugin)
    return plugin


def convert_exchange(force):
    globals_ = _global_parameters(force)
    if "p_ex" in globals_ and globals_["p_ex"] != 1.0:
        raise ValueError("Only the exchange repulsion with exponent 1 can be converted.")
    scale = math.sqrt(globals_.get("K_A", K_A))
    plugin = ip.InteraxExchangeForce()
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        width = v["s"] if _is_active(v) else 0.0
        plugin.addParticle(scale * v["A"], width)
    _copy_pair_settings(force, plugin)
    return plugin


def convert_short_range(force):
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        if v.get("M", 0.0) != 0.0:
            raise ValueError("The mid-range term M of the short-range correction is not supported.")
    for k in range(force.getNumTabulatedFunctions()):
        values = force.getTabulatedFunction(k).getFunctionParameters()[2]
        if any(value != 0.0 for value in values):
            raise ValueError("Element-pair amplitudes of the short-range correction are not supported.")
    plugin = ip.InteraxShortRangeForce()
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        width = v["s"] if _is_active(v) else 0.0
        plugin.addParticle(v["K"], v["L"], width)
    _copy_pair_settings(force, plugin)
    return plugin


def _molecules_from_exclusions(force, num_particles):
    """The molecules of a System: the connected components of the exclusion graph of an exported pair force."""
    parent = list(range(num_particles))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in range(force.getNumExclusions()):
        i, j = force.getExclusionParticles(k)
        parent[find(i)] = find(j)
    molecules = {}
    for i in range(num_particles):
        molecules.setdefault(find(i), []).append(i)
    return sorted(molecules.values())


def convert_three_body(force, dispersion):
    """
    Convert the molecular ATM CustomCentroidBondForce. The export stores only the C9 of every triple, so the
    molecular polarizabilities and frequencies are recomputed from the atomic dispersion parameters, exactly as the
    export computes them, and checked against the stored C9 values.
    """
    expression = force.getEnergyFunction()
    if force.getNumGroupsPerBond() != 3 or "C9" not in expression:
        raise ValueError("Not a molecular three-body dispersion force.")
    cutoff = float(expression.split("step(")[1].split("-")[0])

    # Atomic single-pole oscillators: alpha_disp and w = 4 C6 / (3 alpha^2), C6 = alpha m1 / 2 (homoatomic XDM).
    num_particles = dispersion.getNumParticles()
    alpha, frequency = [0.0] * num_particles, [0.0] * num_particles
    for i in range(num_particles):
        v = _particle_values(dispersion, i)
        if _is_active(v) and v["alpha"] > 0:
            c6 = 0.5 * v["alpha"] * v["m1"]
            alpha[i] = v["alpha"]
            frequency[i] = 4 * c6 / (3 * v["alpha"] ** 2)

    # Molecular oscillators (London combination of the atomic C6).
    molecules = _molecules_from_exclusions(dispersion, num_particles)
    if len(molecules) != force.getNumGroups():
        raise ValueError(f"Found {len(molecules)} molecules but the three-body force has {force.getNumGroups()} groups.")
    molecular = []
    for atoms in molecules:
        a = sum(alpha[i] for i in atoms)
        c6 = sum(1.5 * alpha[i] * alpha[j] * frequency[i] * frequency[j] / max(frequency[i] + frequency[j], 1e-30)
                 for i in atoms for j in atoms)
        molecular.append((a, 4 * c6 / (3 * a * a) if a > 0 else 0.0))

    plugin = ip.InteraxThreeBodyForce()
    plugin.setCutoffDistance(cutoff)
    plugin.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    for g in range(force.getNumGroups()):
        particles, weights = force.getGroupParameters(g)
        plugin.addMolecule(list(particles), *molecular[g])

    # The scale factor is the ratio of a stored C9 to the recomputed one; every triple must agree with it.
    scale = None
    for b in range(min(force.getNumBonds(), 1000)):
        groups, (c9,) = force.getBondParameters(b)
        (a1, w1), (a2, w2), (a3, w3) = (molecular[g] for g in groups)
        expected = 1.5 * a1 * a2 * a3 * w1 * w2 * w3 * (w1 + w2 + w3) / ((w1 + w2) * (w2 + w3) * (w3 + w1))
        if expected == 0.0:
            if c9 != 0.0:
                raise ValueError(f"The stored C9 of triple {tuple(groups)} is nonzero for molecules without polarizability.")
            continue
        ratio = c9 / expected
        scale = ratio if scale is None else scale
        if abs(ratio - scale) > 1e-6 * abs(scale):
            raise ValueError(f"The stored C9 of triple {tuple(groups)} does not match the molecular parameters.")
    plugin.setScaleFactor(1.0 if scale is None else scale)
    plugin.setForceGroup(force.getForceGroup())
    plugin.setName(force.getName())
    return plugin


def convert_system(system, copy=True):
    """
    Return a copy of an exported InteraxFF System in which the dispersion, exchange repulsion, short-range correction
    and molecular three-body dispersion use the plugin forces. Forces keep their group and name; exchange-type forces
    whose amplitudes are all zero (the inactive exchange-induction term) are removed. With copy=False the System is
    converted in place, which avoids serializing the millions of triples of the exported three-body force.
    """
    if copy:
        system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    forces = [system.getForce(i) for i in range(system.getNumForces())]
    dispersion = next((f for f in forces if isinstance(f, openmm.CustomNonbondedForce) and _classify(f) == "dispersion"), None)
    replacements = {}
    for index, force in enumerate(forces):
        if isinstance(force, openmm.CustomNonbondedForce):
            kind = _classify(force)
            if kind == "dispersion":
                replacements[index] = convert_dispersion(force)
            elif kind == "short_range":
                replacements[index] = convert_short_range(force)
            elif kind == "exchange":
                amplitudes = [_particle_values(force, i)["A"] for i in range(force.getNumParticles())]
                replacements[index] = convert_exchange(force) if any(a != 0 for a in amplitudes) else None
        elif isinstance(force, openmm.CustomCentroidBondForce) and "C9" in force.getEnergyFunction():
            if dispersion is None:
                raise ValueError("The three-body force can only be converted together with the dispersion force.")
            replacements[index] = convert_three_body(force, dispersion)
    for index in sorted(replacements, reverse=True):
        system.removeForce(index)
    # Add the converted forces at the end, in their original order (OpenMM has no insertForce).
    for index in sorted(replacements):
        if replacements[index] is not None:
            system.addForce(replacements[index])
    return system
