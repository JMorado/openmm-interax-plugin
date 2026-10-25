"""
Convert an OpenMM System built by the InteraxFF export (interax.openmm.create_system_from_topology) to the plugin
forces: the CustomNonbondedForces of the dispersion, exchange repulsion and short-range correction become
InteraxDispersionForce, InteraxExchangeForce and InteraxShortRangeForce, and the molecular three-body
CustomCentroidBondForce becomes InteraxThreeBodyForce. Every other force (valence terms, HIPPO electrostatics and
polarization, intramolecular nonbonded terms) is kept as it is.

The converted forces reproduce the export to rounding (docs/SPEC.md).
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
    target.setName(type(target).__name__)


def _particle_values(force, index):
    return dict(zip(_per_particle_names(force), force.getParticleParameters(index)))


def _overlap_method(force, plugin_class):
    """
    The plugin overlap method that reproduces an exported force: the InteraxFF Taylor scheme of its switch
    (interax.openmm._taylor_switch_expression), 'issame = step(0.2*s1 - abs(sdifflin))' for order 6 and an absolute
    width difference for order 2.
    """
    expression = force.getEnergyFunction()
    if "issame" not in expression:
        raise ValueError("The force has no Taylor switch of the Slater overlap.")
    switch = expression.split("issame = step(")[1].split("-")[0]
    return plugin_class.InteraxOrder6 if "*s1" in switch else plugin_class.InteraxOrder2


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


def convert_exchange(force, overlap="exact"):
    globals_ = _global_parameters(force)
    if "p_ex" in globals_ and globals_["p_ex"] != 1.0:
        raise ValueError("Only the exchange repulsion with exponent 1 can be converted.")
    scale = math.sqrt(globals_.get("K_A", K_A))
    plugin = ip.InteraxExchangeForce()
    plugin.setOverlapMethod(_overlap_method(force, ip.InteraxExchangeForce) if overlap == "export" else ip.InteraxExchangeForce.Exact)
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        width = v["s"] if _is_active(v) else 0.0
        plugin.addParticle(scale * v["A"], width)
    _copy_pair_settings(force, plugin)
    return plugin


def convert_short_range(force, overlap="exact"):
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        if v.get("M", 0.0) != 0.0:
            raise ValueError("The mid-range term M of the short-range correction is not supported.")
    for k in range(force.getNumTabulatedFunctions()):
        values = force.getTabulatedFunction(k).getFunctionParameters()[2]
        if any(value != 0.0 for value in values):
            raise ValueError("Element-pair amplitudes of the short-range correction are not supported.")
    plugin = ip.InteraxShortRangeForce()
    plugin.setOverlapMethod(_overlap_method(force, ip.InteraxShortRangeForce) if overlap == "export" else ip.InteraxShortRangeForce.Exact)
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


def _atomic_oscillators(dispersion):
    """
    The atomic single-pole oscillators of an exported dispersion force: alpha_disp and w = 4 C6 / (3 alpha^2), with
    the homoatomic XDM C6 = alpha m1 / 2. Particles without polarizability (lone-pair sites) have alpha = w = 0.
    """
    num_particles = dispersion.getNumParticles()
    alpha, frequency = [0.0] * num_particles, [0.0] * num_particles
    for i in range(num_particles):
        v = _particle_values(dispersion, i)
        if _is_active(v) and v["alpha"] > 0:
            c6 = 0.5 * v["alpha"] * v["m1"]
            alpha[i] = v["alpha"]
            frequency[i] = 4 * c6 / (3 * v["alpha"] ** 2)
    return alpha, frequency


def atomic_three_body(dispersion, cutoff=0.8, damping_scale=1.0, scale=1.0, periodic=None):
    """
    Create an atomic Axilrod-Teller-Muto force (InteraxThreeBodyForce with one site per atom and Tang-Toennies damping)
    from the dispersion CustomNonbondedForce of an exported InteraxFF System. Each atom with a dispersion polarizability is a site with the atomic alpha and
    frequency of the molecular ATM's oscillators and the dispersion Slater width; triples within one molecule (a
    connected component of the exclusions) are excluded.

    This is not a term of the trained InteraxFF models: it is provided to evaluate the cost and the effect of an
    atomic ATM, whose scale and damping would have to be fitted.
    """
    alpha, frequency = _atomic_oscillators(dispersion)
    molecules = _molecules_from_exclusions(dispersion, dispersion.getNumParticles())
    plugin = ip.InteraxThreeBodyForce()
    plugin.setCutoffDistance(cutoff)
    plugin.setScaleFactor(scale)
    plugin.setDampingMethod(ip.InteraxThreeBodyForce.TangToennies)
    plugin.setDampingScale(damping_scale)
    if periodic is None:
        periodic = dispersion.getNonbondedMethod() == openmm.CustomNonbondedForce.CutoffPeriodic
    plugin.setUsesPeriodicBoundaryConditions(periodic)
    for m, atoms in enumerate(molecules):
        for i in atoms:
            if alpha[i] > 0:
                plugin.addSite([i], alpha[i], frequency[i], _particle_values(dispersion, i)["s"], m)
    return plugin


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

    num_particles = dispersion.getNumParticles()
    alpha, frequency = _atomic_oscillators(dispersion)

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
        plugin.addSite(list(particles), *molecular[g])

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
    plugin.setName(type(plugin).__name__)
    return plugin


def wrapped_hippo(force):
    """The HippoNonbondedForce of the export (wrapped in a CustomCVForce "E_hippo"), or None."""
    if isinstance(force, openmm.HippoNonbondedForce):
        return force
    if isinstance(force, openmm.CustomCVForce):
        for k in range(force.getNumCollectiveVariables()):
            cv = force.getCollectiveVariable(k)
            if isinstance(cv, openmm.HippoNonbondedForce):
                return cv
    return None


def convert_multipoles(hippo, core_damping=False, mutual_damping_scale=1.0, alchemical_particles=(), alchemical_parameter="lambda_electrostatics",
                       core_damping_scale=1.0):
    """
    An InteraxMultipoleForce with the electrostatics and polarization of an exported HippoNonbondedForce (its
    repulsion, dispersion and charge transfer must be off, as in the export): same particles, frames, exceptions,
    extrapolation coefficients and PME settings. core_damping (with its range core_damping_scale) and mutual_damping_scale select the InteraxFF options
    that HIPPO lacks; the multipoles and polarizabilities of alchemical_particles are scaled by alchemical_parameter.
    The polarizabilities of 1e-14 nm^3 or less that the export gives to non-polarizable atoms (HIPPO needs them
    positive) become zero.
    """
    plugin = ip.InteraxMultipoleForce()
    method = hippo.getNonbondedMethod()
    plugin.setNonbondedMethod(ip.InteraxMultipoleForce.PME if method == openmm.HippoNonbondedForce.PME else ip.InteraxMultipoleForce.NoCutoff)
    plugin.setCutoffDistance(hippo.getCutoffDistance())
    plugin.setEwaldErrorTolerance(hippo.getEwaldErrorTolerance())
    alpha, nx, ny, nz = hippo.getPMEParameters()
    plugin.setPMEParameters(alpha.value_in_unit(alpha.unit) if hasattr(alpha, "unit") else alpha, nx, ny, nz)
    plugin.setExtrapolationCoefficients(list(hippo.getExtrapolationCoefficients()))
    plugin.setCoreDamping(core_damping)
    plugin.setCoreDampingScale(core_damping_scale)
    plugin.setMutualDampingScale(mutual_damping_scale)
    plugin.setAlchemicalParameterName(alchemical_parameter)
    alchemical = set(alchemical_particles)
    strip = lambda v: v.value_in_unit(v.unit) if hasattr(v, "unit") else v  # noqa: E731
    for i in range(hippo.getNumParticles()):
        (charge, dipole, quadrupole, core, exponent, epsilon, _damping, c6, pauli_k, _pauli_q, _pauli_alpha, polarizability,
         axis, z, x, y) = hippo.getParticleParameters(i)
        if strip(epsilon) != 0 or strip(c6) != 0 or strip(pauli_k) != 0:
            raise ValueError(f"HIPPO particle {i} has charge transfer, dispersion or repulsion, which InteraxMultipoleForce does not have.")
        polarizability = strip(polarizability)
        plugin.addParticle(strip(charge), [float(v) for v in strip(dipole)], [float(v) for v in strip(quadrupole)], strip(core),
                           strip(exponent), 0.0 if polarizability <= 1e-14 else polarizability, axis, z, x, y, i in alchemical)
    for k in range(hippo.getNumExceptions()):
        i, j, mm, dm, dd, _disp, _rep, _ct = hippo.getExceptionParameters(k)
        plugin.addException(i, j, mm, dm, dd)
    plugin.setForceGroup(hippo.getForceGroup())
    plugin.setName(type(plugin).__name__)
    return plugin


def convert_system(system, copy=True, overlap="exact"):
    """
    Return a copy of an exported InteraxFF System in which the dispersion, exchange repulsion, short-range correction
    and molecular three-body dispersion use the plugin forces, named after their classes. Forces keep their group;
    exchange-type forces whose amplitudes are all zero (the inactive exchange-induction term) are removed. With copy=False the System is
    converted in place, which avoids serializing the millions of triples of the exported three-body force.

    The Slater overlap of the exchange repulsion and the short-range correction uses, with overlap="export", the
    InteraxFF Taylor scheme of the export, evaluated as the export does (the converted forces reproduce it to
    rounding), and with overlap="exact" the plugin's accurate scheme (the exact integral, which the export
    approximates to ~1e-6 relative).
    """
    if overlap not in ("export", "exact"):
        raise ValueError(f"overlap must be 'export' or 'exact', got {overlap!r}")
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
                replacements[index] = convert_short_range(force, overlap)
            elif kind == "exchange":
                amplitudes = [_particle_values(force, i)["A"] for i in range(force.getNumParticles())]
                replacements[index] = convert_exchange(force, overlap) if any(a != 0 for a in amplitudes) else None
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
