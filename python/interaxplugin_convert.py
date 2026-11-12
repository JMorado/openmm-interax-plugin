"""
Convert OpenMM Systems built by InteraxFF to the forces of the InteraxFF plugin.

InteraxFF (``interax.openmm.create_system_from_topology``) writes its dispersion, exchange repulsion and short-range
correction as CustomNonbondedForces, and its electrostatics and polarization as an InteraxMultipoleForce wrapped in a
CustomCVForce. :func:`convert_system` replaces the pair forces by InteraxDispersionForce, InteraxExchangeForce and
InteraxShortRangeForce, unwraps the InteraxMultipoleForce, and keeps every other force. :func:`convert_multipoles`
creates the InteraxMultipoleForce of a HippoNonbondedForce, which InteraxFF uses to build it.
"""

import math

import openmm

import interaxplugin as ip


def build_info():
    """
    Return the version of the plugin and the git commit it was built from.

    Returns
    -------
    str
        The version and the commit, for example ``"0.1.0 (v0.1.0)"``.
    """
    return f"{ip.__version__} ({ip.getPluginGitCommit()})"


def _per_particle_names(force):
    """Return the names of the per-particle parameters of a CustomNonbondedForce."""
    return [force.getPerParticleParameterName(k) for k in range(force.getNumPerParticleParameters())]


def _global_parameters(force):
    """Return the global parameters of a custom force as a dict of their default values."""
    return {force.getGlobalParameterName(k): force.getGlobalParameterDefaultValue(k) for k in range(force.getNumGlobalParameters())}


def _require_globals(force, names, term):
    """Raise a ValueError unless the global parameters of a force are exactly names: converting it would drop the others."""
    found = set(_global_parameters(force))
    if found != set(names):
        raise ValueError(f"The {term} force has the global parameters {sorted(found)}, not {sorted(names)}.")


def _classify(force):
    """Return the InteraxFF term of a CustomNonbondedForce: "dispersion", "exchange", "short_range" or None."""
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
    """Copy the cutoff, switching function, exclusions and force group of a CustomNonbondedForce to a plugin pair force."""
    methods = {
        openmm.CustomNonbondedForce.NoCutoff: ip.InteraxPairForce.NoCutoff,
        openmm.CustomNonbondedForce.CutoffNonPeriodic: ip.InteraxPairForce.CutoffNonPeriodic,
        openmm.CustomNonbondedForce.CutoffPeriodic: ip.InteraxPairForce.CutoffPeriodic,
    }
    if source.getNumInteractionGroups() > 0:
        raise ValueError("Interaction groups are not supported: use the alchemical flags of the plugin forces instead.")
    target.setNonbondedMethod(methods[source.getNonbondedMethod()])
    target.setCutoffDistance(source.getCutoffDistance())
    target.setUseSwitchingFunction(source.getUseSwitchingFunction())
    target.setSwitchingDistance(source.getSwitchingDistance())
    for k in range(source.getNumExclusions()):
        target.addExclusion(*source.getExclusionParticles(k))
    target.setForceGroup(source.getForceGroup())
    target.setName(type(target).__name__)


def _particle_values(force, index):
    """Return the per-particle parameters of a particle of a CustomNonbondedForce as a dict."""
    return dict(zip(_per_particle_names(force), force.getParticleParameters(index)))


def _overlap_method(force, plugin_class):
    """
    Return the overlap method that reproduces the Slater overlap of an exchange or short-range force of InteraxFF
    (``interax.openmm.overlap_energy_expression``): Exact for the order-6 expansion about the smaller width
    ("l = min(s1, s2)"), which it equals to rounding in double precision, and InteraxOrder2 for the order-2 expansion
    about s1 below an absolute width difference ("issame = step(threshold - abs(sdifflin))").
    """
    expression = force.getEnergyFunction()
    if "l = min(s1, s2)" in expression:
        return plugin_class.Exact
    threshold = expression.partition("issame = step(")[2].partition(" - abs(sdifflin)")[0]
    try:
        float(threshold)
    except ValueError:
        raise ValueError("The force has no Slater overlap scheme that this converter knows.") from None
    return plugin_class.InteraxOrder2


def convert_dispersion(force):
    """
    Create the InteraxDispersionForce of the dispersion of InteraxFF.

    Parameters
    ----------
    force : openmm.CustomNonbondedForce
        The Tang-Toennies damped C6/C8/C10 dispersion written by InteraxFF.

    Returns
    -------
    interaxplugin.InteraxDispersionForce
        A force with the same parameters, cutoff, switching function, long range correction, exclusions and force
        group.
    """
    _require_globals(force, {"tt_scale"}, "dispersion")
    if not all(f"E{n}" in force.getEnergyFunction() for n in (6, 8, 10)):
        raise ValueError("The dispersion must include the C6, C8 and C10 terms.")
    plugin = ip.InteraxDispersionForce()
    plugin.setDampingScale(_global_parameters(force)["tt_scale"])
    plugin.setUseLongRangeCorrection(force.getUseLongRangeCorrection())
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        plugin.addParticle(v["m1"], v["m2"], v["m3"], v["alpha"], v["s"])
    _copy_pair_settings(force, plugin)
    return plugin


def convert_exchange(force, overlap="exact"):
    """
    Create the InteraxExchangeForce of the exchange repulsion of InteraxFF.

    Parameters
    ----------
    force : openmm.CustomNonbondedForce
        The exchange repulsion written by InteraxFF, K_A A1 A2 S with the amplitudes A in atomic units.
    overlap : str, optional
        "exact" for the Exact overlap method, "export" for the method that reproduces the Slater overlap of the force
        (see :func:`convert_system`). Default "exact".

    Returns
    -------
    interaxplugin.InteraxExchangeForce
        A force with the amplitudes scaled by sqrt(K_A) to sqrt(kJ/mol nm^3), and the same widths, cutoff, switching
        function, exclusions and force group.
    """
    _require_globals(force, {"K_A", "K_h"}, "exchange")
    scale = math.sqrt(_global_parameters(force)["K_A"])
    plugin = ip.InteraxExchangeForce()
    plugin.setOverlapMethod(_overlap_method(force, ip.InteraxExchangeForce) if overlap == "export" else ip.InteraxExchangeForce.Exact)
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        plugin.addParticle(scale * v["A"], v["s"])
    _copy_pair_settings(force, plugin)
    return plugin


def convert_short_range(force, overlap="exact"):
    """
    Create the InteraxShortRangeForce of the short-range correction of InteraxFF.

    Parameters
    ----------
    force : openmm.CustomNonbondedForce
        The short-range correction written by InteraxFF.
    overlap : str, optional
        "exact" for the Exact overlap method, "export" for the method that reproduces the Slater overlap of the force
        (see :func:`convert_system`). Default "exact".

    Returns
    -------
    interaxplugin.InteraxShortRangeForce
        A force with the same parameters, cutoff, switching function, exclusions and force group.
    """
    _require_globals(force, {"K_h"}, "short-range correction")
    plugin = ip.InteraxShortRangeForce()
    plugin.setOverlapMethod(_overlap_method(force, ip.InteraxShortRangeForce) if overlap == "export" else ip.InteraxShortRangeForce.Exact)
    for i in range(force.getNumParticles()):
        v = _particle_values(force, i)
        plugin.addParticle(v["K"], v["L"], v["s"])
    _copy_pair_settings(force, plugin)
    return plugin


def convert_multipoles(hippo, core_damping=False, core_damping_scale=1.0):
    """
    Create an InteraxMultipoleForce with the electrostatics and polarization of a HippoNonbondedForce.

    The HippoNonbondedForce must have no repulsion, dispersion or charge transfer. The InteraxMultipoleForce has the
    same particles, frames, exceptions, extrapolation coefficients, PME settings and force group. Polarizabilities of
    1e-14 nm^3 or less, which InteraxFF gives to non-polarizable atoms because HIPPO needs them positive, become zero.

    Parameters
    ----------
    hippo : openmm.HippoNonbondedForce
        The force to convert.
    core_damping : bool, optional
        Whether the permanent field on each atom is damped by the atom's own Slater density. Default False, as in
        HIPPO.
    core_damping_scale : float, optional
        The scale eta of the exponents of the core damping. Default 1.

    Returns
    -------
    interaxplugin.InteraxMultipoleForce
        The new force.
    """

    def strip(value):
        """Return the value of a Quantity in OpenMM's units, or a plain number unchanged."""
        return value.value_in_unit(value.unit) if hasattr(value, "unit") else value

    plugin = ip.InteraxMultipoleForce()
    method = hippo.getNonbondedMethod()
    plugin.setNonbondedMethod(ip.InteraxMultipoleForce.PME if method == openmm.HippoNonbondedForce.PME else ip.InteraxMultipoleForce.NoCutoff)
    plugin.setCutoffDistance(strip(hippo.getCutoffDistance()))
    plugin.setEwaldErrorTolerance(hippo.getEwaldErrorTolerance())
    alpha, nx, ny, nz = hippo.getPMEParameters()
    plugin.setPMEParameters(strip(alpha), nx, ny, nz)
    plugin.setExtrapolationCoefficients(list(hippo.getExtrapolationCoefficients()))
    plugin.setCoreDamping(core_damping)
    plugin.setCoreDampingScale(core_damping_scale)
    for i in range(hippo.getNumParticles()):
        (charge, dipole, quadrupole, core, exponent, epsilon, _damping, c6, pauli_k, _pauli_q, _pauli_alpha, polarizability, axis, z, x,
         y) = hippo.getParticleParameters(i)
        if strip(epsilon) != 0 or strip(c6) != 0 or strip(pauli_k) != 0:
            raise ValueError(f"HIPPO particle {i} has charge transfer, dispersion or repulsion, which InteraxMultipoleForce does not have.")
        polarizability = strip(polarizability)
        plugin.addParticle(strip(charge), [float(v) for v in strip(dipole)], [float(v) for v in strip(quadrupole)], strip(core),
                           strip(exponent), 0.0 if polarizability <= 1e-14 else polarizability, axis, z, x, y)
    for k in range(hippo.getNumExceptions()):
        i, j, mm, dm, dd, _dispersion, _repulsion, _charge_transfer = hippo.getExceptionParameters(k)
        plugin.addException(i, j, mm, dm, dd)
    plugin.setForceGroup(hippo.getForceGroup())
    plugin.setName(type(plugin).__name__)
    return plugin


def unwrap_multipoles(system):
    """
    Put the InteraxMultipoleForces that CustomCVForces merely wrap directly into a System.

    Each CustomCVForce whose energy is an InteraxMultipoleForce alone (its only collective variable, with no global
    parameters) is replaced by a copy of that force, with its own GPU kernel (setUseOwnKernel(True)) and the force group
    of the wrapper, at the end of the force list. The force can then share the context with the other nonbonded
    forces, which saves the inner context of the wrapper. CustomCVForces that combine or scale the force are kept.

    Parameters
    ----------
    system : openmm.System
        The System, modified in place.

    Returns
    -------
    int
        The number of forces unwrapped.
    """
    return _unwrap_multipole_forces(system)


def _unwrap_multipole_forces(system):
    """Implement :func:`unwrap_multipoles`, which the argument of :func:`convert_system` of the same name hides."""
    found = []
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if (isinstance(force, openmm.CustomCVForce) and force.getNumCollectiveVariables() == 1 and force.getNumGlobalParameters() == 0
                and force.getNumEnergyParameterDerivatives() == 0
                and force.getEnergyFunction().strip() == force.getCollectiveVariableName(0)
                and ip.InteraxMultipoleForce.isinstance(force.getCollectiveVariable(0))):
            copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force.getCollectiveVariable(0)))
            typed = ip.InteraxMultipoleForce.cast(copy)  # a view of copy, which owns the force until addForce
            typed.setUseOwnKernel(True)
            typed.setForceGroup(force.getForceGroup())
            found.append((index, copy))
    for index, _ in reversed(found):
        system.removeForce(index)
    for _, copy in found:
        system.addForce(copy)
    return len(found)


def convert_system(system, copy=True, overlap="exact", unwrap_multipoles=True):
    """
    Convert a System built by InteraxFF to the plugin forces.

    The dispersion, exchange repulsion and short-range correction become InteraxDispersionForce, InteraxExchangeForce
    and InteraxShortRangeForce, named after their classes, in the force groups of the forces they replace and at the
    end of the force list. By default the InteraxMultipoleForce that InteraxFF wraps in a CustomCVForce is unwrapped
    (see unwrap_multipoles below). Every other force is kept.

    Parameters
    ----------
    system : openmm.System
        The System built by ``interax.openmm.create_system_from_topology``.
    copy : bool, optional
        Whether to convert a copy (the default) or the System itself.
    overlap : str, optional
        The Slater overlap of the exchange repulsion and the short-range correction. "exact" (the default) uses the
        Exact overlap method. "export" uses the method that reproduces the overlap of the System to rounding: Exact for
        the order-6 scheme of InteraxFF (in double precision), InteraxOrder2 for its order-2 scheme, which differs from
        the exact integral by up to ~1e-6 relative.
    unwrap_multipoles : bool, optional
        Whether to put the InteraxMultipoleForce that InteraxFF wraps in a CustomCVForce directly into the System, with
        its own GPU kernel (see :func:`unwrap_multipoles`), which gives the same energies at a lower cost. Default True.

    Returns
    -------
    openmm.System
        The converted System.
    """
    if overlap not in ("export", "exact"):
        raise ValueError(f"overlap must be 'export' or 'exact', got {overlap!r}")
    if copy:
        system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    replacements = {}
    for index, force in enumerate(system.getForces()):
        if isinstance(force, openmm.CustomNonbondedForce):
            kind = _classify(force)
            if kind == "dispersion":
                replacements[index] = convert_dispersion(force)
            elif kind == "exchange":
                replacements[index] = convert_exchange(force, overlap)
            elif kind == "short_range":
                replacements[index] = convert_short_range(force, overlap)
    for index in sorted(replacements, reverse=True):
        system.removeForce(index)
    for index in sorted(replacements):
        system.addForce(replacements[index])
    if unwrap_multipoles:
        _unwrap_multipole_forces(system)
    return system
