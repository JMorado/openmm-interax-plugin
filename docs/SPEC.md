# The forces of the InteraxFF plugin

This document gives the energy expressions, parameters and conventions of the five forces of the plugin. Lengths are
in nm, energies in kJ/mol and charges in e, as everywhere in OpenMM.

| Force | Energy |
|---|---|
| `InteraxMultipoleForce` | electrostatics of core charges, Slater valence charges, dipoles and quadrupoles, and polarization |
| `InteraxDispersionForce` | Tang-Toennies damped C6/C8/C10 pair dispersion from XDM moments |
| `InteraxExchangeForce` | exchange repulsion, proportional to the overlap of the Slater densities of each pair |
| `InteraxShortRangeForce` | short-range correction, linear and quadratic in the normalized Slater overlap |
| `InteraxThreeBodyForce` | Axilrod-Teller-Muto three-body dispersion between atomic or molecular sites |

The parameters are those of the OpenMM export of InteraxFF (`interax.openmm.create_system_from_topology`, "the export"
below), which `interaxplugin_convert.convert_system` converts to these forces. Systems can also be built from force
field files with the generators of `interaxplugin_forcefield`.

## Common conventions

* **Exclusions.** The pair forces take explicit exclusions (`addExclusion()`), as `CustomNonbondedForce` does.
  InteraxFF excludes every intramolecular pair. On the CUDA platform the pair forces of a System are evaluated in one
  nonbonded kernel, which requires them to have identical exclusions.
* **Cutoff.** `NoCutoff`, `CutoffNonPeriodic` or `CutoffPeriodic` (minimum image), with an optional switching function
  `S(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5`, `x = (r - r_switch)/(r_cutoff - r_switch)`, applied to the pair energy as in
  `CustomNonbondedForce`.
* **Alchemical particles.** Each particle (each site for the three-body force) can be flagged alchemical. The energy
  of a pair with exactly one alchemical particle is multiplied by the context parameter `lambda_sterics` (the name is
  configurable); pairs with no or two alchemical particles are not scaled. A three-body triple is scaled if it contains
  at least one alchemical site. The dispersion long range correction is the exception: every pair that involves an
  alchemical particle enters it scaled by lambda, the pairs within the alchemical molecule included, so that it is that
  of the whole System at lambda = 1 and that of the rest at lambda = 0.
* **Soft core.** With a soft-core radius `R > 0` (`setSoftcoreRadius()`), a pair with exactly one alchemical particle
  is evaluated at `r_eff = (r^6 + (1 - lambda) R^6)^(1/6)`.
* **Particles without density.** A particle with zero width does not interact through the exchange and short-range
  forces, and a particle with zero polarizability not through the dispersion.

## Slater overlap integral

The densities are normalized 1s Slater functions `rho(r) = exp(-r/s) / (8 pi s^3)` of width `s`. The overlap integral
(nm^-3) of two densities of widths `a` and `b` at distance `r` is

    S(r; a, b) = F(r) / (8 pi r)
    F(r) = 4 a^2 b^2 / D^3 (e_a - e_b) + r / D^2 (a e_a + b e_b),     D = b^2 - a^2,  e_a = exp(-r/a),  e_b = exp(-r/b)

and the normalized overlap is `St(r) = 8 pi (a + b)^3 S(r; a, b)`, with `St(0) = 1`. The two terms of F cancel to
relative order (b - a)^2, so for nearly equal widths the integral is evaluated from its Taylor expansion about a width
`l`, with `t = (b - l)/l` (`b` the other width) and `x = r/l` (coefficients `interax.geometry.OVERLAP_TAYLOR_COEFFS`):

    S = exp(-x) / (pi l^3) sum_{n=0}^{N} t^n P_n(x) / D_n

`InteraxExchangeForce` and `InteraxShortRangeForce` select the scheme with `setOverlapMethod()`:

| Method | Order N | l | Taylor expansion where | Accuracy |
|---|---|---|---|---|
| `Exact` (default) | 6 | the smaller width | t < 0.2 and t x < 1.6 in single precision, t < 0.02 and t x < 0.3 in double precision | ~1e-6 relative in single precision, ~1e-10 in double precision |
| `InteraxOrder2` | 2 | the width of the particle with the lower index | \|b - a\| < 0.01 bohr | the order-2 scheme of InteraxFF (MEDFF), with its truncation errors (~1e-6) |

InteraxFF has the same two schemes (`interax.geometry.SLATER_TAYLOR_SCHEMES`). Its "order6" scheme is the `Exact`
method: the export writes it symmetric in the two particles of a pair, with the double precision switch, and `Exact`
reproduces it to rounding in double precision. Its "order2" scheme expands about particle 1 of each pair, which on the
Reference platform is the particle with the lower index; `InteraxOrder2` evaluates it in the same way (the CUDA
kernels compare the original indices of the particles, which the context may reorder). In single precision the closed
form has relative errors up to ~3e-2 just outside the order-2 switch.

## InteraxMultipoleForce

Per particle: the total charge `q` and the core charge `Z` (e; the valence charge is `q - Z`), the dipole (e nm) and
traceless quadrupole (e nm^2) in the local frame defined by the axis type and the z, x and y particles, the exponent
`alpha` of the Slater valence density (nm^-1), the polarizability (nm^3), and the alchemical flag. Pairs of particles
can be exceptions, with scale factors for the multipole-multipole, dipole-multipole and dipole-dipole interactions.

The permanent electrostatics and the polarization are those of `HippoNonbondedForce` (Rackers et al.) without its
repulsion, dispersion and charge transfer: the valence charges have charge penetration damping, the induced dipoles
are computed with the extrapolated perturbation series (`setExtrapolationCoefficients()`; a single coefficient of 1
gives direct polarization), and the polarization energy is `-1/2 sum_i E_i . mu_i`, with `E_i` the permanent field on
particle i. The options:

* **Core damping** (`setCoreDamping()`). The permanent field on a particle is multiplied by
  `f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r)`, the fraction of the particle's own Slater density within `r`, with
  `a = eta alpha` (`setCoreDampingScale()`, `eta = 1` by default). Without it the polarization energy diverges as
  `-polarizability Z^2 / (2 r^4)` when atoms overlap.
* **Mutual damping scale** (`setMutualDampingScale()`). The couplings between induced dipoles are damped with the
  exponents `kappa alpha` (`kappa = 1`, as in HIPPO, by default).
* **Alchemical particles.** The charges, multipoles and polarizabilities of the alchemical particles are multiplied by
  the context parameter `lambda_electrostatics` (the name is configurable).
* **Chirality** (`setChiralityMode()`). With `AmoebaChirality` (the default) the y components of the dipole and the
  xy and yz components of the quadrupole of a ZThenX particle with a y atom change sign when the particle lies on the
  negative side of the plane of its z, x and y atoms; `NoChirality` ignores the y atom.
* **Own kernel** (`setUseOwnKernel()`). On the CUDA platform the real-space interactions are evaluated by a kernel of
  the force instead of the default nonbonded kernel, so that the force can share a Context with the pair forces.

Nearly equal exponents are handled without the cancellation of HIPPO's damping formulas, and zero polarizabilities are
allowed. Particles at the same position are allowed when they are fully excluded from each other (all scale factors
zero), with and without PME; InteraxFF's bond multipole sites sit on their host atoms in this way.

## InteraxDispersionForce

Per particle: the XDM moments `m1`, `m2`, `m3` (`<M_k^2>`, kJ/mol nm^3, nm^5 and nm^7), the dispersion polarizability
`alpha` (nm^3) and the Slater width `s` (nm). Global: the damping scale `beta`.

    E = - sum_{pairs} sum_{n = 6, 8, 10} f_n(x) C_n / r^n,        x = 2 beta r / (s_i + s_j)
    C6  = alpha_i alpha_j m1_i m1_j / D
    C8  = 3/2 alpha_i alpha_j (m1_i m2_j + m2_i m1_j) / D
    C10 = [2 alpha_i alpha_j (m1_i m3_j + m3_i m1_j) + 21/5 alpha_i alpha_j m2_i m2_j] / D,     D = alpha_i m1_j + alpha_j m1_i
    f_n(x) = 1 - exp(-x) sum_{k=0}^{n} x^k / k!

For `x < (n + 1)/2`, where the closed form cancels, `f_n` is evaluated as `exp(-x) sum_{k > n} x^k / k!`.

With periodic boundary conditions the optional long range correction (`setUseLongRangeCorrection()`) is
`E_lrc = (2 pi N^2 / V) <I>`, with `<I>` the average over all `N (N + 1)/2` pairs of particles, self pairs included, of

    I_ij = int_{r_c}^{inf} E_ij(r) r^2 dr + int_{r_s}^{r_c} (1 - S(r)) E_ij(r) r^2 dr

the convention of OpenMM's `CustomNonbondedForce`.

## InteraxExchangeForce

Per particle: the amplitude `A` (sqrt(kJ/mol nm^3)) and the exchange width `s` (nm).

    E = sum_{pairs} A_i A_j S(r; s_i, s_j)

InteraxFF writes `K_A A_1 A_2 S` with the amplitudes in atomic units, so `A = sqrt(K_A) A_InteraxFF`.

## InteraxShortRangeForce

Per particle: the amplitudes `K` and `L` (kJ/mol) and the width `s` (nm).

    E = sum_{pairs} [ (K_i + K_j)/2 St + (L_i + L_j)/2 St^2 ],     St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j)

## InteraxThreeBodyForce

Per site: the particles whose unweighted centroid is the site, the polarizability `alpha` (nm^3), the single-pole
frequency `w` (kJ/mol; the C6 coefficient of the site is `3/4 alpha^2 w`), the width `s` (nm, for the damping), the
molecule of the site and the alchemical flag. Global: the cutoff `r_c` (nm), the scale factor, the damping method and
scale `beta`, and the centroid damping radius `R0` (nm).

    E = scale sum_{A < B < C} C9_ABC (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC r_BC)^3 f_AB f_AC f_BC
    C9_ABC = 3/2 alpha_A alpha_B alpha_C w_A w_B w_C (w_A + w_B + w_C) / ((w_A + w_B)(w_B + w_C)(w_C + w_A))

The sum runs over the triples whose three distances are shorter than the cutoff and whose sites belong to three
different molecules. InteraxFF uses two forms:

* **Atomic**: one site per atom, with its atomic oscillator and the dispersion width, and the Tang-Toennies damping
  (`TangToennies`) of each of the three dipole propagators of a triple, `f_ij = f_3(2 beta r_ij / (s_i + s_j))`.
* **Molecular**: one site per molecule, the centroid of its heavy atoms, with the molecular polarizability and
  frequency (the London combination of the atomic oscillators), without damping (`NoDamping`).

With `R0 > 0` (`setCentroidDampingRadius()`) each `f_ij` is also multiplied by `g(r_ij) = 1 - exp(-(r_ij/R0)^6)`,
which keeps the molecular energy finite when two centroids come close. `R0 = 0` (the default) disables it.

With periodic boundary conditions the angle at each vertex is the angle between the minimum-image vectors to the two
other sites. With `a = x_B - x_A`, `b = x_C - x_A`, `c = x_C - x_B` (each the minimum image), `u, v, w` their squared
lengths, `P = u v w` and `p = a.b`, `q = a.c`, `s = b.c`, the product of the three cosines is `-pqs/P`, and without
damping

    E0 = C9 [P^(-3/2) - 3 pqs P^(-5/2)]

In boxes smaller than three cutoffs the three minimum images need not close into a triangle; this definition is then
the one of the export. The force on a site is shared equally among its particles.

Triples are found from neighbor lists of the sites. On the GPU each site keeps a list of the sites above it within the
cutoff plus a skin of 0.1 nm, rebuilt when a site has moved by more than half the skin; a thread block per site loops
over the pairs of tiles of its neighbors, skipping those whose bounding boxes are more than the cutoff apart, and
computes each triple once. The number of triples is about `N n^2 p / 6`, with `n` the number of sites within the cutoff
of a site and `p ~ 0.47`, so at a given cutoff an atomic force costs (sites per molecule)^3 times a molecular one.

## Tests

`tests/data` holds the energies and forces of each pair and three-body term computed by the export for two systems, a
cluster and a periodic box of methanol (`tests/python/generate_export_values.py`, model `models_v3R4bfdE1`, order-2
Slater scheme). `tests/TestInteraxExportValues.h` (every platform) and `tests/python/test_export_values.py` check the
plugin against them: in double precision `InteraxOrder2` reproduces them to rounding, and `Exact` agrees to ~1e-6.
`tests/TestInteraxMultipoleForce.h` tests the multipole force against `HippoNonbondedForce` and against the
definitions of its options.
