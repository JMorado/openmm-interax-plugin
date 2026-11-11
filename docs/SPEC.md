# The forces of the InteraxFF plugin

This plugin implements the intermolecular terms of the InteraxFF force field as five OpenMM forces, each with
Reference, CPU and CUDA kernels. This document gives their energy expressions, parameters and conventions.

| Force | Energy |
|---|---|
| `InteraxMultipoleForce` | HIPPO-style multipoles and polarization, with exact exponents, core damping, a mutual damping scale, a chirality mode and alchemical scaling |
| `InteraxDispersionForce` | damped C6/C8/C10 pair dispersion (XDM moments, Tang-Toennies damping) |
| `InteraxExchangeForce` | Slater-overlap exchange repulsion |
| `InteraxShortRangeForce` | short-range correction in the normalized Slater overlap |
| `InteraxThreeBodyForce` | Axilrod-Teller-Muto three-body dispersion between atomic or molecular sites |

Units are OpenMM's: nm, kJ/mol, e. The parameters are those of the OpenMM export of InteraxFF
(`interax.openmm.create_system_from_topology`, "the export" below), whose pair terms `interaxplugin_convert.convert_system` converts force
by force to the same energies and forces. Systems can also be built from force field files: `interaxplugin_forcefield`
registers OpenMM `ForceField` generators for the five forces (ffxml tags named after them; parameters per atom type,
frames by the atom types of the z, x, y atoms, every intramolecular pair excluded). Every build records its version
and git commit (`getPluginVersion()`, `getPluginGitCommit()`, `interaxplugin.__version__`).

InteraxMultipoleForce applies the chirality check of ZThenX frames with a y atom on every platform at every evaluation
(`AmoebaChirality`, the default) or never (`NoChirality`). Particles at the same position are allowed when they are
fully excluded from each other (all scale factors zero), with and without PME (the r = 0 limits of the excluded-pair
Ewald terms); InteraxFF's bond-vector multipole sites sit on their host atoms this way.

## Common conventions

* **Exclusions.** The pair forces take explicit exclusions (`addExclusion`), as `CustomNonbondedForce` does. The
  InteraxFF builder excludes every intramolecular pair. All pair forces of a system should have identical
  exclusions so that the CUDA platform can evaluate them in one tiled nonbonded kernel.
* **Cutoff.** `NoCutoff`, `CutoffNonPeriodic` or `CutoffPeriodic` (minimum image). Optional switching function
  `S(r) = 1 - 10x^3 + 15x^4 - 6x^5`, `x = (r - r_switch)/(r_cut - r_switch)`, applied to the pair energy exactly as in
  `CustomNonbondedForce`.
* **Alchemy (decoupling).** Each particle (each molecule for the three-body force) can be flagged alchemical. A pair
  with exactly one alchemical particle is scaled by the context parameter `lambda_sterics` (name configurable);
  pairs with zero or two alchemical particles are unscaled (two alchemical particles are normally excluded, being in
  the same solute). The dispersion long range correction is the exception: every pair that involves an alchemical
  particle enters it scaled by lambda, the pairs within the alchemical molecule included, so that it is that of the
  whole system at lambda = 1 and that of the rest at lambda = 0. A three-body triple is scaled if it contains at least one alchemical molecule. This reproduces
  the absolv decoupling construction (solute-solvent copies scaled by `lambda_sterics`) in a single force.
* **Particles without density.** A particle with zero width `s` (e.g. a lone-pair virtual site) does not interact
  through the density-based pair forces.

## Slater overlap integral

Normalized 1s Slater densities `rho(r) = exp(-r/s) / (8 pi s^3)`. Their overlap integral (nm^-3) for widths a, b at
distance r (MEDFF SI Eq. 32):

    S(r; a, b) = F(r) / (8 pi r)
    F(r) = 4 a^2 b^2 / D^3 * (e_a - e_b) + r / D^2 * (a e_a + b e_b),   D = b^2 - a^2,  e_a = exp(-r/a)
    F'(r) = 4 a^2 b^2 / D^3 * (e_b / b - e_a / a) + (a e_a + b e_b) / D^2 - r (e_a + e_b) / D^2
    dS/dr = (r F' - F) / (8 pi r^2)

The two terms of F cancel to relative order (b - a)^2, so for nearly equal widths the integral is evaluated from its
Taylor expansion about a width l, with t = (b - l)/l (b the other width) and x = r / l
(`interax.geometry.OVERLAP_TAYLOR_COEFFS`):

    S = exp(-x) / (pi l^3) * sum_{n=0}^{N} t^n P_n(x) / D_n
    dS/dr = exp(-x) / (pi l^4) * sum_{n=0}^{N} t^n (P_n'(x) - P_n(x)) / D_n

`InteraxExchangeForce` and `InteraxShortRangeForce` select the scheme with `setOverlapMethod()`:

| Method | Order N | l | Taylor where | Use |
|---|---|---|---|---|
| `Exact` (default) | 6 | the smaller width | t < 0.2 and t x < 1.6 (single precision), t < 0.02 and t x < 0.3 (double) | the exact integral: ~1e-6 relative in single, ~1e-10 in double precision |
| `InteraxOrder2` | 2 | width of the particle with the lower index | \|b - a\| < 0.01 bohr | the InteraxFF default (MEDFF); reproduces the export |
| `InteraxOrder6` | 6 | width of the particle with the lower index | \|t\| < 0.2 | the InteraxFF order-6 scheme; reproduces the export |

The InteraxFF schemes (`interax.geometry.SLATER_TAYLOR_SCHEMES`) are evaluated exactly as the export evaluates them on
the Reference platform, where particle 1 of each pair is the one with the lower index, with the signed width
difference. They carry the export's truncation errors: up to ~1e-6 relative to the exact integral (order 6 degrades as
(t x)^7 at large distances), and in single precision the closed form close to the order-2 switch has relative errors
up to ~3e-2. `convert_system(overlap="export")` picks the scheme of the export, `overlap="exact"` (the default) the
`Exact` method; the force field files use `Exact` unless they set `overlapMethod`. The CUDA kernels compare the original particle indices, which the context may reorder.

The normalized overlap is `St(r) = 8 pi (a + b)^3 S(r; a, b)`, with `St(0) = 1`.

## InteraxDispersionForce

Per particle: `m1, m2, m3` (XDM moments <M_k^2>, kJ/mol nm^(2k+1)), `alpha` (dispersion polarizability, nm^3),
`s` (Slater width, nm). Global: damping scale beta.

    E = - sum_{pairs} sum_{n = 6, 8, 10} f_n(x) C_n / r^n,        x = beta * 2 r / (s_i + s_j)
    Dn = alpha_i m1_j + alpha_j m1_i
    C6  = alpha_i alpha_j m1_i m1_j / Dn
    C8  = 3/2 alpha_i alpha_j (m1_i m2_j + m2_i m1_j) / Dn
    C10 = [2 alpha_i alpha_j (m1_i m3_j + m3_i m1_j) + 21/5 alpha_i alpha_j m2_i m2_j] / Dn
    f_n(x) = 1 - exp(-x) sum_{k=0}^{n} x^k / k!     (Tang-Toennies; regularized incomplete gamma P(n + 1, x))
    f_n'(x) = exp(-x) x^n / n!

`f_n` is evaluated as `exp(-x) sum_{k > n} x^k / k!` for `x < (n + 1)/2`, where the closed form cancels. A pair with
`alpha_i alpha_j = 0` is skipped (its `C_n` are zero), which is also how a lone-pair site is kept out of the
dispersion.

Long-range correction (periodic, optional): `E_lrc = (2 pi N^2 / V) <I>`, with `<I>` the average over all
`N (N + 1)/2` particle pairs (self pairs included) of

    I_ij = int_{r_c}^{inf} E_ij(r) r^2 dr + int_{r_s}^{r_c} (1 - S(r)) E_ij(r) r^2 dr

exactly OpenMM's `CustomNonbondedForceImpl::calcLongRangeCorrection` convention, so that converted systems agree.
Pairs with exactly one alchemical particle enter `<I>` scaled by `lambda_sterics`.

## InteraxExchangeForce

Per particle: `A` (sqrt(kJ/mol nm^3)), `s` (exchange width, nm; the InteraxFF width already scaled per element by
kappa).

    E = sum_{pairs} A_i A_j S(r; s_i, s_j)

(InteraxFF writes `K_A * A1 * A2 * S` with A in atomic units: `A = sqrt(K_A) A_InteraxFF`.)

## InteraxShortRangeForce

Per particle: `K`, `L` (kJ/mol), `s` (nm).

    E = sum_{pairs} [ (K_i + K_j)/2 St + (L_i + L_j)/2 St^2 ],     St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j)

InteraxFF's optional element-pair matrix and mid-range `M` term are not supported.

## InteraxThreeBodyForce

Per site: the particles whose unweighted centroid is the site, the polarizability `alpha` (nm^3), the single-pole
frequency `w` (kJ/mol, `C6_AA = 3/4 alpha^2 w`), a width `s` (nm, for the damping), the molecule the site belongs to,
and the alchemical flag. Global: cutoff `r_c` (every pair of a triple), a scale factor, the damping method and scale.

    E = scale * sum_{A < B < C, all three pairs < r_c} C9_ABC (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC r_BC)^3 * f_AB f_AC f_BC
    C9_ABC = 3/2 a_A a_B a_C w_A w_B w_C (w_A + w_B + w_C) / ((w_A + w_B)(w_B + w_C)(w_C + w_A))

Two forms are supported:

* **Atomic** (InteraxFF's "atomic" mode, or `interaxplugin_convert.atomic_three_body`): one site per atom with its
  atomic oscillator, and the Tang-Toennies damping of each of the three dipole propagators (~ r^-3) of a triple,
  `f_ij = f_3(x)`, `x = beta 2 r_ij / (s_i + s_j)` (as in the dispersion), with the dispersion widths. Triples with two
  sites of the same molecule are excluded.
* **Molecular** (InteraxFF's "molecular" mode): one site per molecule, the centroid of its heavy atoms, with the
  molecular polarizability and frequency (London combination of the atomic oscillators), no damping method, and
  optionally the centroid damping below.

Independently of the damping method, a **centroid damping radius** `R0 > 0` (`setCentroidDampingRadius`, nm) multiplies
each `f_ij` by `g(r_ij) = 1 - exp(-(r_ij/R0)^6)` (`g' = 6 (r/R0)^6 exp(-(r/R0)^6) / r`). The undamped molecular ATM
diverges as `r^-3` per pair when two centroids come close (flexible molecules). `R0 = 0` (the default) means no
centroid damping and leaves every result bit-identical. InteraxFF (`ForceField.atm_damping_r0`, in Angstrom) uses the
same function. The GPU kernel evaluates `1 - exp(-y)` by its series below `y = 0.1`. A force with centroid damping is
serialized with version 3, one without with version 2.

The angle at each vertex is the angle between the minimum-image vectors to its two neighbors, exactly as in the
export (`theta = angle(g1, g2, g3)` of `CustomCentroidBondForce`). With `a = x_B - x_A`, `b = x_C - x_A`,
`c = x_C - x_B` (each the minimum image under periodic boundary conditions), `u, v, w` their squared lengths,
`P = u v w` and `p = a.b, q = a.c, s = b.c`, the product of the three cosines is `-pqs/P` and, undamped,

    E0 = C9 [ P^(-3/2) - 3 pqs P^(-5/2) ]
    dE0/da = a (R/u) - 3 C9 P^(-5/2) (qs b + ps c),    R = C9 (-3 P^(-3/2) + 15 pqs P^(-5/2))
    dE0/db = b (R/v) - 3 C9 P^(-5/2) (qs a + pq c)
    dE0/dc = c (R/w) - 3 C9 P^(-5/2) (ps a + pq b)
    F_A = dE/da + dE/db,   F_B = dE/dc - dE/da,   F_C = -(dE/db + dE/dc)

With damping, `E = E0 f_a f_b f_c` and `dE/da = f_a f_b f_c dE0/da + E0 f_b f_c f_a'(|a|) a/|a|` (and similarly for b,
c). Each site force is shared equally among its particles. When the three minimum images close into a triangle (always
when the box is larger than three cutoffs) this is the familiar `C9 (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC
r_BC)^3`; in smaller boxes (for example a cutoff of 1.2 nm in a 2-3.4 nm box) the images need not close, and this
convention reproduces the export there too. Sites come from the raw particle positions, as
`CustomCentroidBondForce` does.

Triples are found from neighbor lists of the sites instead of being stored explicitly (a `CustomCentroidBondForce`
would store all N^3/6 triples: 20.8 million for a 500-water box). On the GPU each site keeps a half list of the sites
above it within the cutoff plus a skin of 0.1 nm, rebuilt when a site has moved by more than half the skin. At every
evaluation the neighbors within the cutoff are compacted, with the bounding box of every tile of them; a thread block
per site loops over the pairs of tiles, skipping those whose boxes are more than the cutoff apart, and computes each
triple once, from its lowest site. The lists grow when they overflow; until then the triples of a site whose list
overflowed are found by scanning all the sites above it, so the result never depends on the size of the lists. The
number of triples is about
`N n^2 p / 6` with `n = rho (4/3) pi r_c^3` neighbors per site and `p ~ 0.47`, so at a given cutoff an atomic ATM costs
(sites per molecule)^3 times the molecular one: 27x for water, 216x for methanol at 1.2 nm, and `r_c^6` less at a
shorter cutoff.

## Tests

`tests/data` holds the energies and forces of each term computed by the export for two systems (written by
`tests/python/generate_export_values.py`). `tests/TestInteraxExportValues.h` checks every platform against them
(ctest: Reference, CPU, CUDA single/mixed/double), and `tests/python/test_export_values.py` does the same through the
Python API: in double precision the plugin reproduces the export to rounding with its overlap scheme, and the `Exact`
method agrees to ~1e-6. The data were generated with the InteraxFF model `models_v3R4bfdE1` and the order-2 Slater
scheme, which the generator uses whatever the model's.
`tests/TestInteraxMultipoleForce.h` tests the multipole force against HippoNonbondedForce and against the definitions
of its options (core damping, mutual damping scale, alchemical scaling, chirality modes, ThreeFold frames, coincident
particles with and without PME). `tests/python/test_alchemical.py` checks the lambda scaling against decoupled
systems, `tests/python/compare_export.py` compares freshly built systems with the export, and
`tests/python/test_forcefield.py` checks the force field files against forces built with the API.
