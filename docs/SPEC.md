# InteraxFF OpenMM plugin: specification

This plugin implements the non-electrostatic intermolecular terms of the InteraxFF force field as four OpenMM
forces, each with Reference, CPU and CUDA (common compute: CUDA/OpenCL) kernels. Electrostatics and polarization
stay in OpenMM's `HippoNonbondedForce`, as in the existing InteraxFF export (`interax/openmm.py`).

| Force | Replaces (current export) | Energy |
|---|---|---|
| `InteraxDispersionForce` | `CustomNonbondedForce` (XDM, Tang-Toennies) | damped C6/C8/C10 pair dispersion |
| `InteraxExchangeForce` | `CustomNonbondedForce` (`K_A * A1 * A2 * S`) | Slater-overlap exchange repulsion |
| `InteraxShortRangeForce` | `CustomNonbondedForce` (`K St + L St^2`) | short-range correction |
| `InteraxThreeBodyForce` | `CustomCentroidBondForce` (all triples) | molecular Axilrod-Teller-Muto dispersion |

Units are OpenMM's: nm, kJ/mol, e. Parameters are those of the current export, so a system built by
`interax.openmm.create_system_from_topology` can be converted force by force and must give the same energies and
forces (the reference for every test).

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
  the same solute). A three-body triple is scaled if it contains at least one alchemical molecule. This reproduces
  the absolv decoupling construction (solute-solvent copies scaled by `lambda_sterics`) in a single force.
* **Particles without density.** A particle with zero width `s` (e.g. a lone-pair virtual site) does not interact
  through the density-based pair forces. (In the current export such sites need an extra gating parameter, because
  the additive short-range combination rule would otherwise give them their partner's amplitude.)

## Slater overlap integral

Normalized 1s Slater densities `rho(r) = exp(-r/s) / (8 pi s^3)`. Their overlap integral (nm^-3) for widths a, b at
distance r (MEDFF SI Eq. 32):

    S(r; a, b) = F(r) / (8 pi r)
    F(r) = 4 a^2 b^2 / D^3 * (e_a - e_b) + r / D^2 * (a e_a + b e_b),   D = b^2 - a^2,  e_a = exp(-r/a)
    F'(r) = 4 a^2 b^2 / D^3 * (e_b / b - e_a / a) + (a e_a + b e_b) / D^2 - r (e_a + e_b) / D^2
    dS/dr = (r F' - F) / (8 pi r^2)

The two terms of F cancel to relative order (b - a)^2, so for nearly equal widths the integral is evaluated from its
Taylor expansion about the smaller width l = min(a, b), with d = |b - a| and x = r / l
(`interax.geometry.OVERLAP_TAYLOR_COEFFS`):

    S = exp(-x) * sum_{n=0}^{6} d^n P_n(x) / (D_n pi l^(3+n))
    dS/dr = exp(-x) / l * sum_{n=0}^{6} d^n (P_n'(x) - P_n(x)) / (D_n pi l^(3+n))

used when `d < 0.2 l` (the order-6 scheme of `interax.geometry.SLATER_TAYLOR_SCHEMES`). The export uses the order-2
scheme below 0.01 bohr; both evaluate the same exact integral, and the order-6 window also removes the single-
precision cancellation of the closed form close to the switch. Expanding about the smaller width makes the result
independent of the order of the two particles.

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

`f_n` is evaluated as `exp(-x) sum_{k > n} x^k / k!` for `x < (n + 1)/2`, where the closed form cancels. The
export adds 1e-16 to `Dn` to keep `alpha = 0` particles finite; here a pair with `alpha_i alpha_j = 0` is skipped
(its `C_n` are zero), which is also how a lone-pair site is kept out of the dispersion.

Long-range correction (periodic, optional): `E_lrc = (2 pi N^2 / V) <I>`, with `<I>` the average over all
`N (N + 1)/2` particle pairs (self pairs included) of

    I_ij = int_{r_c}^{inf} E_ij(r) r^2 dr + int_{r_s}^{r_c} (1 - S(r)) E_ij(r) r^2 dr

exactly OpenMM's `CustomNonbondedForceImpl::calcLongRangeCorrection` convention, so that converted systems agree.
Pairs with exactly one alchemical particle enter `<I>` scaled by `lambda_sterics`.

## InteraxExchangeForce

Per particle: `A` (sqrt(kJ/mol nm^3)), `s` (exchange width, nm; the InteraxFF width already scaled per element by
kappa).

    E = sum_{pairs} A_i A_j S(r; s_i, s_j)

(The export writes `K_A * A1 * A2 * S` with A in atomic units: `A_plugin = sqrt(K_A) A_export`.)

## InteraxShortRangeForce

Per particle: `K`, `L` (kJ/mol), `s` (nm).

    E = sum_{pairs} [ (K_i + K_j)/2 St + (L_i + L_j)/2 St^2 ],     St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j)

The export's optional element-pair matrix and mid-range `M` term are zero in every current model and are not
supported.

## InteraxThreeBodyForce

Per molecule: the particles whose unweighted centroid is the site (heavy atoms in InteraxFF), the molecular
polarizability `alpha` (nm^3) and the single-pole frequency `w` (kJ/mol, `C6_AA = 3/4 alpha^2 w`),
and the alchemical flag. Global: cutoff `r_c` (every pair of a triple), a scale factor.

    E = scale * sum_{A < B < C, all three pairs < r_c} C9_ABC (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC r_BC)^3
    C9_ABC = 3/2 a_A a_B a_C w_A w_B w_C (w_A + w_B + w_C) / ((w_A + w_B)(w_B + w_C)(w_C + w_A))

The angle at each vertex is the angle between the minimum-image vectors to its two neighbors, exactly as in the
export (`theta = angle(g1, g2, g3)` of `CustomCentroidBondForce`). With `a = x_B - x_A`, `b = x_C - x_A`,
`c = x_C - x_B` (each the minimum image under periodic boundary conditions), `u, v, w` their squared lengths,
`P = u v w` and `p = a.b, q = a.c, s = b.c`, the product of the three cosines is `-pqs/P` and

    E = C9 [ P^(-3/2) - 3 pqs P^(-5/2) ]
    dE/da = a (R/u) - 3 C9 P^(-5/2) (qs b + ps c),    R = C9 (-3 P^(-3/2) + 15 pqs P^(-5/2))
    dE/db = b (R/v) - 3 C9 P^(-5/2) (qs a + pq c)
    dE/dc = c (R/w) - 3 C9 P^(-5/2) (ps a + pq b)
    F_A = dE/da + dE/db,   F_B = dE/dc - dE/da,   F_C = -(dE/db + dE/dc)

each centroid force shared equally among its particles. When the three minimum images close into a triangle
(always when the box is larger than three cutoffs) this is the familiar
`C9 (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC r_BC)^3`; in smaller boxes (the InteraxFF liquid and HFE boxes:
cutoff 1.2 nm, box 2-3.4 nm) the images need not close, and this convention reproduces the export there too.
Centroids come from the raw particle positions, as `CustomCentroidBondForce` does. Triples are found from a centroid neighbour list instead
of being stored explicitly (the export stores all N^3/6 triples: 20.8 million for a 500-water box).
