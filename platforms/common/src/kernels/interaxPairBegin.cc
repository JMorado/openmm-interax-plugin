// Start of the code for one InteraxFF pair force in the nonbonded kernel: pairs that are not excluded, are within
// the cutoff, and whose gate parameters (polarizability or width) are both nonzero.
#ifdef USE_CUTOFF
if (!isExcluded && r2 < CUTOFF_SQUARED && PAIR_GATE) {
#else
if (!isExcluded && PAIR_GATE) {
#endif
    real pairEnergy = 0, pairDerivative = 0;
    // A pair with exactly one alchemical particle is evaluated at the soft-core distance
    // r_eff = (r^6 + (1 - lambda) R^6)^(1/6), if enabled: the interaction below sees r_eff as r.
#if USE_SOFTCORE
    const real softcoreR = (IS_ALCHEMICAL1 != IS_ALCHEMICAL2 ? POW(r2*r2*r2 + (1-(real) LAMBDA)*SOFTCORE_RADIUS6, (real) (1.0/6.0)) : r);
    const real softcoreInvR = 1/softcoreR;
#else
    const real softcoreR = r, softcoreInvR = invR;
#endif
    {
    const real r = softcoreR;
    const real invR = softcoreInvR;
