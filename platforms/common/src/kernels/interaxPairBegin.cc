// Start of the code for one InteraxFF pair force in the nonbonded kernel: pairs that are not excluded, are within
// the cutoff, and whose gate parameters (polarizability or width) are both nonzero.
#ifdef USE_CUTOFF
if (!isExcluded && r2 < CUTOFF_SQUARED && PAIR_GATE) {
#else
if (!isExcluded && PAIR_GATE) {
#endif
    real pairEnergy = 0, pairDerivative = 0;
