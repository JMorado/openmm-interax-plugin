// Part of the OpenMM InteraxFF plugin. Copyright (c) 2026 the Authors. GNU LGPL version 3 or later, see LICENSE.
// Derived from the HIPPO kernels of OpenMM, portions copyright (c) 2008-2021 Stanford University and the Authors.

KERNEL void initExtrapolatedDipoles(GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT extrapolatedDipole) {
    for (int index = GLOBAL_ID; index < 3*NUM_ATOMS; index += GLOBAL_SIZE) {
        extrapolatedDipole[index] = inducedDipole[index];
    }
}

KERNEL void iterateExtrapolatedDipoles(int order, GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT extrapolatedDipole, GLOBAL mm_long* RESTRICT inducedDipoleField,
        GLOBAL const real* RESTRICT polarizability) {
    const real fieldScale = 1/(real) 0x100000000;
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real polar = polarizability[atom]; // The polarizability in the precision of the context
        real3 value = make_real3(inducedDipoleField[atom], inducedDipoleField[atom+PADDED_NUM_ATOMS], inducedDipoleField[atom+2*PADDED_NUM_ATOMS])*fieldScale*polar;
        inducedDipole[3*atom] = value.x;
        inducedDipole[3*atom+1] = value.y;
        inducedDipole[3*atom+2] = value.z;
        extrapolatedDipole[3*(order*NUM_ATOMS+atom)] = value.x;
        extrapolatedDipole[3*(order*NUM_ATOMS+atom)+1] = value.y;
        extrapolatedDipole[3*(order*NUM_ATOMS+atom)+2] = value.z;
    }
}

KERNEL void computeExtrapolatedDipoles(GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT extrapolatedDipole) {
    real coeff[] = {EXTRAPOLATION_COEFFICIENTS_SUM};
    for (int index = GLOBAL_ID; index < 3*NUM_ATOMS; index += GLOBAL_SIZE) {
        real sum = 0;
        for (int order = 0; order < MAX_EXTRAPOLATION_ORDER; order++) {
            sum += extrapolatedDipole[order*3*NUM_ATOMS+index]*coeff[order];
        }
        inducedDipole[index] = sum;
    }
}

/**
 * The polarization energy is -1/2 E.mu, with E the permanent field on the induced dipoles. (HIPPO divides
 * the direct dipoles by the polarizability instead, which fails for particles with zero polarizability.)
 */
KERNEL void computePolarizationEnergy(GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const mm_long* RESTRICT fieldBuffers) {
    const real fieldScale = 1/(real) 0x100000000;
    mixed energy = 0;
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE)
        for (int i = 0; i < 3; i++)
            energy -= (ENERGY_SCALE_FACTOR/2)*(fieldBuffers[atom+i*PADDED_NUM_ATOMS]*fieldScale)*inducedDipole[3*atom+i];
    energyBuffer[GLOBAL_ID] += energy;
}
