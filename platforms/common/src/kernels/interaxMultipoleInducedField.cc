
KERNEL void initExtrapolatedDipoles(GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT extrapolatedDipole
#ifdef USE_GK
        , GLOBAL real* RESTRICT inducedDipoleGk, GLOBAL real* RESTRICT inducedDipoleGkPolar, GLOBAL real* RESTRICT extrapolatedDipoleGk, GLOBAL real* RESTRICT extrapolatedDipoleGkPolar,
        GLOBAL mm_long* RESTRICT inducedDipoleFieldGradientGk, GLOBAL mm_long* RESTRICT inducedDipoleFieldGradientGkPolar
#endif
        ) {
    for (int index = GLOBAL_ID; index < 3*NUM_ATOMS; index += GLOBAL_SIZE) {
        extrapolatedDipole[index] = inducedDipole[index];
#ifdef USE_GK
        extrapolatedDipoleGk[index] = inducedDipoleGk[index];
        extrapolatedDipoleGkPolar[index] = inducedDipoleGkPolar[index];
#endif
    }
}

KERNEL void iterateExtrapolatedDipoles(int order, GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT extrapolatedDipole, GLOBAL mm_long* RESTRICT inducedDipoleField,
#ifdef USE_GK
        GLOBAL real* RESTRICT inducedDipoleGk, GLOBAL real* RESTRICT inducedDipoleGkPolar, GLOBAL real* RESTRICT extrapolatedDipoleGk, GLOBAL real* RESTRICT extrapolatedDipoleGkPolar,
        GLOBAL mm_long* RESTRICT inducedDipoleFieldGradientGk, GLOBAL mm_long* RESTRICT inducedDipoleFieldGradientGkPolar, GLOBAL mm_long* RESTRICT inducedDipoleFieldGk,
        GLOBAL mm_long* RESTRICT inducedDipoleFieldGkPolar, GLOBAL real* RESTRICT extrapolatedDipoleFieldGradientGk, GLOBAL real* RESTRICT extrapolatedDipoleFieldGradientGkPolar,
#endif
        GLOBAL const real* RESTRICT polarizability
    ) {
    const real fieldScale = 1/(real) 0x100000000;
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real polar = polarizability[atom];
        real3 value = make_real3(inducedDipoleField[atom], inducedDipoleField[atom+PADDED_NUM_ATOMS], inducedDipoleField[atom+2*PADDED_NUM_ATOMS])*fieldScale*polar;
        inducedDipole[3*atom] = value.x;
        inducedDipole[3*atom+1] = value.y;
        inducedDipole[3*atom+2] = value.z;
        extrapolatedDipole[3*(order*NUM_ATOMS+atom)] = value.x;
        extrapolatedDipole[3*(order*NUM_ATOMS+atom)+1] = value.y;
        extrapolatedDipole[3*(order*NUM_ATOMS+atom)+2] = value.z;
#ifdef USE_GK
        value = make_real3(inducedDipoleFieldGk[atom], inducedDipoleFieldGk[atom+PADDED_NUM_ATOMS], inducedDipoleFieldGk[atom+2*PADDED_NUM_ATOMS])*fieldScale*polar;
        inducedDipoleGk[3*atom] = value.x;
        inducedDipoleGk[3*atom+1] = value.y;
        inducedDipoleGk[3*atom+2] = value.z;
        extrapolatedDipoleGk[3*(order*NUM_ATOMS+atom)] = value.x;
        extrapolatedDipoleGk[3*(order*NUM_ATOMS+atom)+1] = value.y;
        extrapolatedDipoleGk[3*(order*NUM_ATOMS+atom)+2] = value.z;
        value = make_real3(inducedDipoleFieldGkPolar[atom], inducedDipoleFieldGkPolar[atom+PADDED_NUM_ATOMS], inducedDipoleFieldGkPolar[atom+2*PADDED_NUM_ATOMS])*fieldScale*polar;
        inducedDipoleGkPolar[3*atom] = value.x;
        inducedDipoleGkPolar[3*atom+1] = value.y;
        inducedDipoleGkPolar[3*atom+2] = value.z;
        extrapolatedDipoleGkPolar[3*(order*NUM_ATOMS+atom)] = value.x;
        extrapolatedDipoleGkPolar[3*(order*NUM_ATOMS+atom)+1] = value.y;
        extrapolatedDipoleGkPolar[3*(order*NUM_ATOMS+atom)+2] = value.z;
#endif
    }
}

KERNEL void computeExtrapolatedDipoles(GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT extrapolatedDipole
#ifdef USE_GK
        , GLOBAL real* RESTRICT inducedDipoleGk, GLOBAL real* RESTRICT inducedDipoleGkPolar, GLOBAL real* RESTRICT extrapolatedDipoleGk, GLOBAL real* RESTRICT extrapolatedDipoleGkPolar
#endif
        ) {
    real coeff[] = {EXTRAPOLATION_COEFFICIENTS_SUM};
    for (int index = GLOBAL_ID; index < 3*NUM_ATOMS; index += GLOBAL_SIZE) {
        real sum = 0, sumPolar = 0, sumGk = 0, sumGkPolar = 0;
        for (int order = 0; order < MAX_EXTRAPOLATION_ORDER; order++) {
            sum += extrapolatedDipole[order*3*NUM_ATOMS+index]*coeff[order];
#ifdef USE_GK
            sumGk += extrapolatedDipoleGk[order*3*NUM_ATOMS+index]*coeff[order];
            sumGkPolar += extrapolatedDipoleGkPolar[order*3*NUM_ATOMS+index]*coeff[order];
#endif
        }
        inducedDipole[index] = sum;
#ifdef USE_GK
        inducedDipoleGk[index] = sumGk;
        inducedDipoleGkPolar[index] = sumGkPolar;
#endif
    }
}

KERNEL void addExtrapolatedFieldGradientToForce(GLOBAL mm_long* RESTRICT forceBuffers, GLOBAL real* RESTRICT extrapolatedDipole,
        GLOBAL real* RESTRICT extrapolatedDipolePolar, GLOBAL real* RESTRICT extrapolatedDipoleFieldGradient, GLOBAL real* RESTRICT extrapolatedDipoleFieldGradientPolar
#ifdef USE_GK
        , GLOBAL real* RESTRICT extrapolatedDipoleGk, GLOBAL real* RESTRICT extrapolatedDipoleGkPolar,
        GLOBAL real* RESTRICT extrapolatedDipoleFieldGradientGk, GLOBAL real* RESTRICT extrapolatedDipoleFieldGradientGkPolar
#endif
        ) {
    real coeff[] = {EXTRAPOLATION_COEFFICIENTS_SUM};
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real fx = 0, fy = 0, fz = 0;
        for (int l = 0; l < MAX_EXTRAPOLATION_ORDER-1; l++) {
            int index1 = 3*(l*NUM_ATOMS+atom);
            real dipole[] = {extrapolatedDipole[index1], extrapolatedDipole[index1+1], extrapolatedDipole[index1+2]};
            real dipolePolar[] = {extrapolatedDipolePolar[index1], extrapolatedDipolePolar[index1+1], extrapolatedDipolePolar[index1+2]};
#ifdef USE_GK
            real dipoleGk[] = {extrapolatedDipoleGk[index1], extrapolatedDipoleGk[index1+1], extrapolatedDipoleGk[index1+2]};
            real dipoleGkPolar[] = {extrapolatedDipoleGkPolar[index1], extrapolatedDipoleGkPolar[index1+1], extrapolatedDipoleGkPolar[index1+2]};
#endif
            for (int m = 0; m < MAX_EXTRAPOLATION_ORDER-1-l; m++) {
                int index2 = 6*(m*NUM_ATOMS+atom);
                real scale = 0.5f*coeff[l+m+1]*ENERGY_SCALE_FACTOR;
                real gradient[] = {extrapolatedDipoleFieldGradient[index2], extrapolatedDipoleFieldGradient[index2+1], extrapolatedDipoleFieldGradient[index2+2],
                                   extrapolatedDipoleFieldGradient[index2+3], extrapolatedDipoleFieldGradient[index2+4], extrapolatedDipoleFieldGradient[index2+5]};
                real gradientPolar[] = {extrapolatedDipoleFieldGradientPolar[index2], extrapolatedDipoleFieldGradientPolar[index2+1], extrapolatedDipoleFieldGradientPolar[index2+2],
                                        extrapolatedDipoleFieldGradientPolar[index2+3], extrapolatedDipoleFieldGradientPolar[index2+4], extrapolatedDipoleFieldGradientPolar[index2+5]};
                fx += scale*(dipole[0]*gradientPolar[0] + dipole[1]*gradientPolar[3] + dipole[2]*gradientPolar[4]);
                fy += scale*(dipole[0]*gradientPolar[3] + dipole[1]*gradientPolar[1] + dipole[2]*gradientPolar[5]);
                fz += scale*(dipole[0]*gradientPolar[4] + dipole[1]*gradientPolar[5] + dipole[2]*gradientPolar[2]);
                fx += scale*(dipolePolar[0]*gradient[0] + dipolePolar[1]*gradient[3] + dipolePolar[2]*gradient[4]);
                fy += scale*(dipolePolar[0]*gradient[3] + dipolePolar[1]*gradient[1] + dipolePolar[2]*gradient[5]);
                fz += scale*(dipolePolar[0]*gradient[4] + dipolePolar[1]*gradient[5] + dipolePolar[2]*gradient[2]);
#ifdef USE_GK
                real gradientGk[] = {extrapolatedDipoleFieldGradient[index2], extrapolatedDipoleFieldGradient[index2+1], extrapolatedDipoleFieldGradient[index2+2],
                                   extrapolatedDipoleFieldGradient[index2+3], extrapolatedDipoleFieldGradient[index2+4], extrapolatedDipoleFieldGradient[index2+5]};
                real gradientGkPolar[] = {extrapolatedDipoleFieldGradientPolar[index2], extrapolatedDipoleFieldGradientPolar[index2+1], extrapolatedDipoleFieldGradientPolar[index2+2],
                                        extrapolatedDipoleFieldGradientPolar[index2+3], extrapolatedDipoleFieldGradientPolar[index2+4], extrapolatedDipoleFieldGradientPolar[index2+5]};
                fx += scale*(dipoleGk[0]*gradientGkPolar[0] + dipoleGk[1]*gradientGkPolar[3] + dipoleGk[2]*gradientGkPolar[4]);
                fy += scale*(dipoleGk[0]*gradientGkPolar[3] + dipoleGk[1]*gradientGkPolar[1] + dipoleGk[2]*gradientGkPolar[5]);
                fz += scale*(dipoleGk[0]*gradientGkPolar[4] + dipoleGk[1]*gradientGkPolar[5] + dipoleGk[2]*gradientGkPolar[2]);
                fx += scale*(dipoleGkPolar[0]*gradientGk[0] + dipoleGkPolar[1]*gradientGk[3] + dipoleGkPolar[2]*gradientGk[4]);
                fy += scale*(dipoleGkPolar[0]*gradientGk[3] + dipoleGkPolar[1]*gradientGk[1] + dipoleGkPolar[2]*gradientGk[5]);
                fz += scale*(dipoleGkPolar[0]*gradientGk[4] + dipoleGkPolar[1]*gradientGk[5] + dipoleGkPolar[2]*gradientGk[2]);
#endif
            }
        }
        forceBuffers[atom] += realToFixedPoint(fx);
        forceBuffers[atom+PADDED_NUM_ATOMS] += realToFixedPoint(fy);
        forceBuffers[atom+PADDED_NUM_ATOMS*2] += realToFixedPoint(fz);
    }
}

// -1/2 E.mu with E the permanent field on the induced dipoles (HIPPO divides mu_0 by the polarizability instead, which
// fails for zero polarizabilities).
KERNEL void computePolarizationEnergy(GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const mm_long* RESTRICT fieldBuffers) {
    const real fieldScale = 1/(real) 0x100000000;
    mixed energy = 0;
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE)
        for (int i = 0; i < 3; i++)
            energy -= (ENERGY_SCALE_FACTOR/2)*(fieldBuffers[atom+i*PADDED_NUM_ATOMS]*fieldScale)*inducedDipole[3*atom+i];
    energyBuffer[GLOBAL_ID] += energy;
}