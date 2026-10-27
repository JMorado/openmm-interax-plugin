// Functions called by the InteraxMultipoleForce interaction kernels (from OpenMM's HIPPO kernels, hippoInteractionHeader.cc).

// InteraxFF: whether a particle has no charge, multipole or induced dipole (an alchemical particle at lambda = 0).
// Every pair term contains a moment of each particle, so the pairs of such a particle are skipped: they are exactly
// zero, but the inverse powers of r overflow in single precision when it overlaps another particle (rr11 below
// ~0.001 nm, which a decoupled solute allows), and zero times infinity is not zero.
DEVICE bool isInertParticle(real coreCharge, real valenceCharge, real3 dipole, real qXX, real qXY, real qXZ, real qYY, real qYZ,
                            real3 inducedDipole) {
    return coreCharge == 0 && valenceCharge == 0 && dipole.x == 0 && dipole.y == 0 && dipole.z == 0 && qXX == 0 && qXY == 0 &&
           qXZ == 0 && qYY == 0 && qYZ == 0 && inducedDipole.x == 0 && inducedDipole.y == 0 && inducedDipole.z == 0;
}

DEVICE void formQIRotationMatrix(real3 deltaR, real rInv, real rotationMatrix[3][3]) {
    real3 vectorZ = deltaR*rInv;
    real3 vectorX = make_real3(0);
    if (fabs(vectorZ.y) > fabs(vectorZ.x))
        vectorX.x = 1;
    else
        vectorX.y = 1;

    vectorX -= vectorZ*dot(vectorZ, vectorX);
    vectorX = normalize(vectorX);
    real3 vectorY = cross(vectorZ, vectorX);

    rotationMatrix[0][0] = vectorX.x;
    rotationMatrix[0][1] = vectorX.y;
    rotationMatrix[0][2] = vectorX.z;
    rotationMatrix[1][0] = vectorY.x;
    rotationMatrix[1][1] = vectorY.y;
    rotationMatrix[1][2] = vectorY.z;
    rotationMatrix[2][0] = vectorZ.x;
    rotationMatrix[2][1] = vectorZ.y;
    rotationMatrix[2][2] = vectorZ.z;
}

DEVICE real3 rotateVectorToQI(real3 v, const real mat[3][3]) {
    return make_real3(mat[0][0]*v.x + mat[0][1]*v.y + mat[0][2]*v.z,
                      mat[1][0]*v.x + mat[1][1]*v.y + mat[1][2]*v.z,
                      mat[2][0]*v.x + mat[2][1]*v.y + mat[2][2]*v.z);
}

DEVICE real3 rotateVectorFromQI(real3 v, const real mat[3][3]) {
    return make_real3(mat[0][0]*v.x + mat[1][0]*v.y + mat[2][0]*v.z,
                      mat[0][1]*v.x + mat[1][1]*v.y + mat[2][1]*v.z,
                      mat[0][2]*v.x + mat[1][2]*v.y + mat[2][2]*v.z);
}

DEVICE void rotateQuadrupoleToQI(real qXX, real qXY, real qXZ, real qYY, real qYZ, 
            real* qiQXX, real* qiQXY, real* qiQXZ, real* qiQYY, real* qiQYZ, real* qiQZZ, const real mat[3][3]) {
    real qZZ = -qXX-qYY;
    *qiQXX = mat[0][0]*(mat[0][0]*qXX + 2*(mat[0][1]*qXY + mat[0][2]*qXZ)) + mat[0][1]*(mat[0][1]*qYY + 2*mat[0][2]*qYZ) + mat[0][2]*mat[0][2]*qZZ;
    *qiQYY = mat[1][0]*(mat[1][0]*qXX + 2*(mat[1][1]*qXY + mat[1][2]*qXZ)) + mat[1][1]*(mat[1][1]*qYY + 2*mat[1][2]*qYZ) + mat[1][2]*mat[1][2]*qZZ;
    *qiQXY = mat[0][0]*mat[1][0]*qXX + mat[0][1]*mat[1][1]*qYY + mat[0][2]*mat[1][2]*qZZ + (mat[0][0]*mat[1][1] + mat[0][1]*mat[1][0])*qXY + (mat[0][0]*mat[1][2] + mat[0][2]*mat[1][0])*qXZ + (mat[0][1]*mat[1][2] + mat[0][2]*mat[1][1])*qYZ;
    *qiQXZ = mat[0][0]*mat[2][0]*qXX + mat[0][1]*mat[2][1]*qYY + mat[0][2]*mat[2][2]*qZZ + (mat[0][0]*mat[2][1] + mat[0][1]*mat[2][0])*qXY + (mat[0][0]*mat[2][2] + mat[0][2]*mat[2][0])*qXZ + (mat[0][1]*mat[2][2] + mat[0][2]*mat[2][1])*qYZ;
    *qiQYZ = mat[1][0]*mat[2][0]*qXX + mat[1][1]*mat[2][1]*qYY + mat[1][2]*mat[2][2]*qZZ + (mat[1][0]*mat[2][1] + mat[1][1]*mat[2][0])*qXY + (mat[1][0]*mat[2][2] + mat[1][2]*mat[2][0])*qXZ + (mat[1][1]*mat[2][2] + mat[1][2]*mat[2][1])*qYZ;
    *qiQZZ = -*qiQXX-*qiQYY;
}

/**
 * HIPPO's damping factors of the overlap of two Slater densities with exponents aI and aJ, for the r^-1, r^-3, ...,
 * r^-11 terms. The closed form for unequal exponents loses precision to cancellation as the exponents approach each
 * other, so nearly equal exponents use its series in t = (aJ-aI)/(aI+aJ) to t^6 about the mean exponent: where
 * |t| < PAIR_DAMPING_MAX_T, or |t| < PAIR_DAMPING_MAX_T2 and |t| a r < PAIR_DAMPING_MAX_TX.
 */
DEVICE void computePairDampingFactors(real aI, real aJ, real r, real fdamp[6]) {
    real one = 1;
    if (aI == aJ) {
        real ar = aI*r, ar2 = ar*ar, ar3 = ar2*ar, ar4 = ar2*ar2, ar5 = ar3*ar2, ar6 = ar3*ar3, ar7 = ar4*ar3, ar8 = ar4*ar4;
        real expAR = EXP(-ar);
        fdamp[0] = 1 - (1 + ar*(11*one/16) + ar2*(3*one/16) + ar3*(one/48))*expAR;
        fdamp[1] = 1 - (1 + ar + ar2*(one/2) + ar3*(7*one/48) + ar4*(one/48))*expAR;
        fdamp[2] = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6) + ar4*(one/24) + ar5*(one/144))*expAR;
        fdamp[3] = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6) + ar4*(one/24) + ar5*(one/120) + ar6*(one/720))*expAR;
        fdamp[4] = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6) + ar4*(one/24) + ar5*(one/120) + ar6*(one/720) + ar7*(one/5040))*expAR;
        fdamp[5] = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6) + ar4*(one/24) + ar5*(one/120) + ar6*(one/720) + ar7*(one/5040) + ar8*(one/45360))*expAR;
        return;
    }
    real t = (aJ-aI)/(aI+aJ);
    real x = (aI+aJ)*r*(one/2);
    if (fabs(t) < PAIR_DAMPING_MAX_T || (fabs(t) < PAIR_DAMPING_MAX_T2 && fabs(t)*x < PAIR_DAMPING_MAX_TX)) {
        // f_n = 1 - exp(-x) sum_m t^(2m) sum_j SERIES[n][m][j] x^j (generated with sympy)
        const real SERIES[6][4][15] = {
    {
        {1.0, 0.6875, 0.1875, 0.020833333333333332, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.375, 0.375, 0.15625, 0.03125, 0.0020833333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, -0.0625, -0.0625, 0.0, 0.020833333333333332, 0.009895833333333333, 0.0015625, 7.440476190476191e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, -0.010416666666666666, -0.010416666666666666, -0.003125, 0.00034722222222222224, 0.000285218253968254, 3.7202380952380956e-05, 1.3778659611992946e-06, 0.0, 0.0, 0.0, 0.0, 0.0},
    },
    {
        {1.0, 1.0, 0.5, 0.14583333333333334, 0.020833333333333332, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0625, 0.0625, 0.022916666666666665, 0.0020833333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, -0.0625, -0.0625, -0.01875, 0.0020833333333333333, 0.0011160714285714285, 7.440476190476191e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.020833333333333332, 0.020833333333333332, 0.0020833333333333333, -0.004861111111111111, -0.0013640873015873015, 2.48015873015873e-05, 2.6179453262786595e-05, 1.3778659611992946e-06, 0.0, 0.0, 0.0, 0.0},
    },
    {
        {1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.006944444444444444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.005555555555555556, 0.005555555555555556, 0.0006944444444444445, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, -0.008333333333333333, -0.008333333333333333, -0.0007936507936507937, 0.000248015873015873, 2.48015873015873e-05, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.005555555555555556, 0.005555555555555556, 0.0001984126984126984, -0.000496031746031746, -4.409171075837743e-05, 5.5114638447971785e-06, 4.592886537330982e-07, 0.0, 0.0, 0.0},
    },
    {
        {1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333, 0.001388888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008333333333333334, 0.0001388888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0013492063492063493, -0.00030753968253968255, 2.9761904761904762e-05, 4.96031746031746e-06, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0010317460317460319, 0.00033730158730158733, -6.393298059964727e-05, -1.4329805996472663e-05, 5.511463844797178e-07, 9.185773074661964e-08, 0.0, 0.0},
    },
    {
        {1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333, 0.001388888888888889, 0.0001984126984126984, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.92063492063492e-05, 1.984126984126984e-05, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00014880952380952382, -5.2437641723356006e-05, 2.1258503401360544e-06, 7.086167800453515e-07, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.92063492063492e-05, 6.64525069286974e-05, -2.9919375157470394e-06, -2.362055933484505e-06, 1.3122532963802806e-08, 1.3122532963802806e-08, 0.0},
    },
    {
        {1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333, 0.001388888888888889, 0.0001984126984126984, 2.2045855379188714e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1022927689594357e-05, 1.1022927689594357e-05, 2.204585537918871e-06, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.6534391534391536e-05, -1.6534391534391536e-05, -6.062610229276896e-06, 7.873519778281683e-08, 7.873519778281683e-08, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1022927689594357e-05, 1.1022927689594357e-05, 7.71604938271605e-06, 1.9246381680244115e-07, -2.6682483693065704e-07, -4.374177654600935e-09, 1.4580592182003117e-09},
    },
};
        real t2 = t*t;
        real expX = EXP(-x);
        for (int n = 0; n < 6; n++) {
            real sum = 0;
            for (int m = 3; m >= 0; m--) {
                real poly = 0;
                for (int j = 14; j >= 0; j--)
                    poly = poly*x + SERIES[n][m][j];
                sum = sum*t2 + poly;
            }
            fdamp[n] = 1 - expX*sum;
        }
        return;
    }
    real arI = aI*r, arI2 = arI*arI, arI3 = arI2*arI, arI4 = arI2*arI2, arI5 = arI3*arI2, arI6 = arI3*arI3;
    real arJ = aJ*r, arJ2 = arJ*arJ, arJ3 = arJ2*arJ, arJ4 = arJ2*arJ2, arJ5 = arJ3*arJ2, arJ6 = arJ3*arJ3;
    real expARI = EXP(-arI), expARJ = EXP(-arJ);
    real aI2 = aI*aI, aJ2 = aJ*aJ;
    real A = aJ2/(aJ2-aI2);
    real B = aI2/(aI2-aJ2);
    real A2expARI = A*A*expARI;
    real B2expARJ = B*B*expARJ;
    fdamp[0] = 1 - (1 + 2*B + arI*(one/2))*A2expARI -
                   (1 + 2*A + arJ*(one/2))*B2expARJ;
    fdamp[1] = 1 - (1 + arI + arI2*(one/2))*A2expARI -
                   (1 + arJ + arJ2*(one/2))*B2expARJ -
                   (1 + arI)*2*B*A2expARI -
                   (1 + arJ)*2*A*B2expARJ;
    fdamp[2] = 1 - (1 + arI + arI2*(one/2) + arI3*(one/6))*A2expARI -
                   (1 + arJ + arJ2*(one/2) + arJ3*(one/6))*B2expARJ -
                   (1 + arI + arI2*(one/3))*2*B*A2expARI -
                   (1 + arJ + arJ2*(one/3))*2*A*B2expARJ;
    fdamp[3] = 1 - (1 + arI + arI2*(one/2) + arI3*(one/6) + arI4*(one/30))*A2expARI -
                   (1 + arJ + arJ2*(one/2) + arJ3*(one/6) + arJ4*(one/30))*B2expARJ -
                   (1 + arI + arI2*(2*one/5) + arI3*(one/15))*2*B*A2expARI -
                   (1 + arJ + arJ2*(2*one/5) + arJ3*(one/15))*2*A*B2expARJ;
    fdamp[4] = 1 - (1 + arI + arI2*(one/2) + arI3*(one/6) + arI4*4*(one/105) + arI5*(one/210))*A2expARI -
                   (1 + arJ + arJ2*(one/2) + arJ3*(one/6) + arJ4*4*(one/105) + arJ5*(one/210))*B2expARJ -
                   (1 + arI + arI2*(3*one/7) + arI3*(2*one/21) + arI4*(one/105))*2*B*A2expARI -
                   (1 + arJ + arJ2*(3*one/7) + arJ3*(2*one/21) + arJ4*(one/105))*2*A*B2expARJ;
    fdamp[5] = 1 - (1 + arI + arI2*(one/2) + arI3*(one/6) + arI4*(5*one/126) + arI5*(2*one/315) + arI6*(one/1890))*A2expARI -
                   (1 + arJ + arJ2*(one/2) + arJ3*(one/6) + arJ4*(5*one/126) + arJ5*(2*one/315) + arJ6*(one/1890))*B2expARJ -
                   (1 + arI + arI2*(4*one/9) + arI3*(one/9) + arI4*(one/63) + arI5*(one/945))*2*B*A2expARI -
                   (1 + arJ + arJ2*(4*one/9) + arJ3*(one/9) + arJ4*(one/63) + arJ5*(one/945))*2*A*B2expARJ;
}

/**
 * The damping factors of a single Slater density (exponent a) for the r^-1, ..., r^-9 terms of its interaction with a
 * point multipole.
 */
DEVICE void computeSingleDampingFactors(real a, real r, real* fdamp1, real* fdamp3, real* fdamp5, real* fdamp7, real* fdamp9) {
    real one = 1;
    real ar = a*r, ar2 = ar*ar, ar3 = ar2*ar, ar4 = ar2*ar2, ar5 = ar3*ar2;
    real expAR = EXP(-ar);
    *fdamp1 = 1 - (1 + ar*(one/2))*expAR;
    *fdamp3 = 1 - (1 + ar + ar2*(one/2))*expAR;
    *fdamp5 = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6))*expAR;
    *fdamp7 = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6) + ar4*(one/30))*expAR;
    *fdamp9 = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6) + ar4*(4*one/105) + ar5*(one/210))*expAR;
}

DEVICE void computeOverlapDampingFactors(real alphaI, real alphaJ, real r,
                                  real* fdampI1, real* fdampI3, real* fdampI5, real* fdampI7, real* fdampI9,
                                  real* fdampJ1, real* fdampJ3, real* fdampJ5, real* fdampJ7, real* fdampJ9,
                                  real* fdampIJ1, real* fdampIJ3, real* fdampIJ5, real* fdampIJ7, real* fdampIJ9, real* fdampIJ11) {
    computeSingleDampingFactors(alphaI, r, fdampI1, fdampI3, fdampI5, fdampI7, fdampI9);
    computeSingleDampingFactors(alphaJ, r, fdampJ1, fdampJ3, fdampJ5, fdampJ7, fdampJ9);
    real fdamp[6];
    computePairDampingFactors(alphaI, alphaJ, r, fdamp);
    *fdampIJ1 = fdamp[0];
    *fdampIJ3 = fdamp[1];
    *fdampIJ5 = fdamp[2];
    *fdampIJ7 = fdamp[3];
    *fdampIJ9 = fdamp[4];
    *fdampIJ11 = fdamp[5];
}

/**
 * Core damping: the fraction f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r) of the Slater density (exponent a) of a
 * polarized atom within r, and df/dr.
 */
DEVICE void computeCoreDampingFactor(real alpha, real r, real* f, real* dfdr) {
    real a = CORE_DAMPING_SCALE*alpha;  // the range of the core damping (InteraxMultipoleForce::setCoreDampingScale)
    real ar = a*r;
    real expAR = EXP(-ar);
    *f = 1 - (1 + ar + ar*ar*(real) 0.5)*expAR;
    *dfdr = a*ar*ar*(real) 0.5*expAR;
}
