// OpenMM's common kernels define ATOMIC_ADD but not ATOMIC_MAX.
#ifdef __CUDACC__
    #define ATOMIC_MAX(dest, value) atomicMax(dest, value)
#else
    #define ATOMIC_MAX(dest, value) atomic_max(dest, value)
#endif

/**
 * Compute the position of each site: the unweighted centroid of its particles. This also clears the largest number
 * of neighbors, which findNeighbors() records.
 */
KERNEL void computeSites(int numSites, GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT siteOffsets,
        GLOBAL const int* RESTRICT siteParticles, GLOBAL real4* RESTRICT sites, GLOBAL int* RESTRICT maxNeighborCount) {
    if (GLOBAL_ID == 0)
        *maxNeighborCount = 0;
    for (int site = GLOBAL_ID; site < numSites; site += GLOBAL_SIZE) {
        int first = siteOffsets[site], last = siteOffsets[site+1];
        real3 centroid = make_real3(0);
        for (int i = first; i < last; i++) {
            real4 pos = posq[siteParticles[i]];
            centroid += make_real3(pos.x, pos.y, pos.z);
        }
        centroid *= 1/(real) (last-first);
        sites[site] = make_real4(centroid.x, centroid.y, centroid.z, 0);
    }
}

/**
 * For each site A, list in increasing order the sites B > A of other molecules within the cutoff. At most
 * maxNeighbors are stored; the largest number found is recorded in maxNeighborCount, so that the host can enlarge
 * the list and repeat the evaluation when it overflows.
 */
KERNEL void findNeighbors(int numSites, int maxNeighbors, GLOBAL const real4* RESTRICT sites, GLOBAL const int* RESTRICT siteMolecules,
        GLOBAL int* RESTRICT neighbors, GLOBAL int* RESTRICT numNeighbors, GLOBAL int* RESTRICT maxNeighborCount, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    for (int a = GLOBAL_ID; a < numSites; a += GLOBAL_SIZE) {
        real4 siteA = sites[a];
        int moleculeA = siteMolecules[a];
        int count = 0;
        for (int b = a+1; b < numSites; b++) {
            if (siteMolecules[b] == moleculeA)
                continue;
            real4 siteB = sites[b];
            real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z < CUTOFF_SQUARED) {
                if (count < maxNeighbors)
                    neighbors[a*maxNeighbors+count] = b;
                count++;
            }
        }
        numNeighbors[a] = min(count, maxNeighbors);
        ATOMIC_MAX(maxNeighborCount, count);
    }
}

#ifdef USE_DAMPING
/**
 * The Tang-Toennies damping f_3(x) of a dipole propagator and its derivative. For x < 2 the closed form cancels, and
 * f_3 = exp(-x) sum_{k>3} x^k/k! instead.
 */
DEVICE real tangToenniesDamping3(real x, real* derivative) {
    const real expX = EXP(-x);
    const real term3 = x*x*x*(real) (1.0/6.0);
    *derivative = expX*term3;
    if (x < 2) {
        real term = term3, tail = 0;
        for (int k = 4; k <= 24; k++) {
            term *= x/k;
            tail += term;
        }
        return expX*tail;
    }
    return 1-expX*(1+x+x*x*(real) 0.5+term3);
}
#endif

/**
 * Compute the triples. Each thread takes a pair A < B of neighbors and visits the sites C > B that are neighbors of
 * both, by merging the sorted neighbor lists of A (after B) and of B. The energy is written in terms of the minimum
 * image vectors a = AB, b = AC, c = BC: with u, v, w their squared lengths, P = uvw and p = a.b, q = a.c, s = b.c,
 * E = C9 [P^(-3/2) - 3 pqs P^(-5/2)], times the damping of the three pairs when USE_DAMPING is defined.
 */
KERNEL void computeTriples(int numSites, int maxNeighbors, GLOBAL const real4* RESTRICT sites, GLOBAL const real4* RESTRICT siteParams,
        GLOBAL const int* RESTRICT neighbors, GLOBAL const int* RESTRICT numNeighbors, GLOBAL mm_ulong* RESTRICT siteForce,
        GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT globals, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    const real lambda = globals[0];
    const real scaleFactor = globals[1];
#ifdef USE_DAMPING
    const real dampingScale = globals[2];
#endif
    mixed energy = 0;
    for (int index = GLOBAL_ID; index < numSites*maxNeighbors; index += GLOBAL_SIZE) {
        const int siteA = index/maxNeighbors;
        const int slot = index-siteA*maxNeighbors;
        const int numNeighborsA = numNeighbors[siteA];
        if (slot >= numNeighborsA)
            continue;
        const int siteB = neighbors[index];
        const real4 posA = sites[siteA], posB = sites[siteB];
        const real4 paramsA = siteParams[siteA], paramsB = siteParams[siteB];
        real3 a = make_real3(posB.x-posA.x, posB.y-posA.y, posB.z-posA.z);
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(a)
#endif
        const real u = dot(a, a);
        real3 forceA = make_real3(0), forceB = make_real3(0);
        bool anyTriple = false;
        const GLOBAL int* neighborsA = &neighbors[siteA*maxNeighbors];
        const GLOBAL int* neighborsB = &neighbors[siteB*maxNeighbors];
        const int numNeighborsB = numNeighbors[siteB];
        int i = slot+1, j = 0;
        while (i < numNeighborsA && j < numNeighborsB) {
            const int candidateA = neighborsA[i], candidateB = neighborsB[j];
            if (candidateA != candidateB) {
                if (candidateA < candidateB)
                    i++;
                else
                    j++;
                continue;
            }
            const int siteC = candidateA;
            i++;
            j++;
            anyTriple = true;
            const real4 posC = sites[siteC];
            const real4 paramsC = siteParams[siteC];
            real3 b = make_real3(posC.x-posA.x, posC.y-posA.y, posC.z-posA.z);
            real3 c = make_real3(posC.x-posB.x, posC.y-posB.y, posC.z-posB.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(b)
            APPLY_PERIODIC_TO_DELTA(c)
#endif

            // C9 from the polarizabilities (x) and frequencies (y); w flags alchemical sites.
            const real wA = paramsA.y, wB = paramsB.y, wC = paramsC.y;
            real c9 = scaleFactor*1.5f*paramsA.x*paramsB.x*paramsC.x*wA*wB*wC*(wA+wB+wC)/((wA+wB)*(wB+wC)*(wC+wA));
            if (paramsA.w != 0 || paramsB.w != 0 || paramsC.w != 0)
                c9 *= lambda;
            const real v = dot(b, b), w = dot(c, c);
            const real p = dot(a, b), q = dot(a, c), s = dot(b, c);
            const real invP = 1/(u*v*w);
            const real invP32 = invP*SQRT(invP);
            const real invP52 = invP32*invP;
            const real pqs = p*q*s;
            real tripleEnergy = c9*(invP32 - 3*pqs*invP52);
            const real radial = c9*(-3*invP32 + 15*pqs*invP52);
            const real cross = -3*c9*invP52;
            real3 dEda = a*(radial/u) + (b*(q*s) + c*(p*s))*cross;
            real3 dEdb = b*(radial/v) + (a*(q*s) + c*(p*q))*cross;
            real3 dEdc = c*(radial/w) + (a*(p*s) + b*(p*q))*cross;
#ifdef USE_DAMPING
            // E = E0 fa fb fc with the damping of each pair (z holds the widths).
            const real lengthA = SQRT(u), lengthB = SQRT(v), lengthC = SQRT(w);
            const real rateA = 2*dampingScale/(paramsA.z+paramsB.z);
            const real rateB = 2*dampingScale/(paramsA.z+paramsC.z);
            const real rateC = 2*dampingScale/(paramsB.z+paramsC.z);
            real dfA, dfB, dfC;
            const real fA = tangToenniesDamping3(rateA*lengthA, &dfA);
            const real fB = tangToenniesDamping3(rateB*lengthB, &dfB);
            const real fC = tangToenniesDamping3(rateC*lengthC, &dfC);
            const real damping = fA*fB*fC;
            dEda = dEda*damping + a*(tripleEnergy*rateA*dfA*fB*fC/lengthA);
            dEdb = dEdb*damping + b*(tripleEnergy*fA*rateB*dfB*fC/lengthB);
            dEdc = dEdc*damping + c*(tripleEnergy*fA*fB*rateC*dfC/lengthC);
            tripleEnergy *= damping;
#endif
            energy += tripleEnergy;
            forceA += dEda + dEdb;
            forceB += dEdc - dEda;
            const real3 forceC = -(dEdb + dEdc);
            ATOMIC_ADD(&siteForce[siteC], (mm_ulong) realToFixedPoint(forceC.x));
            ATOMIC_ADD(&siteForce[siteC+numSites], (mm_ulong) realToFixedPoint(forceC.y));
            ATOMIC_ADD(&siteForce[siteC+2*numSites], (mm_ulong) realToFixedPoint(forceC.z));
        }
        if (anyTriple) {
            ATOMIC_ADD(&siteForce[siteA], (mm_ulong) realToFixedPoint(forceA.x));
            ATOMIC_ADD(&siteForce[siteA+numSites], (mm_ulong) realToFixedPoint(forceA.y));
            ATOMIC_ADD(&siteForce[siteA+2*numSites], (mm_ulong) realToFixedPoint(forceA.z));
            ATOMIC_ADD(&siteForce[siteB], (mm_ulong) realToFixedPoint(forceB.x));
            ATOMIC_ADD(&siteForce[siteB+numSites], (mm_ulong) realToFixedPoint(forceB.y));
            ATOMIC_ADD(&siteForce[siteB+2*numSites], (mm_ulong) realToFixedPoint(forceB.z));
        }
    }
    energyBuffer[GLOBAL_ID] += energy;
}

/**
 * Share the force on each site equally among its particles, and clear the site forces for the next evaluation.
 */
KERNEL void applySiteForces(int numSites, GLOBAL mm_long* RESTRICT siteForce, GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL const int* RESTRICT siteOffsets, GLOBAL const int* RESTRICT siteParticles) {
    for (int site = GLOBAL_ID; site < numSites; site += GLOBAL_SIZE) {
        int first = siteOffsets[site], last = siteOffsets[site+1];
        real scale = 1/(real) (last-first);
        real3 force = make_real3(siteForce[site], siteForce[site+numSites], siteForce[site+2*numSites])*(scale/(real) 0x100000000);
        siteForce[site] = 0;
        siteForce[site+numSites] = 0;
        siteForce[site+2*numSites] = 0;
        for (int i = first; i < last; i++) {
            int atom = siteParticles[i];
            ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(force.x));
            ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
            ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
        }
    }
}
