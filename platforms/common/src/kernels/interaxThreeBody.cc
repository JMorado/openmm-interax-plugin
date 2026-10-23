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
 * For each site A, list in increasing order the sites B of other molecules within the cutoff, with the minimum image
 * vectors from A to them. At most maxNeighbors are stored, but numNeighbors holds the full count: computeTriples()
 * scans all the sites for a site whose list overflowed, so the result never depends on the size of the lists. The
 * largest count is recorded in maxNeighborCount, which the host reads now and then to enlarge the lists.
 */
KERNEL void findNeighbors(int numSites, int maxNeighbors, GLOBAL const real4* RESTRICT sites, GLOBAL const int* RESTRICT siteMolecules,
        GLOBAL int* RESTRICT neighbors, GLOBAL real4* RESTRICT neighborDeltas, GLOBAL int* RESTRICT numNeighbors,
        GLOBAL int* RESTRICT maxNeighborCount, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX,
        real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    for (int a = GLOBAL_ID; a < numSites; a += GLOBAL_SIZE) {
        real4 siteA = sites[a];
        int moleculeA = siteMolecules[a];
        int count = 0;
        for (int b = 0; b < numSites; b++) {
            if (siteMolecules[b] == moleculeA)
                continue;
            real4 siteB = sites[b];
            real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z < CUTOFF_SQUARED) {
                if (count < maxNeighbors) {
                    neighbors[a*maxNeighbors+count] = b;
                    neighborDeltas[a*maxNeighbors+count] = make_real4(delta.x, delta.y, delta.z, 0);
                }
                count++;
            }
        }
        numNeighbors[a] = count;
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

// The molecule recorded for a candidate that is not a neighbor of A (when scanning all the sites after an overflow).
#define NOT_A_NEIGHBOR (-2147483647-1)

/**
 * Load the index'th candidate neighbor of site A into local memory: the index'th entry of its neighbor list, or, if
 * the list overflowed, the index'th site, marked NOT_A_NEIGHBOR unless it is a neighbor of A.
 */
DEVICE void loadNeighbor(int siteA, int index, bool overflow, int maxNeighbors, int moleculeA, real4 siteA4, GLOBAL const real4* RESTRICT sites,
        GLOBAL const real4* RESTRICT siteParams, GLOBAL const int* RESTRICT siteMolecules, GLOBAL const int* RESTRICT neighbors,
        GLOBAL const real4* RESTRICT neighborDeltas, LOCAL_ARG real4* delta, LOCAL_ARG real4* params, LOCAL_ARG int* molecule, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    if (!overflow) {
        const int site = neighbors[siteA*maxNeighbors+index];
        *delta = neighborDeltas[siteA*maxNeighbors+index];
        *params = siteParams[site];
        *molecule = siteMolecules[site];
        return;
    }
    const real4 siteB = sites[index];
    real3 d = make_real3(siteB.x-siteA4.x, siteB.y-siteA4.y, siteB.z-siteA4.z);
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(d)
#endif
    const int moleculeB = siteMolecules[index];
    *delta = make_real4(d.x, d.y, d.z, 0);
    *params = siteParams[index];
    *molecule = (moleculeB != moleculeA && dot(d, d) < CUTOFF_SQUARED ? moleculeB : NOT_A_NEIGHBOR);
}

/**
 * Compute the triples. Every triple is computed once from each of its three sites, which only accumulates the force on
 * its own site and a third of the energy, so no atomic additions are needed per triple. Each thread block takes a site
 * A and loops over the pairs (B, C) of its neighbors: tiles of TILE_SIZE neighbors (their vectors from A, parameters
 * and molecules) are loaded into local memory, and the threads share out the pairs of each pair of tiles. If the
 * neighbor list of A overflowed, the candidates are all the sites instead, and those that are not neighbors of A
 * are skipped. A triple
 * interacts if C is also within the cutoff of B and in a different molecule. With the minimum image vectors a = AB,
 * b = AC (from the neighbor list) and c = BC (the minimum image of b - a), u, v, w their squared lengths, P = uvw and
 * p = a.b, q = a.c, s = b.c, E = C9 [P^(-3/2) - 3 pqs P^(-5/2)], times the damping of the three pairs when
 * USE_DAMPING is defined.
 */
KERNEL void computeTriples(int numSites, int maxNeighbors, GLOBAL const real4* RESTRICT sites, GLOBAL const real4* RESTRICT siteParams,
        GLOBAL const int* RESTRICT siteMolecules,
        GLOBAL const int* RESTRICT neighbors, GLOBAL const real4* RESTRICT neighborDeltas, GLOBAL const int* RESTRICT numNeighbors,
        GLOBAL mm_ulong* RESTRICT siteForce, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT globals, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    LOCAL real4 tileDeltaJ[TILE_SIZE], tileDeltaK[TILE_SIZE], tileParamsJ[TILE_SIZE], tileParamsK[TILE_SIZE];
    LOCAL int tileMoleculeJ[TILE_SIZE], tileMoleculeK[TILE_SIZE];
    LOCAL real3 reduction[TRIPLE_BLOCK_SIZE];
    LOCAL mixed energyReduction[TRIPLE_BLOCK_SIZE];
    const real lambda = globals[0];
    const real scaleFactor = globals[1];
#ifdef USE_DAMPING
    const real dampingScale = globals[2];
#endif
    mixed energy = 0;
    for (int siteA = GROUP_ID; siteA < numSites; siteA += NUM_GROUPS) {
        const bool overflow = (numNeighbors[siteA] > maxNeighbors);
        const int numNeighborsA = (overflow ? numSites : numNeighbors[siteA]);
        const real4 paramsA = siteParams[siteA];
        const real4 siteA4 = sites[siteA];
        const int moleculeA = siteMolecules[siteA];
        const bool alchemicalA = (paramsA.w != 0);
        real3 forceA = make_real3(0);
        for (int startJ = 0; startJ < numNeighborsA; startJ += TILE_SIZE) {
            const int sizeJ = min(TILE_SIZE, numNeighborsA-startJ);
            for (int startK = startJ; startK < numNeighborsA; startK += TILE_SIZE) {
                const int sizeK = min(TILE_SIZE, numNeighborsA-startK);

                // Load the two tiles of neighbors.

                SYNC_THREADS;
                for (int i = LOCAL_ID; i < TILE_SIZE; i += LOCAL_SIZE) {
                    if (i < sizeJ)
                        loadNeighbor(siteA, startJ+i, overflow, maxNeighbors, moleculeA, siteA4, sites, siteParams, siteMolecules, neighbors, neighborDeltas,
                                &tileDeltaJ[i], &tileParamsJ[i], &tileMoleculeJ[i], periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY,
                                periodicBoxVecZ);
                    if (i < sizeK)
                        loadNeighbor(siteA, startK+i, overflow, maxNeighbors, moleculeA, siteA4, sites, siteParams, siteMolecules, neighbors, neighborDeltas,
                                &tileDeltaK[i], &tileParamsK[i], &tileMoleculeK[i], periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY,
                                periodicBoxVecZ);
                }
                SYNC_THREADS;

                // Compute the triples of the pairs (B, C) with B in the first tile and C after it in the second.

                for (int pair = LOCAL_ID; pair < TILE_SIZE*TILE_SIZE; pair += LOCAL_SIZE) {
                    const int j = pair/TILE_SIZE, k = pair-j*TILE_SIZE;
                    if (j >= sizeJ || k >= sizeK || startK+k <= startJ+j || tileMoleculeJ[j] == tileMoleculeK[k] || tileMoleculeJ[j] == NOT_A_NEIGHBOR ||
                            tileMoleculeK[k] == NOT_A_NEIGHBOR)
                        continue;
                    const real3 a = trimTo3(tileDeltaJ[j]);
                    const real3 b = trimTo3(tileDeltaK[k]);
                    real3 c = b-a;
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(c)
#endif
                    const real w = dot(c, c);
                    if (w >= CUTOFF_SQUARED)
                        continue;
                    const real4 paramsB = tileParamsJ[j], paramsC = tileParamsK[k];

                    // C9 from the polarizabilities (x) and frequencies (y); w flags alchemical sites.
                    const real wA = paramsA.y, wB = paramsB.y, wC = paramsC.y;
                    real c9 = scaleFactor*1.5f*paramsA.x*paramsB.x*paramsC.x*wA*wB*wC*(wA+wB+wC)/((wA+wB)*(wB+wC)*(wC+wA));
                    if (alchemicalA || paramsB.w != 0 || paramsC.w != 0)
                        c9 *= lambda;
                    const real u = dot(a, a), v = dot(b, b);
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
#ifdef USE_DAMPING
                    // E = E0 fa fb fc with the damping of each pair (z holds the widths). Only the gradients with
                    // respect to a and b move site A.
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
                    tripleEnergy *= damping;
#endif
                    energy += tripleEnergy;
                    forceA += dEda + dEdb;
                }
            }
        }

        // Sum the force on A over the threads of the block.

        reduction[LOCAL_ID] = forceA;
        SYNC_THREADS;
        for (int step = LOCAL_SIZE/2; step > 0; step /= 2) {
            if (LOCAL_ID < step)
                reduction[LOCAL_ID] += reduction[LOCAL_ID+step];
            SYNC_THREADS;
        }
        if (LOCAL_ID == 0) {
            const real3 force = reduction[0];
            ATOMIC_ADD(&siteForce[siteA], (mm_ulong) realToFixedPoint(force.x));
            ATOMIC_ADD(&siteForce[siteA+numSites], (mm_ulong) realToFixedPoint(force.y));
            ATOMIC_ADD(&siteForce[siteA+2*numSites], (mm_ulong) realToFixedPoint(force.z));
        }
    }
    // Sum the energy over the threads of the block. Every triple was computed three times.

    energyReduction[LOCAL_ID] = energy;
    SYNC_THREADS;
    for (int step = LOCAL_SIZE/2; step > 0; step /= 2) {
        if (LOCAL_ID < step)
            energyReduction[LOCAL_ID] += energyReduction[LOCAL_ID+step];
        SYNC_THREADS;
    }
    if (LOCAL_ID == 0)
        energyBuffer[GROUP_ID] += energyReduction[0]*(mixed) (1.0/3.0);
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
