/**
 * Compute the centroid of each molecule from the positions of its particles.
 */
KERNEL void computeCentroids(int numMolecules, GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT moleculeOffsets,
        GLOBAL const int* RESTRICT moleculeParticles, GLOBAL real4* RESTRICT centroids) {
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        int first = moleculeOffsets[molecule], last = moleculeOffsets[molecule+1];
        real3 centroid = make_real3(0);
        for (int i = first; i < last; i++) {
            real4 pos = posq[moleculeParticles[i]];
            centroid += make_real3(pos.x, pos.y, pos.z);
        }
        centroid *= 1/(real) (last-first);
        centroids[molecule] = make_real4(centroid.x, centroid.y, centroid.z, 0);
    }
}

/**
 * For each molecule A, list the molecules B > A whose centroids are within the cutoff, and record them in a bit mask.
 */
KERNEL void findNeighbors(int numMolecules, GLOBAL const real4* RESTRICT centroids, GLOBAL int* RESTRICT neighbors,
        GLOBAL int* RESTRICT numNeighbors, GLOBAL unsigned int* RESTRICT neighborMask, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    for (int a = GLOBAL_ID; a < numMolecules; a += GLOBAL_SIZE) {
        real4 centroidA = centroids[a];
        int count = 0;
        for (int word = 0; word < MASK_WORDS; word++)
            neighborMask[a*MASK_WORDS+word] = 0;
        for (int b = a+1; b < numMolecules; b++) {
            real4 centroidB = centroids[b];
            real3 delta = make_real3(centroidB.x-centroidA.x, centroidB.y-centroidA.y, centroidB.z-centroidA.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z < CUTOFF_SQUARED) {
                neighbors[a*MAX_NEIGHBORS+count++] = b;
                neighborMask[a*MASK_WORDS+b/32] |= 1u << (b%32);
            }
        }
        numNeighbors[a] = count;
    }
}

/**
 * Compute the triples. Each thread takes a pair A < B of neighbors and loops over the neighbors C > B of B that are
 * also neighbors of A. The energy is written in terms of the minimum image vectors a = AB, b = AC, c = BC: with u, v,
 * w their squared lengths, P = uvw and p = a.b, q = a.c, s = b.c, E = C9 [P^(-3/2) - 3 pqs P^(-5/2)].
 */
KERNEL void computeTriples(int numMolecules, GLOBAL const real4* RESTRICT centroids, GLOBAL const real4* RESTRICT moleculeParams,
        GLOBAL const int* RESTRICT neighbors, GLOBAL const int* RESTRICT numNeighbors, GLOBAL const unsigned int* RESTRICT neighborMask,
        GLOBAL mm_ulong* RESTRICT centroidForce, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT globals,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    const real lambda = globals[0];
    const real scaleFactor = globals[1];
    mixed energy = 0;
    for (int index = GLOBAL_ID; index < numMolecules*MAX_NEIGHBORS; index += GLOBAL_SIZE) {
        int moleculeA = index/MAX_NEIGHBORS;
        int slot = index-moleculeA*MAX_NEIGHBORS;
        if (slot >= numNeighbors[moleculeA])
            continue;
        int moleculeB = neighbors[index];
        real4 centroidA = centroids[moleculeA], centroidB = centroids[moleculeB];
        real4 paramsA = moleculeParams[moleculeA], paramsB = moleculeParams[moleculeB];
        real3 a = make_real3(centroidB.x-centroidA.x, centroidB.y-centroidA.y, centroidB.z-centroidA.z);
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(a)
#endif
        real u = dot(a, a);
        real3 forceA = make_real3(0), forceB = make_real3(0);
        bool anyTriple = false;
        int numNeighborsB = numNeighbors[moleculeB];
        for (int k = 0; k < numNeighborsB; k++) {
            int moleculeC = neighbors[moleculeB*MAX_NEIGHBORS+k];
            if ((neighborMask[moleculeA*MASK_WORDS+moleculeC/32] & (1u << (moleculeC%32))) == 0)
                continue;
            anyTriple = true;
            real4 centroidC = centroids[moleculeC];
            real4 paramsC = moleculeParams[moleculeC];
            real3 b = make_real3(centroidC.x-centroidA.x, centroidC.y-centroidA.y, centroidC.z-centroidA.z);
            real3 c = make_real3(centroidC.x-centroidB.x, centroidC.y-centroidB.y, centroidC.z-centroidB.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(b)
            APPLY_PERIODIC_TO_DELTA(c)
#endif

            // C9 from the polarizabilities (x) and frequencies (y); z flags alchemical molecules.
            real wA = paramsA.y, wB = paramsB.y, wC = paramsC.y;
            real c9 = scaleFactor*1.5f*paramsA.x*paramsB.x*paramsC.x*wA*wB*wC*(wA+wB+wC)/((wA+wB)*(wB+wC)*(wC+wA));
            if (paramsA.z != 0 || paramsB.z != 0 || paramsC.z != 0)
                c9 *= lambda;
            real v = dot(b, b), w = dot(c, c);
            real p = dot(a, b), q = dot(a, c), s = dot(b, c);
            real invP = 1/(u*v*w);
            real invP32 = invP*SQRT(invP);
            real invP52 = invP32*invP;
            real pqs = p*q*s;
            energy += c9*(invP32 - 3*pqs*invP52);
            real radial = c9*(-3*invP32 + 15*pqs*invP52);
            real cross = -3*c9*invP52;
            real3 dEda = a*(radial/u) + (b*(q*s) + c*(p*s))*cross;
            real3 dEdb = b*(radial/v) + (a*(q*s) + c*(p*q))*cross;
            real3 dEdc = c*(radial/w) + (a*(p*s) + b*(p*q))*cross;
            forceA += dEda + dEdb;
            forceB += dEdc - dEda;
            real3 forceC = -(dEdb + dEdc);
            ATOMIC_ADD(&centroidForce[moleculeC], (mm_ulong) realToFixedPoint(forceC.x));
            ATOMIC_ADD(&centroidForce[moleculeC+numMolecules], (mm_ulong) realToFixedPoint(forceC.y));
            ATOMIC_ADD(&centroidForce[moleculeC+2*numMolecules], (mm_ulong) realToFixedPoint(forceC.z));
        }
        if (anyTriple) {
            ATOMIC_ADD(&centroidForce[moleculeA], (mm_ulong) realToFixedPoint(forceA.x));
            ATOMIC_ADD(&centroidForce[moleculeA+numMolecules], (mm_ulong) realToFixedPoint(forceA.y));
            ATOMIC_ADD(&centroidForce[moleculeA+2*numMolecules], (mm_ulong) realToFixedPoint(forceA.z));
            ATOMIC_ADD(&centroidForce[moleculeB], (mm_ulong) realToFixedPoint(forceB.x));
            ATOMIC_ADD(&centroidForce[moleculeB+numMolecules], (mm_ulong) realToFixedPoint(forceB.y));
            ATOMIC_ADD(&centroidForce[moleculeB+2*numMolecules], (mm_ulong) realToFixedPoint(forceB.z));
        }
    }
    energyBuffer[GLOBAL_ID] += energy;
}

/**
 * Share the force on each centroid equally among the particles of its molecule, and clear the centroid forces for the
 * next evaluation.
 */
KERNEL void applyCentroidForces(int numMolecules, GLOBAL mm_long* RESTRICT centroidForce, GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL const int* RESTRICT moleculeOffsets, GLOBAL const int* RESTRICT moleculeParticles) {
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        int first = moleculeOffsets[molecule], last = moleculeOffsets[molecule+1];
        real scale = 1/(real) (last-first);
        real3 force = make_real3(centroidForce[molecule], centroidForce[molecule+numMolecules], centroidForce[molecule+2*numMolecules])*(scale/(real) 0x100000000);
        centroidForce[molecule] = 0;
        centroidForce[molecule+numMolecules] = 0;
        centroidForce[molecule+2*numMolecules] = 0;
        for (int i = first; i < last; i++) {
            int atom = moleculeParticles[i];
            ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(force.x));
            ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
            ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
        }
    }
}
