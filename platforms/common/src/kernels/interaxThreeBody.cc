// OpenMM's common kernels define ATOMIC_ADD but not ATOMIC_MAX.
#define ATOMIC_MAX(dest, value) atomicMax(dest, value)

/**
 * Compute the position of each site: the unweighted centroid of its particles. This also clears the largest number
 * of neighbors, which findNeighbors() records, and requests a rebuild of the neighbor lists when a site has moved by
 * more than half the skin since the last build (any jump, e.g. after the atoms are reordered, counts as a move).
 */
KERNEL void computeSites(int numSites, GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT siteOffsets,
        GLOBAL const int* RESTRICT siteParticles, GLOBAL real4* RESTRICT sites, GLOBAL int* RESTRICT maxNeighborCount,
        GLOBAL const real4* RESTRICT sitesAtBuild, GLOBAL int* RESTRICT rebuildFlag) {
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
        const real4 old = sitesAtBuild[site];
        const real3 moved = make_real3(centroid.x-old.x, centroid.y-old.y, centroid.z-old.z);
        if (dot(moved, moved) > HALF_SKIN_SQUARED)
            *rebuildFlag = 1;
    }
}

/**
 * The bucket of a neighbor at vector d from site A: a 4x4x4 grid over the cube of side 2 LIST_CUTOFF centered on A, in
 * Morton (Z) order, so that consecutive buckets are adjacent in space.
 */
DEVICE int neighborBucket(real3 d) {
    const real scale = (real) (2.0/LIST_CUTOFF);  // 4 cells over 2 LIST_CUTOFF
    const int ix = min(3, max(0, (int) ((d.x+(real) LIST_CUTOFF)*scale)));
    const int iy = min(3, max(0, (int) ((d.y+(real) LIST_CUTOFF)*scale)));
    const int iz = min(3, max(0, (int) ((d.z+(real) LIST_CUTOFF)*scale)));
    return (ix&1) | ((iy&1)<<1) | ((iz&1)<<2) | ((ix&2)<<2) | ((iy&2)<<3) | ((iz&2)<<4);
}

/**
 * For each site A, list the sites B > A of other molecules within the cutoff plus the skin (a half list: each triple
 * is then found once, from its lowest site), bucket by bucket (neighborBucket()), each bucket in increasing order. The lists are rebuilt only when computeSites()
 * flagged a move larger than half the skin; otherwise they are kept, and the triples recompute every distance from the
 * current positions and apply the exact cutoff. At most maxNeighbors are stored, but numNeighbors holds the full count:
 * computeTriples() scans all the higher sites for a site whose list overflowed, so the result never depends on the
 * size of the lists. The largest count is recorded in maxNeighborCount, which the host reads now and then to enlarge
 * the lists.
 */
KERNEL void findNeighbors(int numSites, int maxNeighbors, GLOBAL const real4* RESTRICT sites, GLOBAL const int* RESTRICT siteMolecules,
        GLOBAL int* RESTRICT neighbors, GLOBAL int* RESTRICT numNeighbors, GLOBAL int* RESTRICT maxNeighborCount,
        GLOBAL real4* RESTRICT sitesAtBuild, GLOBAL const int* RESTRICT rebuildFlag, real4 periodicBoxSize,
        real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    const bool rebuild = (*rebuildFlag != 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // A warp per site: the candidates b > a are scanned 32 at a time. The list is the same, in the same order, as the
    // serial version below: within a bucket, the lanes with the same bucket are ranked by lane (__match_any_sync), and the
    // batches come in increasing b.
    LOCAL int bucketCount[NEIGHBOR_BLOCK_SIZE/32][64];
    const int lane = LOCAL_ID%32, warpInBlock = LOCAL_ID/32;
    LOCAL_ARG int* count64 = bucketCount[warpInBlock];
    for (int a = GLOBAL_ID/32; a < numSites; a += GLOBAL_SIZE/32) {
        if (!rebuild) {
            if (lane == 0)
                ATOMIC_MAX(maxNeighborCount, numNeighbors[a]);
            continue;
        }
        const real4 siteA = sites[a];
        const int moleculeA = siteMolecules[a];
        count64[lane] = 0;
        count64[lane+32] = 0;
        __syncwarp();
        for (int base = a+1; base < numSites; base += 32) {
            const int b = base+lane;
            int bucket = -1;
            if (b < numSites && siteMolecules[b] != moleculeA) {
                const real4 siteB = sites[b];
                real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                if (dot(delta, delta) < LIST_CUTOFF_SQUARED)
                    bucket = neighborBucket(delta);
            }
            if (bucket >= 0)
                atomicAdd(&count64[bucket], 1);
        }
        __syncwarp();
        // Exclusive prefix sum of the 64 bucket counts (two per lane) and the total.
        const int c0 = count64[2*lane], c1 = count64[2*lane+1];
        int sum = c0+c1;
        for (int offset = 1; offset < 32; offset *= 2) {
            const int other = __shfl_up_sync(0xffffffff, sum, offset);
            if (lane >= offset)
                sum += other;
        }
        const int total = __shfl_sync(0xffffffff, sum, 31);
        const int first = sum-c0-c1;
        __syncwarp();
        count64[2*lane] = first;
        count64[2*lane+1] = first+c0;
        __syncwarp();
        if (lane == 0) {
            numNeighbors[a] = total;
            sitesAtBuild[a] = siteA;
            ATOMIC_MAX(maxNeighborCount, total);
        }
        if (total > maxNeighbors)
            continue;  // computeTriples() scans all the higher sites instead
        for (int base = a+1; base < numSites; base += 32) {
            const int b = base+lane;
            int bucket = -1;
            if (b < numSites && siteMolecules[b] != moleculeA) {
                const real4 siteB = sites[b];
                real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                if (dot(delta, delta) < LIST_CUTOFF_SQUARED)
                    bucket = neighborBucket(delta);
            }
            const unsigned int peers = __match_any_sync(0xffffffff, bucket);
            const int rank = __popc(peers & ((1u<<lane)-1));
            const int offset = (bucket >= 0 ? count64[bucket] : 0);
            __syncwarp();
            if (bucket >= 0) {
                neighbors[a*maxNeighbors+offset+rank] = b;
                if (rank == 0)
                    count64[bucket] = offset+__popc(peers);
            }
            __syncwarp();
        }
    }
#else
    for (int a = GLOBAL_ID; a < numSites; a += GLOBAL_SIZE) {
        if (!rebuild) {
            ATOMIC_MAX(maxNeighborCount, numNeighbors[a]);
            continue;
        }
        real4 siteA = sites[a];
        int moleculeA = siteMolecules[a];

        // Count the neighbors in each bucket.

        int start[64];
        for (int i = 0; i < 64; i++)
            start[i] = 0;
        int count = 0;
        for (int b = a+1; b < numSites; b++) {
            if (siteMolecules[b] == moleculeA)
                continue;
            real4 siteB = sites[b];
            real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            if (dot(delta, delta) < LIST_CUTOFF_SQUARED) {
                start[neighborBucket(delta)]++;
                count++;
            }
        }
        numNeighbors[a] = count;
        sitesAtBuild[a] = siteA;
        ATOMIC_MAX(maxNeighborCount, count);
        if (count > maxNeighbors)
            continue;  // computeTriples() scans all the higher sites instead

        // Store the neighbors bucket by bucket (each bucket in increasing order of index).

        int offset = 0;
        for (int i = 0; i < 64; i++) {
            const int n = start[i];
            start[i] = offset;
            offset += n;
        }
        for (int b = a+1; b < numSites; b++) {
            if (siteMolecules[b] == moleculeA)
                continue;
            real4 siteB = sites[b];
            real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            if (dot(delta, delta) < LIST_CUTOFF_SQUARED)
                neighbors[a*maxNeighbors+start[neighborBucket(delta)]++] = b;
        }
    }
#endif
}

/**
 * At every evaluation, copy from the (skin) neighbor list of each site A the neighbors now within the cutoff, in the
 * same order, to activeNeighbors, their number to numActive (-1 if the list overflowed: computeTriples() then scans
 * all the higher sites), and the bounding box (vectors from A) of each tile of TILE_SIZE consecutive active neighbors
 * to tileBoxes. The triples then loop over the neighbors within the cutoff only. On CUDA a warp handles a site and
 * compacts 32 candidates at a time with a ballot; elsewhere a thread handles a site.
 */
KERNEL void compactNeighbors(int numSites, int maxNeighbors, GLOBAL const real4* RESTRICT sites, GLOBAL const int* RESTRICT neighbors,
        GLOBAL const int* RESTRICT numNeighbors, GLOBAL int* RESTRICT activeNeighbors, GLOBAL int* RESTRICT numActive,
        GLOBAL real4* RESTRICT tileBoxes, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY,
        real4 periodicBoxVecZ) {
    const int maxTiles = (maxNeighbors+TILE_SIZE-1)/TILE_SIZE;
    const int lane = LOCAL_ID%32;
    const int firstWarp = GLOBAL_ID/32, numWarps = GLOBAL_SIZE/32;
    for (int a = firstWarp; a < numSites; a += numWarps) {
        const int count = numNeighbors[a];
        if (count > maxNeighbors) {
            if (lane == 0)
                numActive[a] = -1;
            continue;
        }
        const real4 siteA = sites[a];
        GLOBAL const int* list = &neighbors[a*maxNeighbors];
        GLOBAL int* active = &activeNeighbors[a*maxNeighbors];
        int n = 0;
        for (int base = 0; base < count; base += 32) {
            const int i = base+lane;
            int b = 0;
            bool keep = false;
            if (i < count) {
                b = list[i];
                const real4 siteB = sites[b];
                real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                keep = (dot(delta, delta) < CUTOFF_SQUARED);
            }
            const unsigned int mask = __ballot_sync(0xffffffff, keep);
            if (keep)
                active[n+__popc(mask & ((1u<<lane)-1))] = b;
            n += __popc(mask);
        }
        __syncwarp();
        for (int t = 0; t*TILE_SIZE < n; t++) {
            real3 lo = make_real3((real) 1e30), hi = make_real3((real) -1e30);
            for (int i = t*TILE_SIZE+lane; i < min(n, (t+1)*TILE_SIZE); i += 32) {
                const real4 siteB = sites[active[i]];
                real3 delta = make_real3(siteB.x-siteA.x, siteB.y-siteA.y, siteB.z-siteA.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                lo = make_real3(min(lo.x, delta.x), min(lo.y, delta.y), min(lo.z, delta.z));
                hi = make_real3(max(hi.x, delta.x), max(hi.y, delta.y), max(hi.z, delta.z));
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                lo.x = min(lo.x, __shfl_xor_sync(0xffffffff, lo.x, offset));
                lo.y = min(lo.y, __shfl_xor_sync(0xffffffff, lo.y, offset));
                lo.z = min(lo.z, __shfl_xor_sync(0xffffffff, lo.z, offset));
                hi.x = max(hi.x, __shfl_xor_sync(0xffffffff, hi.x, offset));
                hi.y = max(hi.y, __shfl_xor_sync(0xffffffff, hi.y, offset));
                hi.z = max(hi.z, __shfl_xor_sync(0xffffffff, hi.z, offset));
            }
            if (lane == 0) {
                tileBoxes[2*(a*maxTiles+t)] = make_real4(lo.x, lo.y, lo.z, 0);
                tileBoxes[2*(a*maxTiles+t)+1] = make_real4(hi.x, hi.y, hi.z, 0);
            }
        }
        if (lane == 0)
            numActive[a] = n;
    }
}

/**
 * The smallest distance along one axis between the intervals [lo1, hi1] and [lo2, hi2], or the images of the second
 * shifted by one box length L. When L >= 2 CUTOFF and the vectors are within the cutoff, these are all the images the
 * minimum image of a vector between them can reach.
 */
DEVICE real intervalGap(real lo1, real hi1, real lo2, real hi2, real L, bool periodic) {
    real g = max((real) 0, max(lo2-hi1, lo1-hi2));
    if (periodic) {
        g = min(g, max((real) 0, max(lo2+L-hi1, lo1-hi2-L)));
        g = min(g, max((real) 0, max(lo2-L-hi1, lo1-hi2+L)));
    }
    return g;
}

#ifdef USE_DAMPING
/**
 * The Tang-Toennies damping f_3(x) of a dipole propagator and its derivative. For x < 2 the closed form cancels, and
 * f_3 = exp(-x) sum_{k>3} x^k/k! instead.
 */
DEVICE real tangToenniesDamping3(real x, real* derivative) {
    // Beyond TT_SKIP, exp(-x) sum_{k<=3} x^k/k! is below the rounding of 1 (and so is the derivative): skip the exp.
    if (x > (real) TT_SKIP) {
        *derivative = 0;
        return 1;
    }
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

#ifdef USE_CENTROID_DAMPING
/**
 * The centroid damping g = 1 - exp(-x^6) of a pair of sites, x = r/R0, and its derivative dg/dx = 6 x^5 exp(-x^6).
 * For x^6 < 0.1 the closed form cancels, and the series of 1 - exp(-y) is used instead (relative error < 3e-15).
 */
DEVICE real centroidDamping(real x, real* derivative) {
    const real x3 = x*x*x;
    const real y = x3*x3;
    const real expY = EXP(-y);
    *derivative = 6*x3*x*x*expY;
    if (y < (real) 0.1) {
        real series = 1;
        for (int k = 9; k >= 2; k--)
            series = 1-series*y/k;
        return y*series;
    }
    return 1-expY;
}
#endif

/**
 * The damping f of one pair at distance r (the product of the Tang-Toennies and centroid dampings that are enabled)
 * and (df/dr)/r, which multiplies the pair vector in the gradient.
 */
DEVICE real pairDampingRate(real r, real invR, real rate, real invDampingRadius, real* dfdrOverR) {
    real f = 1, dfdr = 0;
#ifdef USE_DAMPING
    {
        real df;
        f = tangToenniesDamping3(rate*r, &df);
        dfdr = rate*df;
    }
#endif
#ifdef USE_CENTROID_DAMPING
    {
        real dg;
        const real g = centroidDamping(r*invDampingRadius, &dg);
        dfdr = dfdr*g + f*invDampingRadius*dg;
        f *= g;
    }
#endif
    *dfdrOverR = dfdr*invR;
    return f;
}

DEVICE real pairDamping(real r, real widthSum, real dampingScale, real invDampingRadius, real* dfdrOverR) {
    return pairDampingRate(r, 1/r, 2*dampingScale/widthSum, invDampingRadius, dfdrOverR);
}

// The molecule recorded for a candidate that is not a neighbor of A (when scanning all the sites after an overflow).
#define NOT_A_NEIGHBOR (-2147483647-1)

/**
 * Load the index'th candidate neighbor of site A into local memory: the index'th entry of its neighbor list, or, if
 * the list overflowed, the site A+1+index. The vector from A is recomputed from the current positions (the lists
 * are only rebuilt now and then), and a candidate of the same molecule or beyond the cutoff is marked NOT_A_NEIGHBOR.
 * With damping, the damping of the pair A-B and its (df/dr)/r are stored as well (once per neighbor, not per triple).
 * The parameters stored are (g, w, width, alchemical) with g = alpha w / (w_A + w), so that
 * C9 = 1.5 alpha_A w_A g_B g_C (w_A + w_B + w_C) / (w_B + w_C).
 */
DEVICE void loadNeighbor(int siteA, int index, bool overflow, int maxNeighbors, int moleculeA, real4 siteA4, real widthA, real frequencyA, GLOBAL const real4* RESTRICT sites,
        GLOBAL const real4* RESTRICT siteParams, GLOBAL const int* RESTRICT siteMolecules, GLOBAL const int* RESTRICT neighbors,
        LOCAL_ARG real4* delta, LOCAL_ARG real4* params, LOCAL_ARG int* molecule, LOCAL_ARG int* siteIndex, LOCAL_ARG real2* damping,
        real dampingScale, real invDampingRadius, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY,
        real4 periodicBoxVecZ) {
    const int site = (overflow ? siteA+1+index : neighbors[siteA*maxNeighbors+index]);
    const real4 siteB = sites[site];
    real3 d = make_real3(siteB.x-siteA4.x, siteB.y-siteA4.y, siteB.z-siteA4.z);
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(d)
#endif
    const int moleculeB = siteMolecules[site];
    const real r2 = dot(d, d);
    const real4 p = siteParams[site];
    *delta = make_real4(d.x, d.y, d.z, r2);
    *params = make_real4(p.x*p.y/(frequencyA+p.y), p.y, p.z, p.w);
    *siteIndex = site;
    const bool inside = (moleculeB != moleculeA && r2 < CUTOFF_SQUARED);
    *molecule = (inside ? moleculeB : NOT_A_NEIGHBOR);
#if defined(USE_DAMPING) || defined(USE_CENTROID_DAMPING)
    real dfdrOverR = 0, f = 1;
    if (inside)
        f = pairDamping(SQRT(r2), widthA+p.z, dampingScale, invDampingRadius, &dfdrOverR);
    *damping = make_real2(f, dfdrOverR);
#endif
}

/**
 * Compute the triples, each once, from its lowest site A: each thread block takes a site A and loops over the pairs
 * (B, C), B < C, of its half neighbor list (all the sites above A if the list overflowed). Tiles of TILE_SIZE neighbors
 * (their vectors from A, parameters, molecules, indices and the damping of the pair with A) are loaded into local
 * memory, and the threads share out the pairs of each pair of tiles. A triple interacts if C is also within the cutoff
 * of B and in a different molecule. With the minimum image vectors a = AB, b = AC and c = BC (the minimum image of
 * b - a), u, v, w their squared lengths, P = uvw and p = a.b, q = a.c, s = b.c, E = C9 [P^(-3/2) - 3 pqs P^(-5/2)],
 * times the damping of the three pairs when damping is enabled. The force on A is summed over the block; those on B
 * (-dE/da + dE/dc) and C (-dE/db - dE/dc) are added to the site forces atomically.
 */
KERNEL void computeTriples(int numSites, int maxNeighbors, GLOBAL const real4* RESTRICT sites, GLOBAL const real4* RESTRICT siteParams,
        GLOBAL const int* RESTRICT siteMolecules, GLOBAL const int* RESTRICT neighbors, GLOBAL const int* RESTRICT numActive,
        GLOBAL mm_ulong* RESTRICT siteForce, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT globals,
        GLOBAL int* RESTRICT rebuildFlag, GLOBAL const real4* RESTRICT tileBoxes, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    LOCAL real4 tileDeltaJ[TILE_SIZE], tileDeltaK[TILE_SIZE], tileParamsJ[TILE_SIZE], tileParamsK[TILE_SIZE];
    LOCAL int tileMoleculeJ[TILE_SIZE], tileMoleculeK[TILE_SIZE], tileSiteJ[TILE_SIZE], tileSiteK[TILE_SIZE];
    LOCAL real2 tileDampJ[TILE_SIZE], tileDampK[TILE_SIZE];
    // The forces on the B and C sites of the tiles are accumulated here (shared-memory atomics) and added to the site
    // forces once per tile pair, instead of six global atomic additions per triple.
    LOCAL real tileForceJ[3*TILE_SIZE], tileForceK[3*TILE_SIZE];
    LOCAL real3 reduction[TRIPLE_BLOCK_SIZE];
    LOCAL mixed energyReduction[TRIPLE_BLOCK_SIZE];
    // findNeighbors() has run: the next step starts without a rebuild request.
    if (GLOBAL_ID == 0)
        *rebuildFlag = 0;
    const real lambda = globals[0];
    const real scaleFactor = globals[1];
    const real dampingScale = globals[2];
    const real invDampingRadius = globals[3];
    mixed energy = 0;
    // Tile pairs whose bounding boxes (from the current positions, compactNeighbors()) are more than the cutoff apart are
    // skipped. Not for triclinic boxes or boxes narrower than 2 CUTOFF.
    const int maxTiles = (maxNeighbors+TILE_SIZE-1)/TILE_SIZE;
#ifdef USE_PERIODIC
    const bool pruneTiles = (periodicBoxVecY.x == 0 && periodicBoxVecZ.x == 0 && periodicBoxVecZ.y == 0 && periodicBoxSize.x >= 2*(real) CUTOFF &&
            periodicBoxSize.y >= 2*(real) CUTOFF && periodicBoxSize.z >= 2*(real) CUTOFF);
    const bool periodic = true;
#else
    const bool pruneTiles = true;
    const bool periodic = false;
#endif
    for (int siteA = GROUP_ID; siteA < numSites; siteA += NUM_GROUPS) {
        const bool overflow = (numActive[siteA] < 0);
        const int numNeighborsA = (overflow ? numSites-siteA-1 : numActive[siteA]);
        const real4 paramsA = siteParams[siteA];
        const real4 siteA4 = sites[siteA];
        const int moleculeA = siteMolecules[siteA];
        const bool alchemicalA = (paramsA.w != 0);
        real3 forceA = make_real3(0);
        for (int startJ = 0; startJ < numNeighborsA; startJ += TILE_SIZE) {
            const int sizeJ = min(TILE_SIZE, numNeighborsA-startJ);
            for (int startK = startJ; startK < numNeighborsA; startK += TILE_SIZE) {
                const int sizeK = min(TILE_SIZE, numNeighborsA-startK);
                if (pruneTiles && !overflow) {
                    const int boxJ = 2*(siteA*maxTiles+startJ/TILE_SIZE), boxK = 2*(siteA*maxTiles+startK/TILE_SIZE);
                    const real4 loJ = tileBoxes[boxJ], hiJ = tileBoxes[boxJ+1], loK = tileBoxes[boxK], hiK = tileBoxes[boxK+1];
                    const real gx = intervalGap(loJ.x, hiJ.x, loK.x, hiK.x, periodicBoxSize.x, periodic);
                    const real gy = intervalGap(loJ.y, hiJ.y, loK.y, hiK.y, periodicBoxSize.y, periodic);
                    const real gz = intervalGap(loJ.z, hiJ.z, loK.z, hiK.z, periodicBoxSize.z, periodic);
                    if (gx*gx+gy*gy+gz*gz >= PRUNE_SQUARED)
                        continue;  // the same for every thread of the block
                }

                // Load the two tiles of neighbors.

                SYNC_THREADS;
                for (int i = LOCAL_ID; i < TILE_SIZE; i += LOCAL_SIZE) {
                    if (i < sizeJ)
                        loadNeighbor(siteA, startJ+i, overflow, maxNeighbors, moleculeA, siteA4, paramsA.z, paramsA.y, sites, siteParams, siteMolecules, neighbors,
                                &tileDeltaJ[i], &tileParamsJ[i], &tileMoleculeJ[i], &tileSiteJ[i], &tileDampJ[i], dampingScale, invDampingRadius,
                                periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    if (i < sizeK)
                        loadNeighbor(siteA, startK+i, overflow, maxNeighbors, moleculeA, siteA4, paramsA.z, paramsA.y, sites, siteParams, siteMolecules, neighbors,
                                &tileDeltaK[i], &tileParamsK[i], &tileMoleculeK[i], &tileSiteK[i], &tileDampK[i], dampingScale, invDampingRadius,
                                periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
                for (int i = LOCAL_ID; i < 3*TILE_SIZE; i += LOCAL_SIZE) {
                    tileForceJ[i] = 0;
                    tileForceK[i] = 0;
                }
                SYNC_THREADS;

                // Compute the triples of the pairs (B, C) with B in the first tile and C after it in the second.

                const real wA = paramsA.y;
                const real cA = scaleFactor*(real) 1.5*paramsA.x*wA;
                // Each warp takes rows j (the B sites) and its 32 threads the columns k = lane + 32 kk (the C sites). The
                // values of B are read once per row; the force on B accumulates in registers over the row and is summed
                // over the warp once per row. A thread always has the same columns, so the force on each of its C sites
                // accumulates in registers over all the rows of the warp and is added to local memory once per tile pair.
                const int warp = LOCAL_ID/32, lane = LOCAL_ID%32, numWarps = LOCAL_SIZE/32;
                real3 columnForceC[TILE_SIZE/32];
                for (int kk = 0; kk < TILE_SIZE/32; kk++)
                    columnForceC[kk] = make_real3(0);
                for (int j = warp; j < sizeJ; j += numWarps) {
                    const int moleculeB = tileMoleculeJ[j];
                    if (moleculeB == NOT_A_NEIGHBOR)
                        continue;  // the same for the whole warp
                    const real4 deltaJ = tileDeltaJ[j];
                    const real4 paramsB = tileParamsJ[j];
                    const real3 a = trimTo3(deltaJ);
                    const real u = deltaJ.w, wB = paramsB.y, cAB = cA*paramsB.x;
                    const bool alchemicalAB = (alchemicalA || paramsB.w != 0);
    #if defined(USE_DAMPING) || defined(USE_CENTROID_DAMPING)
                    const real2 dampA = tileDampJ[j];
    #endif
                    real3 rowForceB = make_real3(0);
                    #pragma unroll
                    for (int kk = 0; kk < TILE_SIZE/32; kk++) {
                        const int k = lane+32*kk;
                        if (k >= sizeK || startK+k <= startJ+j)
                            continue;
                        const int moleculeC = tileMoleculeK[k];
                        if (moleculeC == NOT_A_NEIGHBOR || moleculeC == moleculeB)
                            continue;
                        const real4 deltaK = tileDeltaK[k];
                        const real3 b = trimTo3(deltaK);
                        real3 c = b-a;
    #ifdef USE_PERIODIC
                        APPLY_PERIODIC_TO_DELTA(c)
    #endif
                        const real w = dot(c, c);
                        if (w >= CUTOFF_SQUARED)
                            continue;
                        const real4 paramsC = tileParamsK[k];
                        const real wBC = wB+paramsC.y;
    #ifdef USE_DAMPING
                        // One reciprocal for 1/(w_B + w_C) and the Tang-Toennies rate 2 s / (width_B + width_C).
                        const real widthBC = paramsB.z+paramsC.z;
                        const real invBoth = 1/(wBC*widthBC);
                        real c9 = cAB*paramsC.x*(wA+wBC)*(widthBC*invBoth);
                        const real rateBC = 2*dampingScale*(wBC*invBoth);
    #else
                        real c9 = cAB*paramsC.x*(wA+wBC)/wBC;
                        const real rateBC = 0;
    #endif
                        if (alchemicalAB || paramsC.w != 0)
                            c9 *= lambda;
                        // A switched-off triple is exactly zero; skip it, as 1/(u v w)^(5/2) overflows in single
                        // precision when two of its sites nearly coincide (possible for a decoupled site).
                        if (c9 == 0)
                            continue;
                        const real v = deltaK.w;
                        const real p = dot(a, b), q = dot(a, c), s = dot(b, c);
                        const real rsqrtP = RSQRT(u*v*w);
                        const real invP = rsqrtP*rsqrtP;
                        const real invP32 = invP*rsqrtP;
                        const real invP52 = invP32*invP;
                        const real pqs = p*q*s;
                        real tripleEnergy = c9*(invP32 - 3*pqs*invP52);
                        const real radial = c9*(-3*invP32 + 15*pqs*invP52);
                        const real cross = -3*c9*invP52;
                        real3 dEda = a*(radial/u) + (b*(q*s) + c*(p*s))*cross;
                        real3 dEdb = b*(radial/v) + (a*(q*s) + c*(p*q))*cross;
                        real3 dEdc = c*(radial/w) + (a*(p*s) + b*(p*q))*cross;
    #if defined(USE_DAMPING) || defined(USE_CENTROID_DAMPING)
                        // E = E0 fa fb fc: the dampings of A-B and A-C were computed with the tiles, that of B-C here.
                        const real2 dampB = tileDampK[k];
                        const real rsqrtW = RSQRT(w);
                        real dC;
                        const real fC = pairDampingRate(w*rsqrtW, rsqrtW, rateBC, invDampingRadius, &dC);
                        const real fA = dampA.x, fB = dampB.x;
                        const real damping = fA*fB*fC;
                        dEda = dEda*damping + a*(tripleEnergy*dampA.y*fB*fC);
                        dEdb = dEdb*damping + b*(tripleEnergy*fA*dampB.y*fC);
                        dEdc = dEdc*damping + c*(tripleEnergy*fA*fB*dC);
                        tripleEnergy *= damping;
    #endif
                        energy += tripleEnergy;
                        forceA += dEda + dEdb;
                        rowForceB += dEdc-dEda;
                        columnForceC[kk] -= dEdb+dEdc;
                    }
                    for (int offset = 16; offset > 0; offset /= 2) {
                        rowForceB.x += __shfl_down_sync(0xffffffff, rowForceB.x, offset);
                        rowForceB.y += __shfl_down_sync(0xffffffff, rowForceB.y, offset);
                        rowForceB.z += __shfl_down_sync(0xffffffff, rowForceB.z, offset);
                    }
                    if (lane == 0) {
                        atomicAdd(&tileForceJ[3*j], rowForceB.x);
                        atomicAdd(&tileForceJ[3*j+1], rowForceB.y);
                        atomicAdd(&tileForceJ[3*j+2], rowForceB.z);
                    }
                }
                for (int kk = 0; kk < TILE_SIZE/32; kk++) {
                    const int k = lane+32*kk;
                    if (k < sizeK) {
                        atomicAdd(&tileForceK[3*k], columnForceC[kk].x);
                        atomicAdd(&tileForceK[3*k+1], columnForceC[kk].y);
                        atomicAdd(&tileForceK[3*k+2], columnForceC[kk].z);
                    }
                }

                // Add the forces accumulated on the sites of the two tiles.
                SYNC_THREADS;
                for (int i = LOCAL_ID; i < TILE_SIZE; i += LOCAL_SIZE) {
                    if (i < sizeJ && tileMoleculeJ[i] != NOT_A_NEIGHBOR) {
                        const int site = tileSiteJ[i];
                        ATOMIC_ADD(&siteForce[site], (mm_ulong) realToFixedPoint(tileForceJ[3*i]));
                        ATOMIC_ADD(&siteForce[site+numSites], (mm_ulong) realToFixedPoint(tileForceJ[3*i+1]));
                        ATOMIC_ADD(&siteForce[site+2*numSites], (mm_ulong) realToFixedPoint(tileForceJ[3*i+2]));
                    }
                    if (i < sizeK && tileMoleculeK[i] != NOT_A_NEIGHBOR) {
                        const int site = tileSiteK[i];
                        ATOMIC_ADD(&siteForce[site], (mm_ulong) realToFixedPoint(tileForceK[3*i]));
                        ATOMIC_ADD(&siteForce[site+numSites], (mm_ulong) realToFixedPoint(tileForceK[3*i+1]));
                        ATOMIC_ADD(&siteForce[site+2*numSites], (mm_ulong) realToFixedPoint(tileForceK[3*i+2]));
                    }
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
    // Sum the energy over the threads of the block (every triple was computed once).

    energyReduction[LOCAL_ID] = energy;
    SYNC_THREADS;
    for (int step = LOCAL_SIZE/2; step > 0; step /= 2) {
        if (LOCAL_ID < step)
            energyReduction[LOCAL_ID] += energyReduction[LOCAL_ID+step];
        SYNC_THREADS;
    }
    if (LOCAL_ID == 0)
        energyBuffer[GROUP_ID] += energyReduction[0];
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
