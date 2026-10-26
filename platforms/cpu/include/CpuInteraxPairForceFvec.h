#ifndef OPENMM_CPU_INTERAX_PAIR_FORCE_FVEC_H_
#define OPENMM_CPU_INTERAX_PAIR_FORCE_FVEC_H_

/* -------------------------------------------------------------------------- *
 *                            OpenMM InteraxFF Plugin                         *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM InteraxFF plugin, built on the OpenMM molecular *
 * simulation toolkit (https://openmm.org) and its example plugin.            *
 *                                                                            *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Joao Morado                                                       *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CpuInteraxPairForce.h"
#include "internal/InteraxPairTerms.h"
#include "openmm/OpenMMException.h"
#include "openmm/cpu/CpuNeighborList.h"
#include <atomic>
#include <set>

namespace InteraxPlugin {

template <class FVEC>
void CpuInteraxDispersionTerm::compute(FVEC r, const FVEC* p1, const FVEC* p2, const CpuInteraxPairGlobals& globals, FVEC& energy, FVEC& dEdR) {
    InteraxPairTerms::dispersionPair(r, p1[0], p1[1], p1[2], p1[3], p1[4], p2[0], p2[1], p2[2], p2[3], p2[4], FVEC(globals.dampingScale), energy, dEdR);
}

template <class FVEC>
void CpuInteraxExchangeTerm::compute(FVEC r, const FVEC* p1, const FVEC* p2, const CpuInteraxPairGlobals& globals, FVEC& energy, FVEC& dEdR) {
    InteraxPairTerms::exchangePair(r, p1[0], p1[1], p2[0], p2[1], globals.taylorScheme, energy, dEdR);
}

template <class FVEC>
void CpuInteraxShortRangeTerm::compute(FVEC r, const FVEC* p1, const FVEC* p2, const CpuInteraxPairGlobals& globals, FVEC& energy, FVEC& dEdR) {
    InteraxPairTerms::shortRangePair(r, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], globals.taylorScheme, energy, dEdR);
}

/**
 * This class implements an InteraxFF pair force on the CPU platform, with the term described by Term, vectorized with
 * FVEC (fvec4 for SSE, fvec8 for AVX).  Particles are grouped into blocks of BLOCK_SIZE (the width of FVEC) by a
 * CpuNeighborList.  Each block is processed by one thread, which computes the interactions of its BLOCK_SIZE particles
 * with one neighbor at a time, one SIMD lane per pair.
 */
template <class FVEC, int BLOCK_SIZE, class Term>
class CpuInteraxPairForceFvec : public CpuInteraxPairForce {
public:
    /**
     * Create a new CpuInteraxPairForceFvec.
     *
     * @param force      the force to create it for
     * @param threads    the thread pool to use
     */
    CpuInteraxPairForceFvec(const InteraxPairForce& force, OpenMM::ThreadPool& threads);
    /**
     * Set the per-particle parameters.
     *
     * @param parameters   parameters[i][k] is the k'th parameter of particle i
     */
    void setParameters(const std::vector<std::vector<double> >& parameters);
    /**
     * Calculate the pair interactions.
     *
     * @param posq             atom coordinates and charges in float format, as stored by the CPU platform
     * @param boxVectors       the vectors defining the periodic box
     * @param lambda           the value of the alchemical parameter
     * @param globals          the global parameters of the pair term
     * @param threadForce      the collection of arrays for each thread to add forces to
     * @param includeEnergy    whether to compute energy
     * @param totalEnergy      the energy is added to this
     */
    void calculatePairIxn(const OpenMM::AlignedArray<float>& posq, const OpenMM::Vec3* boxVectors, double lambda, const CpuInteraxPairGlobals& globals,
            std::vector<OpenMM::AlignedArray<float> >& threadForce, bool includeEnergy, double& totalEnergy);
private:
    InteraxPairForce::NonbondedMethod nonbondedMethod;
    float cutoff, switchingDistance, softcoreRadius;
    bool useSwitch, periodic;
    float boxSize[3], invBoxSize[3];
    OpenMM::ThreadPool& threads;
    std::vector<float> parameters[Term::NUM_PARAMETERS];
    std::vector<float> isAlchemical;
    OpenMM::CpuNeighborList neighborList;
    std::vector<std::set<int> > exclusions;
    std::vector<double> threadEnergy;
    // The following variables are used to make information accessible to the individual threads.  The threads copy
    // them before the loop over blocks, so the cache line of atomicCounter is only touched by the increments.
    const float* posq;
    std::vector<OpenMM::AlignedArray<float> >* threadForce;
    float lambda;
    CpuInteraxPairGlobals globals;
    bool includeEnergy;
    std::atomic<int> atomicCounter;

    /**
     * This routine contains the code executed by each thread.
     */
    void threadComputeForce(OpenMM::ThreadPool& threads, int threadIndex);

    /**
     * Calculate all the interactions for one block of atoms.  The values recorded for the threads are passed as
     * arguments so that the compiler can keep them in registers in the inner loop.
     *
     * @param blockIndex      the index of the atom block
     * @param posq            atom coordinates and charges in float format
     * @param forces          force array (forces added)
     * @param totalEnergy     total energy
     * @param lambda          the value of the alchemical parameter
     * @param globals         the global parameters of the pair term
     * @param includeEnergy   whether to compute energy
     */
    void calculateBlockIxn(int blockIndex, const float* posq, float* forces, double& totalEnergy, float lambda, const CpuInteraxPairGlobals& globals,
            bool includeEnergy);
};

template <class FVEC, int BLOCK_SIZE, class Term>
CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::CpuInteraxPairForceFvec(const InteraxPairForce& force, OpenMM::ThreadPool& threads) :
        threads(threads), neighborList(BLOCK_SIZE) {
    int numParticles = force.getNumParticles();
    nonbondedMethod = force.getNonbondedMethod();
    cutoff = force.getCutoffDistance();
    periodic = (nonbondedMethod == InteraxPairForce::CutoffPeriodic);
    useSwitch = (nonbondedMethod != InteraxPairForce::NoCutoff && force.getUseSwitchingFunction());
    switchingDistance = force.getSwitchingDistance();
    softcoreRadius = force.getSoftcoreRadius();
    exclusions.resize(numParticles);
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
    }
    isAlchemical.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        isAlchemical[i] = (force.getParticleIsAlchemical(i) ? 1.0f : 0.0f);
    if (nonbondedMethod == InteraxPairForce::NoCutoff)
        neighborList.createDenseNeighborList(numParticles, exclusions);
}

template <class FVEC, int BLOCK_SIZE, class Term>
void CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::setParameters(const std::vector<std::vector<double> >& particleParameters) {
    int numParticles = particleParameters.size();
    for (int k = 0; k < Term::NUM_PARAMETERS; k++) {
        parameters[k].resize(numParticles);
        for (int i = 0; i < numParticles; i++)
            parameters[k][i] = (float) particleParameters[i][k];
    }
}

template <class FVEC, int BLOCK_SIZE, class Term>
void CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::calculatePairIxn(const OpenMM::AlignedArray<float>& posq, const OpenMM::Vec3* boxVectors,
        double lambda, const CpuInteraxPairGlobals& globals, std::vector<OpenMM::AlignedArray<float> >& threadForce, bool includeEnergy, double& totalEnergy) {
    if (periodic) {
        if (boxVectors[0][1] != 0 || boxVectors[0][2] != 0 || boxVectors[1][0] != 0 || boxVectors[1][2] != 0 || boxVectors[2][0] != 0 || boxVectors[2][1] != 0)
            throw OpenMM::OpenMMException("The CPU implementation of the InteraxFF pair forces does not support triclinic boxes");
        for (int i = 0; i < 3; i++) {
            boxSize[i] = (float) boxVectors[i][i];
            invBoxSize[i] = (float) (1/boxVectors[i][i]);
        }
    }
    int numParticles = isAlchemical.size();
    if (nonbondedMethod != InteraxPairForce::NoCutoff)
        neighborList.computeNeighborList(numParticles, posq, exclusions, boxVectors, periodic, cutoff, threads);

    // Record the parameters for the threads.

    this->posq = &posq[0];
    this->threadForce = &threadForce;
    this->lambda = (float) lambda;
    this->globals = globals;
    this->includeEnergy = includeEnergy;
    threadEnergy.resize(threads.getNumThreads());
    atomicCounter = 0;

    // Signal the threads to start running and wait for them to finish.

    threads.execute([&] (OpenMM::ThreadPool& threads, int threadIndex) { threadComputeForce(threads, threadIndex); });
    threads.waitForThreads();

    // Combine the energies from all the threads.

    if (includeEnergy) {
        int numThreads = threads.getNumThreads();
        for (int i = 0; i < numThreads; i++)
            totalEnergy += threadEnergy[i];
    }
}

template <class FVEC, int BLOCK_SIZE, class Term>
void CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::threadComputeForce(OpenMM::ThreadPool& threads, int threadIndex) {
    // Compute this thread's subset of interactions.

    threadEnergy[threadIndex] = 0;
    double& energy = threadEnergy[threadIndex];
    float* forces = &(*threadForce)[threadIndex][0];
    const float* posq = this->posq;
    const float lambda = this->lambda;
    const CpuInteraxPairGlobals globals = this->globals;
    const bool includeEnergy = this->includeEnergy;
    const int numBlocks = neighborList.getNumBlocks();
    while (true) {
        int blockIndex = atomicCounter++;
        if (blockIndex >= numBlocks)
            break;
        calculateBlockIxn(blockIndex, posq, forces, energy, lambda, globals, includeEnergy);
    }
}

template <class FVEC, int BLOCK_SIZE, class Term>
void CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::calculateBlockIxn(int blockIndex, const float* posq, float* forces, double& totalEnergy,
        float lambda, const CpuInteraxPairGlobals& globals, bool includeEnergy) {
    // Load the positions and parameters of the atoms in the block, one lane per atom.

    const int32_t* blockAtom = &neighborList.getSortedAtoms()[BLOCK_SIZE*blockIndex];
    float x[BLOCK_SIZE], y[BLOCK_SIZE], z[BLOCK_SIZE], values[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; i++) {
        x[i] = posq[4*blockAtom[i]];
        y[i] = posq[4*blockAtom[i]+1];
        z[i] = posq[4*blockAtom[i]+2];
    }
    FVEC blockX(x), blockY(y), blockZ(z);
    FVEC blockParameters[Term::NUM_PARAMETERS];
    for (int k = 0; k < Term::NUM_PARAMETERS; k++) {
        for (int i = 0; i < BLOCK_SIZE; i++)
            values[i] = parameters[k][blockAtom[i]];
        blockParameters[k] = FVEC(values);
    }
    for (int i = 0; i < BLOCK_SIZE; i++)
        values[i] = isAlchemical[blockAtom[i]];
    FVEC blockAlchemical(values);
    for (int i = 0; i < BLOCK_SIZE; i++)
        values[i] = (float) blockAtom[i];
    FVEC blockAtomIndex(values);
    const bool orderByIndex = !globals.taylorScheme.aboutSmallerWidth;
    FVEC blockForceX(0.0f), blockForceY(0.0f), blockForceZ(0.0f);
    FVEC blockEnergy(0.0f);
    const float invSwitchingInterval = 1/(cutoff-switchingDistance);
    const FVEC cutoffSquared(cutoff*cutoff);

    // Loop over neighbors for this block.

    OpenMM::CpuNeighborList::NeighborIterator neighbors = neighborList.getNeighborIterator(blockIndex);
    while (neighbors.next()) {
        int atom = neighbors.getNeighbor();

        // Compute the distances to the block atoms.

        FVEC dx = blockX-posq[4*atom];
        FVEC dy = blockY-posq[4*atom+1];
        FVEC dz = blockZ-posq[4*atom+2];
        if (periodic) {
            dx -= round(dx*invBoxSize[0])*boxSize[0];
            dy -= round(dy*invBoxSize[1])*boxSize[1];
            dz -= round(dz*invBoxSize[2])*boxSize[2];
        }
        FVEC r2 = dx*dx + dy*dy + dz*dz;

        // Select the lanes that interact: not excluded, within the cutoff, and with positive gate parameters.

        FVEC atomParameters[Term::NUM_PARAMETERS];
        for (int k = 0; k < Term::NUM_PARAMETERS; k++)
            atomParameters[k] = FVEC(parameters[k][atom]);
        auto include = FVEC::expandBitsToMask(~neighbors.getExclusions());
        if (nonbondedMethod != InteraxPairForce::NoCutoff)
            include = blendZero(r2 < cutoffSquared, include);
        include = blendZero(blockParameters[Term::GATE_PARAMETER]*atomParameters[Term::GATE_PARAMETER] > 0.0f, include);
        if (!any(include))
            continue; // No interactions to compute.

        // Compute the interactions.

        FVEC invR = rsqrt(r2);
        FVEC r = r2*invR;

        // A pair with exactly one alchemical particle is evaluated at the soft-core distance
        // r_eff = (r^6 + (1 - lambda) R^6)^(1/6), if enabled; dEdR is multiplied by d r_eff / d r = (r / r_eff)^5.
        auto alchemicalLane = abs(blockAlchemical-isAlchemical[atom]) > 0.5f;
        FVEC rEff = r, softcoreFactor = 1.0f;
        if (softcoreRadius > 0 && any(alchemicalLane)) {
            float rValues[BLOCK_SIZE], alchemicalValues[BLOCK_SIZE], rEffValues[BLOCK_SIZE], factorValues[BLOCK_SIZE];
            r.store(rValues);
            blend(FVEC(0.0f), FVEC(1.0f), alchemicalLane).store(alchemicalValues);
            float shift = (1-lambda)*pow(softcoreRadius, 6.0f);
            for (int i = 0; i < BLOCK_SIZE; i++) {
                rEffValues[i] = (alchemicalValues[i] != 0 ? pow(pow(rValues[i], 6.0f)+shift, 1.0f/6.0f) : rValues[i]);
                factorValues[i] = (alchemicalValues[i] != 0 ? pow(rValues[i]/rEffValues[i], 5.0f) : 1.0f);
            }
            rEff = FVEC(rEffValues);
            softcoreFactor = FVEC(factorValues);
        }
        FVEC pairEnergy, dEdR;
        if (orderByIndex) {
            // The InteraxFF Taylor schemes of the Slater overlap expand about the width of the particle with the
            // lower index, so put that particle first in each lane.
            FVEC blockFirst = blockAtomIndex < (float) atom;
            FVEC first[Term::NUM_PARAMETERS], second[Term::NUM_PARAMETERS];
            for (int k = 0; k < Term::NUM_PARAMETERS; k++) {
                first[k] = blendIf(blockFirst, blockParameters[k], atomParameters[k]);
                second[k] = blendIf(blockFirst, atomParameters[k], blockParameters[k]);
            }
            Term::compute(rEff, first, second, globals, pairEnergy, dEdR);
        }
        else
            Term::compute(rEff, blockParameters, atomParameters, globals, pairEnergy, dEdR);
        dEdR *= softcoreFactor;
        if (useSwitch) {
            FVEC t = blendZero((r-switchingDistance)*invSwitchingInterval, r > switchingDistance);
            FVEC switchValue = 1.0f+t*t*t*(-10.0f+t*(15.0f-t*6.0f));
            FVEC switchDeriv = t*t*(-30.0f+t*(60.0f-t*30.0f))*invSwitchingInterval;
            dEdR = dEdR*switchValue + pairEnergy*switchDeriv;
            pairEnergy *= switchValue;
        }
        FVEC scale = blend(FVEC(1.0f), FVEC(lambda), alchemicalLane);
        if (includeEnergy)
            blockEnergy += blendZero(scale*pairEnergy, include);

        // Accumulate forces.

        FVEC forceScale = blendZero(scale*dEdR*invR, include);
        FVEC fx = dx*forceScale, fy = dy*forceScale, fz = dz*forceScale;
        blockForceX -= fx;
        blockForceY -= fy;
        blockForceZ -= fz;
        float* atomForce = forces+4*atom;
        (fvec4(atomForce) + reduceToVec3(fx, fy, fz)).store(atomForce);
    }
    if (includeEnergy)
        totalEnergy += reduceAdd(blockEnergy);

    // Record the forces on the block atoms.

    fvec4 f[BLOCK_SIZE];
    transpose(blockForceX, blockForceY, blockForceZ, 0.0f, f);
    for (int i = 0; i < BLOCK_SIZE; i++)
        (fvec4(forces+4*blockAtom[i]) + f[i]).store(forces+4*blockAtom[i]);
}

} // namespace InteraxPlugin

#endif /*OPENMM_CPU_INTERAX_PAIR_FORCE_FVEC_H_*/
