#ifndef CPU_INTERAX_PAIR_FORCE_FVEC_H_
#define CPU_INTERAX_PAIR_FORCE_FVEC_H_

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
void CpuInteraxDispersionTerm::compute(FVEC r, const FVEC* p1, const FVEC* p2, float dampingScale, FVEC& energy, FVEC& dEdR) {
    InteraxPairTerms::dispersionPair(r, p1[0], p1[1], p1[2], p1[3], p1[4], p2[0], p2[1], p2[2], p2[3], p2[4], FVEC(dampingScale), energy, dEdR);
}

template <class FVEC>
void CpuInteraxExchangeTerm::compute(FVEC r, const FVEC* p1, const FVEC* p2, float unused, FVEC& energy, FVEC& dEdR) {
    InteraxPairTerms::exchangePair(r, p1[0], p1[1], p2[0], p2[1], energy, dEdR);
}

template <class FVEC>
void CpuInteraxShortRangeTerm::compute(FVEC r, const FVEC* p1, const FVEC* p2, float unused, FVEC& energy, FVEC& dEdR) {
    InteraxPairTerms::shortRangePair(r, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], energy, dEdR);
}

/**
 * This class implements an InteraxFF pair force on the CPU platform. Particles are grouped into blocks of
 * BLOCK_SIZE (the width of FVEC) by a CpuNeighborList; each block is processed by one thread, which computes the
 * interactions of its BLOCK_SIZE particles with each neighbor at once, one SIMD lane per pair.
 */
template <class FVEC, int BLOCK_SIZE, class Term>
class CpuInteraxPairForceFvec : public CpuInteraxPairForce {
public:
    explicit CpuInteraxPairForceFvec(const InteraxPairForce& force);
    void setParameters(const std::vector<std::vector<double> >& parameters);
    double calculateIxn(const OpenMM::AlignedArray<float>& posq, const OpenMM::Vec3* boxVectors, double lambda, double globalParameter,
            std::vector<OpenMM::AlignedArray<float> >& threadForce, OpenMM::ThreadPool& threads, bool includeEnergy);
private:
    void calculateBlockIxn(int blockIndex, const float* posq, float* forces, double& energy, float lambda, float globalParameter, bool includeEnergy);
    InteraxPairForce::NonbondedMethod nonbondedMethod;
    float cutoff, switchingDistance;
    bool useSwitch, periodic;
    float boxSize[3], invBoxSize[3];
    std::vector<std::set<int> > exclusions;
    std::vector<float> parameters[Term::NUM_PARAMETERS];
    std::vector<float> isAlchemical;
    OpenMM::CpuNeighborList neighborList;
};

template <class FVEC, int BLOCK_SIZE, class Term>
CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::CpuInteraxPairForceFvec(const InteraxPairForce& force) : neighborList(BLOCK_SIZE) {
    int numParticles = force.getNumParticles();
    nonbondedMethod = force.getNonbondedMethod();
    cutoff = force.getCutoffDistance();
    periodic = (nonbondedMethod == InteraxPairForce::CutoffPeriodic);
    useSwitch = (nonbondedMethod != InteraxPairForce::NoCutoff && force.getUseSwitchingFunction());
    switchingDistance = force.getSwitchingDistance();
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
double CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::calculateIxn(const OpenMM::AlignedArray<float>& posq, const OpenMM::Vec3* boxVectors,
        double lambda, double globalParameter, std::vector<OpenMM::AlignedArray<float> >& threadForce, OpenMM::ThreadPool& threads, bool includeEnergy) {
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
    std::vector<double> threadEnergy(threads.getNumThreads(), 0.0);
    std::atomic<int> atomicCounter(0);
    threads.execute([&] (OpenMM::ThreadPool& threads, int threadIndex) {
        float* forces = &threadForce[threadIndex][0];
        while (true) {
            int blockIndex = atomicCounter++;
            if (blockIndex >= neighborList.getNumBlocks())
                break;
            calculateBlockIxn(blockIndex, &posq[0], forces, threadEnergy[threadIndex], (float) lambda, (float) globalParameter, includeEnergy);
        }
    });
    threads.waitForThreads();
    double energy = 0;
    for (double e : threadEnergy)
        energy += e;
    return energy;
}

template <class FVEC, int BLOCK_SIZE, class Term>
void CpuInteraxPairForceFvec<FVEC, BLOCK_SIZE, Term>::calculateBlockIxn(int blockIndex, const float* posq, float* forces, double& energy,
        float lambda, float globalParameter, bool includeEnergy) {
    // Load the positions and parameters of the particles in the block, one lane per particle.

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
    FVEC blockForceX(0.0f), blockForceY(0.0f), blockForceZ(0.0f);
    FVEC blockEnergy(0.0f);
    const float invSwitchingInterval = 1/(cutoff-switchingDistance);
    const FVEC cutoffSquared(cutoff*cutoff);

    // Loop over the neighbors of the block.

    OpenMM::CpuNeighborList::NeighborIterator neighbors = neighborList.getNeighborIterator(blockIndex);
    while (neighbors.next()) {
        int atom = neighbors.getNeighbor();

        // The vectors from the neighbor to the block particles.

        FVEC dx = blockX-posq[4*atom];
        FVEC dy = blockY-posq[4*atom+1];
        FVEC dz = blockZ-posq[4*atom+2];
        if (periodic) {
            dx -= round(dx*invBoxSize[0])*boxSize[0];
            dy -= round(dy*invBoxSize[1])*boxSize[1];
            dz -= round(dz*invBoxSize[2])*boxSize[2];
        }
        FVEC r2 = dx*dx + dy*dy + dz*dz;

        // Which lanes interact: not excluded, within the cutoff, and with nonzero gate parameters.

        FVEC atomParameters[Term::NUM_PARAMETERS];
        for (int k = 0; k < Term::NUM_PARAMETERS; k++)
            atomParameters[k] = FVEC(parameters[k][atom]);
        auto include = FVEC::expandBitsToMask(~neighbors.getExclusions());
        if (nonbondedMethod != InteraxPairForce::NoCutoff)
            include = blendZero(r2 < cutoffSquared, include);
        include = blendZero(blockParameters[Term::GATE_PARAMETER]*atomParameters[Term::GATE_PARAMETER] > 0.0f, include);
        if (!any(include))
            continue;

        // Compute the interactions.

        FVEC invR = rsqrt(r2);
        FVEC r = r2*invR;
        FVEC pairEnergy, dEdR;
        Term::compute(r, blockParameters, atomParameters, globalParameter, pairEnergy, dEdR);
        if (useSwitch) {
            FVEC t = blendZero((r-switchingDistance)*invSwitchingInterval, r > switchingDistance);
            FVEC switchValue = 1.0f+t*t*t*(-10.0f+t*(15.0f-t*6.0f));
            FVEC switchDeriv = t*t*(-30.0f+t*(60.0f-t*30.0f))*invSwitchingInterval;
            dEdR = dEdR*switchValue + pairEnergy*switchDeriv;
            pairEnergy *= switchValue;
        }
        FVEC scale = blend(FVEC(1.0f), FVEC(lambda), abs(blockAlchemical-isAlchemical[atom]) > 0.5f);
        if (includeEnergy)
            blockEnergy += blendZero(scale*pairEnergy, include);

        // Accumulate the forces.

        FVEC forceScale = blendZero(scale*dEdR*invR, include);
        FVEC fx = dx*forceScale, fy = dy*forceScale, fz = dz*forceScale;
        blockForceX -= fx;
        blockForceY -= fy;
        blockForceZ -= fz;
        float* atomForce = forces+4*atom;
        (fvec4(atomForce) + reduceToVec3(fx, fy, fz)).store(atomForce);
    }
    if (includeEnergy)
        energy += reduceAdd(blockEnergy);

    // Record the forces on the block particles.

    fvec4 f[BLOCK_SIZE];
    transpose(blockForceX, blockForceY, blockForceZ, 0.0f, f);
    for (int i = 0; i < BLOCK_SIZE; i++)
        (fvec4(forces+4*blockAtom[i]) + f[i]).store(forces+4*blockAtom[i]);
}

} // namespace InteraxPlugin

#endif /*CPU_INTERAX_PAIR_FORCE_FVEC_H_*/
