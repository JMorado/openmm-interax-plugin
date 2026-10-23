#ifndef CPU_INTERAX_PAIR_FORCE_H_
#define CPU_INTERAX_PAIR_FORCE_H_

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

#include "InteraxPairForce.h"
#include "openmm/Vec3.h"
#include "openmm/cpu/AlignedArray.h"
#include "openmm/internal/ThreadPool.h"
#include <vector>

namespace InteraxPlugin {

/**
 * This is the interface of the CPU implementation of an InteraxFF pair force. The implementation
 * (CpuInteraxPairForceFvec) is vectorized with SSE or AVX, chosen at runtime by createCpuInteraxPairForce().
 */
class CpuInteraxPairForce {
public:
    virtual ~CpuInteraxPairForce() {
    }
    /**
     * Set the per-particle parameters: parameters[i][k] is the k'th parameter of particle i.
     */
    virtual void setParameters(const std::vector<std::vector<double> >& parameters) = 0;
    /**
     * Compute the interactions.
     *
     * @param posq             the single precision positions (x, y, z, charge) of the CPU platform
     * @param boxVectors       the periodic box vectors
     * @param lambda           the value of the alchemical parameter
     * @param globalParameter  the global parameter of the pair term (the damping scale of the dispersion)
     * @param threadForce      the per-thread force buffers of the CPU platform; the forces are added to them
     * @param threads          the thread pool to use
     * @param includeEnergy    true if the energy should be calculated
     * @return the energy
     */
    virtual double calculateIxn(const OpenMM::AlignedArray<float>& posq, const OpenMM::Vec3* boxVectors, double lambda, double globalParameter,
            std::vector<OpenMM::AlignedArray<float> >& threadForce, OpenMM::ThreadPool& threads, bool includeEnergy) = 0;
};

/**
 * The per-particle parameters and the pair energy of InteraxDispersionForce, in the form used by
 * CpuInteraxPairForceFvec. A pair interacts only if the product of its GATE_PARAMETER values is positive.
 */
struct CpuInteraxDispersionTerm {
    static const int NUM_PARAMETERS = 5;  // m1, m2, m3, alpha, width
    static const int GATE_PARAMETER = 3;  // alpha
    template <class FVEC>
    static void compute(FVEC r, const FVEC* p1, const FVEC* p2, float dampingScale, FVEC& energy, FVEC& dEdR);
};

/**
 * The per-particle parameters and the pair energy of InteraxExchangeForce.
 */
struct CpuInteraxExchangeTerm {
    static const int NUM_PARAMETERS = 2;  // amplitude, width
    static const int GATE_PARAMETER = 1;  // width
    template <class FVEC>
    static void compute(FVEC r, const FVEC* p1, const FVEC* p2, float unused, FVEC& energy, FVEC& dEdR);
};

/**
 * The per-particle parameters and the pair energy of InteraxShortRangeForce.
 */
struct CpuInteraxShortRangeTerm {
    static const int NUM_PARAMETERS = 3;  // k, l, width
    static const int GATE_PARAMETER = 2;  // width
    template <class FVEC>
    static void compute(FVEC r, const FVEC* p1, const FVEC* p2, float unused, FVEC& energy, FVEC& dEdR);
};

/**
 * Create the CPU implementation of a pair force, vectorized with AVX if the processor supports it and with SSE
 * otherwise. Term is one of CpuInteraxDispersionTerm, CpuInteraxExchangeTerm, CpuInteraxShortRangeTerm.
 */
template <class Term>
CpuInteraxPairForce* createCpuInteraxPairForce(const InteraxPairForce& force);

} // namespace InteraxPlugin

#endif /*CPU_INTERAX_PAIR_FORCE_H_*/
