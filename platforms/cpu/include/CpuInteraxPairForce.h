#ifndef OPENMM_CPU_INTERAX_PAIR_FORCE_H_
#define OPENMM_CPU_INTERAX_PAIR_FORCE_H_

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
#include "internal/InteraxPairTerms.h"
#include "openmm/Vec3.h"
#include "openmm/cpu/AlignedArray.h"
#include "openmm/internal/ThreadPool.h"
#include <vector>

namespace InteraxPlugin {

/**
 * The global parameters of the pair terms: the damping scale of the dispersion, and the Taylor scheme of the Slater
 * overlap used by the exchange repulsion and the short-range correction.
 */
struct CpuInteraxPairGlobals {
    float dampingScale;
    InteraxPairTerms::SlaterTaylorScheme taylorScheme;
};

/**
 * This is the abstract interface to the CPU implementation of the InteraxFF pair forces (InteraxDispersionForce,
 * InteraxExchangeForce and InteraxShortRangeForce).  The implementation, CpuInteraxPairForceFvec, is vectorized with
 * SSE or AVX.  Call createCpuInteraxPairForce() to create an instance of the version appropriate for the current CPU.
 */
class CpuInteraxPairForce {
public:
    virtual ~CpuInteraxPairForce() {
    }
    /**
     * Set the per-particle parameters.
     *
     * @param parameters   parameters[i][k] is the k'th parameter of particle i
     */
    virtual void setParameters(const std::vector<std::vector<double> >& parameters) = 0;
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
    virtual void calculatePairIxn(const OpenMM::AlignedArray<float>& posq, const OpenMM::Vec3* boxVectors, double lambda, const CpuInteraxPairGlobals& globals,
            std::vector<OpenMM::AlignedArray<float> >& threadForce, bool includeEnergy, double& totalEnergy) = 0;
};

/**
 * This describes the per-particle parameters and the pair energy of InteraxDispersionForce to CpuInteraxPairForceFvec.
 * A pair interacts only if the product of the values of GATE_PARAMETER of its two particles is positive.
 */
struct CpuInteraxDispersionTerm {
    static const int NUM_PARAMETERS = 5;  // m1, m2, m3, alpha, width
    static const int GATE_PARAMETER = 3;  // alpha
    /**
     * Compute the energy of a pair and its derivative with respect to the distance, one pair per SIMD lane.
     */
    template <class FVEC>
    static void compute(FVEC r, const FVEC* p1, const FVEC* p2, const CpuInteraxPairGlobals& globals, FVEC& energy, FVEC& dEdR);
};

/**
 * This describes the per-particle parameters and the pair energy of InteraxExchangeForce to CpuInteraxPairForceFvec.
 */
struct CpuInteraxExchangeTerm {
    static const int NUM_PARAMETERS = 2;  // amplitude, width
    static const int GATE_PARAMETER = 1;  // width
    /**
     * Compute the energy of a pair and its derivative with respect to the distance, one pair per SIMD lane.
     */
    template <class FVEC>
    static void compute(FVEC r, const FVEC* p1, const FVEC* p2, const CpuInteraxPairGlobals& globals, FVEC& energy, FVEC& dEdR);
};

/**
 * This describes the per-particle parameters and the pair energy of InteraxShortRangeForce to CpuInteraxPairForceFvec.
 */
struct CpuInteraxShortRangeTerm {
    static const int NUM_PARAMETERS = 3;  // k, l, width
    static const int GATE_PARAMETER = 2;  // width
    /**
     * Compute the energy of a pair and its derivative with respect to the distance, one pair per SIMD lane.
     */
    template <class FVEC>
    static void compute(FVEC r, const FVEC* p1, const FVEC* p2, const CpuInteraxPairGlobals& globals, FVEC& energy, FVEC& dEdR);
};

/**
 * This function is called to create an instance of an appropriate subclass for the current CPU: vectorized with AVX
 * if the processor supports it, and with SSE otherwise.  Term is CpuInteraxDispersionTerm, CpuInteraxExchangeTerm or
 * CpuInteraxShortRangeTerm.
 *
 * @param force      the force to create it for
 * @param threads    the thread pool to use
 */
template <class Term>
CpuInteraxPairForce* createCpuInteraxPairForce(const InteraxPairForce& force, OpenMM::ThreadPool& threads);

} // namespace InteraxPlugin

#endif /*OPENMM_CPU_INTERAX_PAIR_FORCE_H_*/
