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

// This file is compiled with AVX enabled (see platforms/cpu/CMakeLists.txt).  It is only called when the processor
// supports AVX.

#include "CpuInteraxPairForceFvec.h"
#include "openmm/OpenMMException.h"
#ifdef __AVX__
#include "openmm/internal/vectorizeAvx.h"
#endif

using namespace InteraxPlugin;
using namespace OpenMM;

namespace InteraxPlugin {

#ifdef __AVX__
template <class Term>
CpuInteraxPairForce* createCpuInteraxPairForceAvx(const InteraxPairForce& force, ThreadPool& threads) {
    return new CpuInteraxPairForceFvec<fvec8, 8, Term>(force, threads);
}

#else
template <class Term>
CpuInteraxPairForce* createCpuInteraxPairForceAvx(const InteraxPairForce& force, ThreadPool& threads) {
    throw OpenMMException("Internal error: the InteraxFF plugin was compiled without AVX support");
}
#endif

template CpuInteraxPairForce* createCpuInteraxPairForceAvx<CpuInteraxDispersionTerm>(const InteraxPairForce& force, ThreadPool& threads);
template CpuInteraxPairForce* createCpuInteraxPairForceAvx<CpuInteraxExchangeTerm>(const InteraxPairForce& force, ThreadPool& threads);
template CpuInteraxPairForce* createCpuInteraxPairForceAvx<CpuInteraxShortRangeTerm>(const InteraxPairForce& force, ThreadPool& threads);

} // namespace InteraxPlugin
