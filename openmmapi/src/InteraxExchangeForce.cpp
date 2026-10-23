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

#include "InteraxExchangeForce.h"
#include "internal/InteraxPairForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxExchangeForce::InteraxExchangeForce() : overlapMethod(Exact) {
}

InteraxExchangeForce::OverlapMethod InteraxExchangeForce::getOverlapMethod() const {
    return overlapMethod;
}

void InteraxExchangeForce::setOverlapMethod(OverlapMethod method) {
    if (method != Exact && method != InteraxOrder2 && method != InteraxOrder6)
        throw OpenMMException("InteraxExchangeForce: Illegal value for overlap method");
    overlapMethod = method;
}

int InteraxExchangeForce::addParticle(double amplitude, double width, bool isAlchemical) {
    particles.push_back(ParticleInfo(amplitude, width, isAlchemical));
    return particles.size()-1;
}

void InteraxExchangeForce::getParticleParameters(int index, double& amplitude, double& width, bool& isAlchemical) const {
    ASSERT_VALID_INDEX(index, particles);
    amplitude = particles[index].amplitude;
    width = particles[index].width;
    isAlchemical = particles[index].isAlchemical;
}

void InteraxExchangeForce::setParticleParameters(int index, double amplitude, double width, bool isAlchemical) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].amplitude = amplitude;
    particles[index].width = width;
    particles[index].isAlchemical = isAlchemical;
}

bool InteraxExchangeForce::getParticleIsAlchemical(int index) const {
    ASSERT_VALID_INDEX(index, particles);
    return particles[index].isAlchemical;
}

ForceImpl* InteraxExchangeForce::createImpl() const {
    return new InteraxExchangeForceImpl(*this);
}

void InteraxExchangeForce::updateParametersInContext(Context& context) {
    dynamic_cast<InteraxExchangeForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
