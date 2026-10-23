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

#include "InteraxDispersionForce.h"
#include "internal/InteraxPairForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxDispersionForce::InteraxDispersionForce() : dampingScale(1.0), useLongRangeCorrection(false) {
}

int InteraxDispersionForce::addParticle(double m1, double m2, double m3, double alpha, double width, bool isAlchemical) {
    particles.push_back(ParticleInfo(m1, m2, m3, alpha, width, isAlchemical));
    return particles.size()-1;
}

void InteraxDispersionForce::getParticleParameters(int index, double& m1, double& m2, double& m3, double& alpha, double& width, bool& isAlchemical) const {
    ASSERT_VALID_INDEX(index, particles);
    m1 = particles[index].m1;
    m2 = particles[index].m2;
    m3 = particles[index].m3;
    alpha = particles[index].alpha;
    width = particles[index].width;
    isAlchemical = particles[index].isAlchemical;
}

void InteraxDispersionForce::setParticleParameters(int index, double m1, double m2, double m3, double alpha, double width, bool isAlchemical) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].m1 = m1;
    particles[index].m2 = m2;
    particles[index].m3 = m3;
    particles[index].alpha = alpha;
    particles[index].width = width;
    particles[index].isAlchemical = isAlchemical;
}

bool InteraxDispersionForce::getParticleIsAlchemical(int index) const {
    ASSERT_VALID_INDEX(index, particles);
    return particles[index].isAlchemical;
}

double InteraxDispersionForce::getDampingScale() const {
    return dampingScale;
}

void InteraxDispersionForce::setDampingScale(double scale) {
    dampingScale = scale;
}

bool InteraxDispersionForce::getUseLongRangeCorrection() const {
    return useLongRangeCorrection;
}

void InteraxDispersionForce::setUseLongRangeCorrection(bool use) {
    useLongRangeCorrection = use;
}

ForceImpl* InteraxDispersionForce::createImpl() const {
    return new InteraxDispersionForceImpl(*this);
}

void InteraxDispersionForce::updateParametersInContext(Context& context) {
    dynamic_cast<InteraxDispersionForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
