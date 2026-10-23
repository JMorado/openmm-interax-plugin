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

#include "InteraxThreeBodyForce.h"
#include "internal/InteraxThreeBodyForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxThreeBodyForce::InteraxThreeBodyForce() : cutoffDistance(1.2), scaleFactor(1.0), usePeriodic(false),
        alchemicalParameterName("lambda_sterics") {
}

int InteraxThreeBodyForce::addMolecule(const vector<int>& particles, double alpha, double frequency, bool isAlchemical) {
    molecules.push_back(MoleculeInfo(particles, alpha, frequency, isAlchemical));
    return molecules.size()-1;
}

void InteraxThreeBodyForce::getMoleculeParameters(int index, vector<int>& particles, double& alpha, double& frequency, bool& isAlchemical) const {
    ASSERT_VALID_INDEX(index, molecules);
    particles = molecules[index].particles;
    alpha = molecules[index].alpha;
    frequency = molecules[index].frequency;
    isAlchemical = molecules[index].isAlchemical;
}

void InteraxThreeBodyForce::setMoleculeParameters(int index, const vector<int>& particles, double alpha, double frequency, bool isAlchemical) {
    ASSERT_VALID_INDEX(index, molecules);
    molecules[index].particles = particles;
    molecules[index].alpha = alpha;
    molecules[index].frequency = frequency;
    molecules[index].isAlchemical = isAlchemical;
}

double InteraxThreeBodyForce::getCutoffDistance() const {
    return cutoffDistance;
}

void InteraxThreeBodyForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

double InteraxThreeBodyForce::getScaleFactor() const {
    return scaleFactor;
}

void InteraxThreeBodyForce::setScaleFactor(double scale) {
    scaleFactor = scale;
}

void InteraxThreeBodyForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

const string& InteraxThreeBodyForce::getAlchemicalParameterName() const {
    return alchemicalParameterName;
}

void InteraxThreeBodyForce::setAlchemicalParameterName(const string& name) {
    alchemicalParameterName = name;
}

ForceImpl* InteraxThreeBodyForce::createImpl() const {
    return new InteraxThreeBodyForceImpl(*this);
}

void InteraxThreeBodyForce::updateParametersInContext(Context& context) {
    dynamic_cast<InteraxThreeBodyForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
