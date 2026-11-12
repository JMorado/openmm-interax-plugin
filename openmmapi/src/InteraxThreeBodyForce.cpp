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

InteraxThreeBodyForce::InteraxThreeBodyForce() : cutoffDistance(1.2), scaleFactor(1.0), dampingScale(1.0), centroidDampingRadius(0.0),
        dampingMethod(NoDamping), usePeriodic(false), alchemicalParameterName("lambda_sterics") {
}

int InteraxThreeBodyForce::addSite(const vector<int>& particles, double alpha, double frequency, double width, int molecule, bool isAlchemical) {
    sites.push_back(SiteInfo(particles, alpha, frequency, width, molecule, isAlchemical));
    return sites.size()-1;
}

void InteraxThreeBodyForce::getSiteParameters(int index, vector<int>& particles, double& alpha, double& frequency, double& width, int& molecule,
        bool& isAlchemical) const {
    ASSERT_VALID_INDEX(index, sites);
    particles = sites[index].particles;
    alpha = sites[index].alpha;
    frequency = sites[index].frequency;
    width = sites[index].width;
    molecule = sites[index].molecule;
    isAlchemical = sites[index].isAlchemical;
}

void InteraxThreeBodyForce::setSiteParameters(int index, const vector<int>& particles, double alpha, double frequency, double width, int molecule,
        bool isAlchemical) {
    ASSERT_VALID_INDEX(index, sites);
    sites[index].particles = particles;
    sites[index].alpha = alpha;
    sites[index].frequency = frequency;
    sites[index].width = width;
    sites[index].molecule = molecule;
    sites[index].isAlchemical = isAlchemical;
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

InteraxThreeBodyForce::DampingMethod InteraxThreeBodyForce::getDampingMethod() const {
    return dampingMethod;
}

void InteraxThreeBodyForce::setDampingMethod(DampingMethod method) {
    if (method != NoDamping && method != TangToennies)
        throw OpenMMException("InteraxThreeBodyForce: Illegal value for damping method");
    dampingMethod = method;
}

double InteraxThreeBodyForce::getDampingScale() const {
    return dampingScale;
}

void InteraxThreeBodyForce::setDampingScale(double scale) {
    dampingScale = scale;
}

double InteraxThreeBodyForce::getCentroidDampingRadius() const {
    return centroidDampingRadius;
}

void InteraxThreeBodyForce::setCentroidDampingRadius(double radius) {
    if (radius < 0)
        throw OpenMMException("InteraxThreeBodyForce: The centroid damping radius cannot be negative");
    centroidDampingRadius = radius;
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
