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
#include "internal/InteraxPairForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/internal/AssertionUtilities.h"
#include <algorithm>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxPairForce::InteraxPairForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), switchingDistance(-1.0),
        softcoreRadius(0.0), useSwitchingFunction(false), alchemicalParameterName("lambda_sterics") {
}

InteraxPairForce::NonbondedMethod InteraxPairForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void InteraxPairForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 2)
        throw OpenMMException("InteraxPairForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double InteraxPairForce::getCutoffDistance() const {
    return cutoffDistance;
}

void InteraxPairForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

bool InteraxPairForce::getUseSwitchingFunction() const {
    return useSwitchingFunction;
}

void InteraxPairForce::setUseSwitchingFunction(bool use) {
    useSwitchingFunction = use;
}

double InteraxPairForce::getSwitchingDistance() const {
    return switchingDistance;
}

void InteraxPairForce::setSwitchingDistance(double distance) {
    switchingDistance = distance;
}

int InteraxPairForce::addExclusion(int particle1, int particle2) {
    exclusions.push_back(make_pair(particle1, particle2));
    return exclusions.size()-1;
}

void InteraxPairForce::getExclusionParticles(int index, int& particle1, int& particle2) const {
    ASSERT_VALID_INDEX(index, exclusions);
    particle1 = exclusions[index].first;
    particle2 = exclusions[index].second;
}

void InteraxPairForce::setExclusionParticles(int index, int particle1, int particle2) {
    ASSERT_VALID_INDEX(index, exclusions);
    exclusions[index] = make_pair(particle1, particle2);
}

const string& InteraxPairForce::getAlchemicalParameterName() const {
    return alchemicalParameterName;
}

void InteraxPairForce::setAlchemicalParameterName(const string& name) {
    alchemicalParameterName = name;
}

double InteraxPairForce::getSoftcoreRadius() const {
    return softcoreRadius;
}

void InteraxPairForce::setSoftcoreRadius(double radius) {
    if (radius < 0)
        throw OpenMMException("InteraxPairForce: The soft-core radius cannot be negative");
    softcoreRadius = radius;
}

void InteraxPlugin::checkPairForce(const InteraxPairForce& force, const System& system, const string& forceName) {
    int numParticles = force.getNumParticles();
    if (numParticles != system.getNumParticles())
        throw OpenMMException(forceName+" must have exactly as many particles as the System it belongs to.");
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        if (particle1 < 0 || particle1 >= numParticles || particle2 < 0 || particle2 >= numParticles)
            throw OpenMMException(forceName+": Illegal particle index for an exclusion");
    }
    if (force.getNonbondedMethod() == InteraxPairForce::NoCutoff)
        return;
    if (force.getCutoffDistance() <= 0)
        throw OpenMMException(forceName+": The cutoff distance must be positive");
    if (force.getUseSwitchingFunction() && (force.getSwitchingDistance() < 0 || force.getSwitchingDistance() >= force.getCutoffDistance()))
        throw OpenMMException(forceName+": Switching distance must satisfy 0 <= r_switch < r_cutoff");
    if (force.getNonbondedMethod() == InteraxPairForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = force.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException(forceName+": The cutoff distance cannot be greater than half the periodic box size.");
    }
}

map<string, double> InteraxPlugin::getPairForceDefaultParameters(const InteraxPairForce& force) {
    map<string, double> parameters;
    for (int i = 0; i < force.getNumParticles(); i++)
        if (force.getParticleIsAlchemical(i)) {
            parameters[force.getAlchemicalParameterName()] = 1.0;
            break;
        }
    return parameters;
}
