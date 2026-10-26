/* -------------------------------------------------------------------------- *
 *                            OpenMM InteraxFF Plugin                         *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM InteraxFF plugin, built on the OpenMM molecular *
 * simulation toolkit (https://openmm.org). It is derived from OpenMM's       *
 * HippoNonbondedForce (AMOEBA plugin).                                       *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Peter Eastman (HippoNonbondedForce), Joao Morado                  *
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

#include "InteraxMultipoleForce.h"
#include "internal/InteraxMultipoleForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxMultipoleForce::InteraxMultipoleForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), ewaldErrorTol(1e-4), alpha(0.0),
        nx(0), ny(0), nz(0), coreDamping(false), useOwnKernel(false), coreDampingScale(1.0), mutualDampingScale(1.0),
        alchemicalParameterName("lambda_electrostatics") {
    extrapolationCoefficients = {0.042, 0.635, 0.414};
}

InteraxMultipoleForce::NonbondedMethod InteraxMultipoleForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void InteraxMultipoleForce::setNonbondedMethod(InteraxMultipoleForce::NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("InteraxMultipoleForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double InteraxMultipoleForce::getCutoffDistance() const {
    return cutoffDistance;
}

void InteraxMultipoleForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

const vector<double>& InteraxMultipoleForce::getExtrapolationCoefficients() const {
    return extrapolationCoefficients;
}

void InteraxMultipoleForce::setExtrapolationCoefficients(const vector<double>& coefficients) {
    if (coefficients.empty())
        throw OpenMMException("InteraxMultipoleForce: At least one extrapolation coefficient is required");
    extrapolationCoefficients = coefficients;
}

void InteraxMultipoleForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void InteraxMultipoleForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

void InteraxMultipoleForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const InteraxMultipoleForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

double InteraxMultipoleForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void InteraxMultipoleForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

bool InteraxMultipoleForce::getCoreDamping() const {
    return coreDamping;
}

void InteraxMultipoleForce::setCoreDamping(bool enabled) {
    coreDamping = enabled;
}

double InteraxMultipoleForce::getCoreDampingScale() const {
    return coreDampingScale;
}

void InteraxMultipoleForce::setCoreDampingScale(double eta) {
    if (eta <= 0)
        throw OpenMMException("InteraxMultipoleForce: The core damping scale must be positive");
    coreDampingScale = eta;
}

double InteraxMultipoleForce::getMutualDampingScale() const {
    return mutualDampingScale;
}

void InteraxMultipoleForce::setMutualDampingScale(double kappa) {
    if (kappa <= 0)
        throw OpenMMException("InteraxMultipoleForce: The mutual damping scale must be positive");
    mutualDampingScale = kappa;
}

bool InteraxMultipoleForce::getUseOwnKernel() const {
    return useOwnKernel;
}

void InteraxMultipoleForce::setUseOwnKernel(bool enabled) {
    useOwnKernel = enabled;
}

const string& InteraxMultipoleForce::getAlchemicalParameterName() const {
    return alchemicalParameterName;
}

void InteraxMultipoleForce::setAlchemicalParameterName(const string& name) {
    alchemicalParameterName = name;
}

int InteraxMultipoleForce::addParticle(double charge, const vector<double>& dipole, const vector<double>& quadrupole, double coreCharge,
                                       double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                                       bool isAlchemical) {
    particles.push_back(ParticleInfo(charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, multipoleAtomZ, multipoleAtomX,
                                     multipoleAtomY, isAlchemical));
    return particles.size()-1;
}

void InteraxMultipoleForce::getParticleParameters(int index, double& charge, vector<double>& dipole, vector<double>& quadrupole, double& coreCharge,
                                                  double& alpha, double& polarizability, int& axisType, int& multipoleAtomZ, int& multipoleAtomX,
                                                  int& multipoleAtomY, bool& isAlchemical) const {
    ASSERT_VALID_INDEX(index, particles);
    const ParticleInfo& p = particles[index];
    charge = p.charge;
    dipole = p.dipole;
    quadrupole = p.quadrupole;
    coreCharge = p.coreCharge;
    alpha = p.alpha;
    polarizability = p.polarizability;
    axisType = p.axisType;
    multipoleAtomZ = p.multipoleAtomZ;
    multipoleAtomX = p.multipoleAtomX;
    multipoleAtomY = p.multipoleAtomY;
    isAlchemical = p.isAlchemical;
}

void InteraxMultipoleForce::setParticleParameters(int index, double charge, const vector<double>& dipole, const vector<double>& quadrupole, double coreCharge,
                                                  double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX,
                                                  int multipoleAtomY, bool isAlchemical) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index] = ParticleInfo(charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, multipoleAtomZ, multipoleAtomX,
                                    multipoleAtomY, isAlchemical);
}

int InteraxMultipoleForce::addException(int particle1, int particle2, double multipoleMultipoleScale, double dipoleMultipoleScale,
                                        double dipoleDipoleScale, bool replace) {
    map<pair<int, int>, int>::iterator iter = exceptionMap.find(pair<int, int>(particle1, particle2));
    int newIndex;
    if (iter == exceptionMap.end())
        iter = exceptionMap.find(pair<int, int>(particle2, particle1));
    if (iter != exceptionMap.end()) {
        if (!replace) {
            stringstream msg;
            msg << "InteraxMultipoleForce: There is already an exception for particles " << particle1 << " and " << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[iter->second] = ExceptionInfo(particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale);
        newIndex = iter->second;
        exceptionMap.erase(iter->first);
    }
    else {
        exceptions.push_back(ExceptionInfo(particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale));
        newIndex = exceptions.size()-1;
    }
    exceptionMap[pair<int, int>(particle1, particle2)] = newIndex;
    return newIndex;
}

void InteraxMultipoleForce::getExceptionParameters(int index, int& particle1, int& particle2, double& multipoleMultipoleScale,
                                                   double& dipoleMultipoleScale, double& dipoleDipoleScale) const {
    ASSERT_VALID_INDEX(index, exceptions);
    particle1 = exceptions[index].particle1;
    particle2 = exceptions[index].particle2;
    multipoleMultipoleScale = exceptions[index].multipoleMultipoleScale;
    dipoleMultipoleScale = exceptions[index].dipoleMultipoleScale;
    dipoleDipoleScale = exceptions[index].dipoleDipoleScale;
}

void InteraxMultipoleForce::setExceptionParameters(int index, int particle1, int particle2, double multipoleMultipoleScale,
                                                   double dipoleMultipoleScale, double dipoleDipoleScale) {
    ASSERT_VALID_INDEX(index, exceptions);
    exceptions[index] = ExceptionInfo(particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale);
}

void InteraxMultipoleForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<InteraxMultipoleForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void InteraxMultipoleForce::getLabFramePermanentDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<InteraxMultipoleForceImpl&>(getImplInContext(context)).getLabFramePermanentDipoles(getContextImpl(context), dipoles);
}

ForceImpl* InteraxMultipoleForce::createImpl() const {
    return new InteraxMultipoleForceImpl(*this);
}

void InteraxMultipoleForce::updateParametersInContext(Context& context) {
    dynamic_cast<InteraxMultipoleForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
