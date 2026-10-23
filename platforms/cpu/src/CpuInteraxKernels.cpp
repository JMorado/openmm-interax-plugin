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

#include "CpuInteraxKernels.h"
#include "internal/InteraxPairForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

static bool anyParticleIsAlchemical(const InteraxPairForce& force) {
    for (int i = 0; i < force.getNumParticles(); i++)
        if (force.getParticleIsAlchemical(i))
            return true;
    return false;
}

static double getAlchemicalParameter(ContextImpl& context, bool isAlchemical, const string& name) {
    return (isAlchemical ? context.getParameter(name) : 1.0);
}

/* -------------------------------------------------------------------------- *
 *                         InteraxDispersionForce                             *
 * -------------------------------------------------------------------------- */

void CpuCalcInteraxDispersionForceKernel::initialize(const System& system, const InteraxDispersionForce& force) {
    ixn.reset(createCpuInteraxPairForce<CpuInteraxDispersionTerm>(force));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    setParameters(force);
}

double CpuCalcInteraxDispersionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    Vec3* boxVectors = extractBoxVectors(context);
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    double energy = ixn->calculateIxn(data.posq, boxVectors, lambda, dampingScale, data.threadForce, data.threads, includeEnergy);
    if (longRangeCoefficient != 0 || alchemicalLongRangeCoefficient != 0)
        energy += (longRangeCoefficient+lambda*alchemicalLongRangeCoefficient)/(boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2]);
    return energy;
}

void CpuCalcInteraxDispersionForceKernel::copyParametersToContext(ContextImpl& context, const InteraxDispersionForce& force) {
    if (force.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    setParameters(force);
}

void CpuCalcInteraxDispersionForceKernel::setParameters(const InteraxDispersionForce& force) {
    vector<vector<double> > parameters(force.getNumParticles(), vector<double>(CpuInteraxDispersionTerm::NUM_PARAMETERS));
    for (int i = 0; i < force.getNumParticles(); i++) {
        vector<double>& p = parameters[i];
        bool isAlchemical;
        force.getParticleParameters(i, p[0], p[1], p[2], p[3], p[4], isAlchemical);
    }
    ixn->setParameters(parameters);
    dampingScale = force.getDampingScale();
    InteraxDispersionForceImpl::calcLongRangeCorrection(force, longRangeCoefficient, alchemicalLongRangeCoefficient);
}

/* -------------------------------------------------------------------------- *
 *                          InteraxExchangeForce                              *
 * -------------------------------------------------------------------------- */

void CpuCalcInteraxExchangeForceKernel::initialize(const System& system, const InteraxExchangeForce& force) {
    ixn.reset(createCpuInteraxPairForce<CpuInteraxExchangeTerm>(force));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    setParameters(force);
}

double CpuCalcInteraxExchangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    return ixn->calculateIxn(data.posq, extractBoxVectors(context), lambda, 0.0, data.threadForce, data.threads, includeEnergy);
}

void CpuCalcInteraxExchangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxExchangeForce& force) {
    if (force.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    setParameters(force);
}

void CpuCalcInteraxExchangeForceKernel::setParameters(const InteraxExchangeForce& force) {
    vector<vector<double> > parameters(force.getNumParticles(), vector<double>(CpuInteraxExchangeTerm::NUM_PARAMETERS));
    for (int i = 0; i < force.getNumParticles(); i++) {
        bool isAlchemical;
        force.getParticleParameters(i, parameters[i][0], parameters[i][1], isAlchemical);
    }
    ixn->setParameters(parameters);
}

/* -------------------------------------------------------------------------- *
 *                         InteraxShortRangeForce                             *
 * -------------------------------------------------------------------------- */

void CpuCalcInteraxShortRangeForceKernel::initialize(const System& system, const InteraxShortRangeForce& force) {
    ixn.reset(createCpuInteraxPairForce<CpuInteraxShortRangeTerm>(force));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    setParameters(force);
}

double CpuCalcInteraxShortRangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    return ixn->calculateIxn(data.posq, extractBoxVectors(context), lambda, 0.0, data.threadForce, data.threads, includeEnergy);
}

void CpuCalcInteraxShortRangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxShortRangeForce& force) {
    if (force.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    setParameters(force);
}

void CpuCalcInteraxShortRangeForceKernel::setParameters(const InteraxShortRangeForce& force) {
    vector<vector<double> > parameters(force.getNumParticles(), vector<double>(CpuInteraxShortRangeTerm::NUM_PARAMETERS));
    for (int i = 0; i < force.getNumParticles(); i++) {
        bool isAlchemical;
        force.getParticleParameters(i, parameters[i][0], parameters[i][1], parameters[i][2], isAlchemical);
    }
    ixn->setParameters(parameters);
}

/* -------------------------------------------------------------------------- *
 *                          InteraxThreeBodyForce                             *
 * -------------------------------------------------------------------------- */

void CpuCalcInteraxThreeBodyForceKernel::initialize(const System& system, const InteraxThreeBodyForce& force) {
    ixn.reset(new CpuInteraxThreeBodyForce(force, data.threads.getNumThreads()));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalMolecules = false;
    for (int i = 0; i < force.getNumMolecules(); i++) {
        vector<int> particles;
        double alpha, frequency;
        bool isAlchemical;
        force.getMoleculeParameters(i, particles, alpha, frequency, isAlchemical);
        hasAlchemicalMolecules |= isAlchemical;
    }
}

double CpuCalcInteraxThreeBodyForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalMolecules, alchemicalParameter);
    return ixn->calculateIxn(extractPositions(context), extractBoxVectors(context), lambda, extractForces(context), data.threads);
}

void CpuCalcInteraxThreeBodyForceKernel::copyParametersToContext(ContextImpl& context, const InteraxThreeBodyForce& force) {
    ixn->setParameters(force);
}
