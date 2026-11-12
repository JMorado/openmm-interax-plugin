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
    return *data->positions;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}

/**
 * Get whether any particle of a pair force is alchemical.
 */
static bool anyParticleIsAlchemical(const InteraxPairForce& force) {
    for (int i = 0; i < force.getNumParticles(); i++)
        if (force.getParticleIsAlchemical(i))
            return true;
    return false;
}

/**
 * Get the value of the alchemical parameter, or 1 if the force has no alchemical particles (in which case the
 * parameter need not be defined).
 */
static double getAlchemicalParameter(ContextImpl& context, bool isAlchemical, const string& name) {
    return (isAlchemical ? context.getParameter(name) : 1.0);
}

/* -------------------------------------------------------------------------- *
 *                         InteraxDispersionForce                             *
 * -------------------------------------------------------------------------- */

void CpuCalcInteraxDispersionForceKernel::initialize(const System& system, const InteraxDispersionForce& force) {
    ixn.reset(createCpuInteraxPairForce<CpuInteraxDispersionTerm>(force, data.threads));
    globals.taylorScheme = InteraxPairTerms::SlaterTaylorScheme::accurate(false); // Not used by the dispersion.
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    setParameters(force);
}

double CpuCalcInteraxDispersionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    Vec3* boxVectors = extractBoxVectors(context);
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    double energy = 0;
    ixn->calculatePairIxn(data.posq, boxVectors, lambda, globals, data.threadForce, includeEnergy, energy);

    // Add in the long range correction.

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
    globals.dampingScale = force.getDampingScale();
    InteraxDispersionForceImpl::calcLongRangeCorrection(force, longRangeCoefficient, alchemicalLongRangeCoefficient);
}

/* -------------------------------------------------------------------------- *
 *                          InteraxExchangeForce                              *
 * -------------------------------------------------------------------------- */

void CpuCalcInteraxExchangeForceKernel::initialize(const System& system, const InteraxExchangeForce& force) {
    ixn.reset(createCpuInteraxPairForce<CpuInteraxExchangeTerm>(force, data.threads));
    globals.dampingScale = 0;
    overlapMethod = force.getOverlapMethod();
    globals.taylorScheme = getSlaterTaylorScheme(force, false);
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    setParameters(force);
}

double CpuCalcInteraxExchangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    double energy = 0;
    ixn->calculatePairIxn(data.posq, extractBoxVectors(context), lambda, globals, data.threadForce, includeEnergy, energy);
    return energy;
}

void CpuCalcInteraxExchangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxExchangeForce& force) {
    if (force.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getOverlapMethod() != overlapMethod)
        throw OpenMMException("updateParametersInContext: The overlap method has changed");
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
    ixn.reset(createCpuInteraxPairForce<CpuInteraxShortRangeTerm>(force, data.threads));
    globals.dampingScale = 0;
    overlapMethod = force.getOverlapMethod();
    globals.taylorScheme = getSlaterTaylorScheme(force, false);
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    setParameters(force);
}

double CpuCalcInteraxShortRangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    double energy = 0;
    ixn->calculatePairIxn(data.posq, extractBoxVectors(context), lambda, globals, data.threadForce, includeEnergy, energy);
    return energy;
}

void CpuCalcInteraxShortRangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxShortRangeForce& force) {
    if (force.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getOverlapMethod() != overlapMethod)
        throw OpenMMException("updateParametersInContext: The overlap method has changed");
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
    ixn.reset(new CpuInteraxThreeBodyForce(force, data.threads));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalSites = false;
    for (int i = 0; i < force.getNumSites(); i++) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force.getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        hasAlchemicalSites |= isAlchemical;
    }
}

double CpuCalcInteraxThreeBodyForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalSites, alchemicalParameter);
    double energy = 0;
    ixn->calculateIxn(extractPositions(context), extractBoxVectors(context), lambda, extractForces(context), energy);
    return energy;
}

void CpuCalcInteraxThreeBodyForceKernel::copyParametersToContext(ContextImpl& context, const InteraxThreeBodyForce& force) {
    ixn->setParameters(force);
}
