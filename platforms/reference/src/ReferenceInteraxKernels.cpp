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

#include "ReferenceInteraxKernels.h"
#include "internal/InteraxPairForceImpl.h"
#include "internal/InteraxPairTerms.h"
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

void ReferenceCalcInteraxDispersionForceKernel::initialize(const System& system, const InteraxDispersionForce& force) {
    ixn.reset(new ReferenceInteraxPairIxn(force));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    particles.resize(force.getNumParticles());
    setParameters(force);
}

double ReferenceCalcInteraxDispersionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& positions = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3* boxVectors = extractBoxVectors(context);
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    double energy = ixn->calculatePairIxn(positions, boxVectors, lambda, forces, [&] (int i, int j, double r, double& pairEnergy, double& dEdR) {
        const ParticleParameters& p1 = particles[i];
        const ParticleParameters& p2 = particles[j];
        if (p1.alpha*p2.alpha == 0)
            return false;
        InteraxPairTerms::dispersionPair(r, p1.m1, p1.m2, p1.m3, p1.alpha, p1.width, p2.m1, p2.m2, p2.m3, p2.alpha, p2.width,
                dampingScale, pairEnergy, dEdR);
        return true;
    });
    if (longRangeCoefficient != 0 || alchemicalLongRangeCoefficient != 0)
        energy += (longRangeCoefficient+lambda*alchemicalLongRangeCoefficient)/(boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2]);
    return energy;
}

void ReferenceCalcInteraxDispersionForceKernel::copyParametersToContext(ContextImpl& context, const InteraxDispersionForce& force) {
    if (force.getNumParticles() != particles.size())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    setParameters(force);
}

void ReferenceCalcInteraxDispersionForceKernel::setParameters(const InteraxDispersionForce& force) {
    for (int i = 0; i < force.getNumParticles(); i++) {
        ParticleParameters& p = particles[i];
        bool isAlchemical;
        force.getParticleParameters(i, p.m1, p.m2, p.m3, p.alpha, p.width, isAlchemical);
    }
    dampingScale = force.getDampingScale();
    InteraxDispersionForceImpl::calcLongRangeCorrection(force, longRangeCoefficient, alchemicalLongRangeCoefficient);
}

/* -------------------------------------------------------------------------- *
 *                          InteraxExchangeForce                              *
 * -------------------------------------------------------------------------- */

void ReferenceCalcInteraxExchangeForceKernel::initialize(const System& system, const InteraxExchangeForce& force) {
    overlapMethod = force.getOverlapMethod();
    taylorScheme = getSlaterTaylorScheme(force, true);
    ixn.reset(new ReferenceInteraxPairIxn(force));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    particles.resize(force.getNumParticles());
    setParameters(force);
}

double ReferenceCalcInteraxExchangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    return ixn->calculatePairIxn(extractPositions(context), extractBoxVectors(context), lambda, extractForces(context),
            [&] (int i, int j, double r, double& pairEnergy, double& dEdR) {
        // The InteraxFF Taylor schemes expand about the width of the particle with the lower index.
        const ParticleParameters& p1 = particles[min(i, j)];
        const ParticleParameters& p2 = particles[max(i, j)];
        if (p1.width*p2.width == 0)
            return false;
        InteraxPairTerms::exchangePair(r, p1.amplitude, p1.width, p2.amplitude, p2.width, taylorScheme, pairEnergy, dEdR);
        return true;
    });
}

void ReferenceCalcInteraxExchangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxExchangeForce& force) {
    if (force.getNumParticles() != particles.size())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getOverlapMethod() != overlapMethod)
        throw OpenMMException("updateParametersInContext: The overlap method has changed");
    setParameters(force);
}

void ReferenceCalcInteraxExchangeForceKernel::setParameters(const InteraxExchangeForce& force) {
    for (int i = 0; i < force.getNumParticles(); i++) {
        bool isAlchemical;
        force.getParticleParameters(i, particles[i].amplitude, particles[i].width, isAlchemical);
    }
}

/* -------------------------------------------------------------------------- *
 *                         InteraxShortRangeForce                             *
 * -------------------------------------------------------------------------- */

void ReferenceCalcInteraxShortRangeForceKernel::initialize(const System& system, const InteraxShortRangeForce& force) {
    overlapMethod = force.getOverlapMethod();
    taylorScheme = getSlaterTaylorScheme(force, true);
    ixn.reset(new ReferenceInteraxPairIxn(force));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = anyParticleIsAlchemical(force);
    particles.resize(force.getNumParticles());
    setParameters(force);
}

double ReferenceCalcInteraxShortRangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter);
    return ixn->calculatePairIxn(extractPositions(context), extractBoxVectors(context), lambda, extractForces(context),
            [&] (int i, int j, double r, double& pairEnergy, double& dEdR) {
        // The InteraxFF Taylor schemes expand about the width of the particle with the lower index.
        const ParticleParameters& p1 = particles[min(i, j)];
        const ParticleParameters& p2 = particles[max(i, j)];
        if (p1.width*p2.width == 0)
            return false;
        InteraxPairTerms::shortRangePair(r, p1.k, p1.l, p1.width, p2.k, p2.l, p2.width, taylorScheme, pairEnergy, dEdR);
        return true;
    });
}

void ReferenceCalcInteraxShortRangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxShortRangeForce& force) {
    if (force.getNumParticles() != particles.size())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getOverlapMethod() != overlapMethod)
        throw OpenMMException("updateParametersInContext: The overlap method has changed");
    setParameters(force);
}

void ReferenceCalcInteraxShortRangeForceKernel::setParameters(const InteraxShortRangeForce& force) {
    for (int i = 0; i < force.getNumParticles(); i++) {
        bool isAlchemical;
        force.getParticleParameters(i, particles[i].k, particles[i].l, particles[i].width, isAlchemical);
    }
}

/* -------------------------------------------------------------------------- *
 *                          InteraxThreeBodyForce                             *
 * -------------------------------------------------------------------------- */

void ReferenceCalcInteraxThreeBodyForceKernel::initialize(const System& system, const InteraxThreeBodyForce& force) {
    ixn.reset(new ReferenceInteraxThreeBodyIxn(force));
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

double ReferenceCalcInteraxThreeBodyForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    double lambda = getAlchemicalParameter(context, hasAlchemicalSites, alchemicalParameter);
    return ixn->calculateIxn(extractPositions(context), extractBoxVectors(context), lambda, extractForces(context));
}

void ReferenceCalcInteraxThreeBodyForceKernel::copyParametersToContext(ContextImpl& context, const InteraxThreeBodyForce& force) {
    ixn->setParameters(force);
}

/* -------------------------------------------------------------------------- *
 *                          InteraxMultipoleForce                             *
 * -------------------------------------------------------------------------- */

static ReferenceInteraxMultipole* createMultipoleIxn(const InteraxMultipoleForce& force, const System& system) {
    if (force.getNonbondedMethod() == InteraxMultipoleForce::PME)
        return new ReferenceInteraxMultipolePme(force, system);
    return new ReferenceInteraxMultipole(force);
}

void ReferenceCalcInteraxMultipoleForceKernel::initialize(const System& system, const InteraxMultipoleForce& force) {
    numParticles = force.getNumParticles();
    ixn.reset(createMultipoleIxn(force, system));
    alchemicalParameter = force.getAlchemicalParameterName();
    hasAlchemicalParticles = false;
    for (int i = 0; i < numParticles; i++) {
        int axisType, atomZ, atomX, atomY;
        double charge, coreCharge, alpha, polarizability;
        bool isAlchemical;
        vector<double> dipole, quadrupole;
        force.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, atomZ, atomX, atomY, isAlchemical);
        hasAlchemicalParticles |= isAlchemical;
    }
}

void ReferenceCalcInteraxMultipoleForceKernel::setup(ContextImpl& context) {
    ixn->setLambda(getAlchemicalParameter(context, hasAlchemicalParticles, alchemicalParameter));
    if (ixn->getNonbondedMethod() == InteraxMultipoleForce::PME) {
        ReferenceInteraxMultipolePme* force = dynamic_cast<ReferenceInteraxMultipolePme*>(ixn.get());
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*force->getCutoffDistance();
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        force->setPeriodicBoxSize(boxVectors);
    }
}

double ReferenceCalcInteraxMultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    setup(context);
    return ixn->calculateForceAndEnergy(extractPositions(context), extractForces(context));
}

void ReferenceCalcInteraxMultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    setup(context);
    ixn->calculateInducedDipoles(extractPositions(context), dipoles);
}

void ReferenceCalcInteraxMultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    setup(context);
    ixn->calculateLabFramePermanentDipoles(extractPositions(context), dipoles);
}

void ReferenceCalcInteraxMultipoleForceKernel::copyParametersToContext(ContextImpl& context, const InteraxMultipoleForce& force) {
    if (numParticles != force.getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    ixn.reset(createMultipoleIxn(force, context.getSystem()));
}

void ReferenceCalcInteraxMultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (ixn->getNonbondedMethod() != InteraxMultipoleForce::PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    ReferenceInteraxMultipolePme* force = dynamic_cast<ReferenceInteraxMultipolePme*>(ixn.get());
    alpha = force->getAlphaEwald();
    vector<int> dim;
    force->getPmeGridDimensions(dim);
    nx = dim[0];
    ny = dim[1];
    nz = dim[2];
}
