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

#include "internal/InteraxThreeBodyForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxThreeBodyForceImpl::InteraxThreeBodyForceImpl(const InteraxThreeBodyForce& owner) : owner(owner) {
}

void InteraxThreeBodyForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    for (int i = 0; i < owner.getNumMolecules(); i++) {
        vector<int> particles;
        double alpha, frequency;
        bool isAlchemical;
        owner.getMoleculeParameters(i, particles, alpha, frequency, isAlchemical);
        if (particles.empty())
            throw OpenMMException("InteraxThreeBodyForce: A molecule must contain at least one particle");
        for (int particle : particles)
            if (particle < 0 || particle >= system.getNumParticles())
                throw OpenMMException("InteraxThreeBodyForce: Illegal particle index for a molecule");
    }
    if (owner.getCutoffDistance() <= 0)
        throw OpenMMException("InteraxThreeBodyForce: The cutoff distance must be positive");
    kernel = context.getPlatform().createKernel(CalcInteraxThreeBodyForceKernel::Name(), context);
    kernel.getAs<CalcInteraxThreeBodyForceKernel>().initialize(context.getSystem(), owner);
}

double InteraxThreeBodyForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcInteraxThreeBodyForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

map<string, double> InteraxThreeBodyForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int i = 0; i < owner.getNumMolecules(); i++) {
        vector<int> particles;
        double alpha, frequency;
        bool isAlchemical;
        owner.getMoleculeParameters(i, particles, alpha, frequency, isAlchemical);
        if (isAlchemical) {
            parameters[owner.getAlchemicalParameterName()] = 1.0;
            break;
        }
    }
    return parameters;
}

vector<string> InteraxThreeBodyForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcInteraxThreeBodyForceKernel::Name());
    return names;
}

void InteraxThreeBodyForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcInteraxThreeBodyForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}
