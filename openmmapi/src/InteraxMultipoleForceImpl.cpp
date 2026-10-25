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

#include "internal/InteraxMultipoleForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/Messages.h"
#include <cmath>
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxMultipoleForceImpl::InteraxMultipoleForceImpl(const InteraxMultipoleForce& owner) : owner(owner) {
}

void InteraxMultipoleForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (owner.getNumParticles() != numParticles)
        throw OpenMMException("InteraxMultipoleForce must have exactly as many particles as the System it belongs to.");
    if (owner.getNonbondedMethod() == InteraxMultipoleForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("InteraxMultipoleForce: "+Messages::cutoffTooLarge);
    }
    const double tol = 1e-5;
    for (int i = 0; i < numParticles; i++) {
        int axisType, atomZ, atomX, atomY;
        double charge, coreCharge, alpha, polarizability;
        bool isAlchemical;
        vector<double> dipole, quadrupole;
        owner.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, atomZ, atomX, atomY, isAlchemical);
        stringstream msg;
        msg << "InteraxMultipoleForce: particle " << i << ": ";
        if (dipole.size() != 3 || quadrupole.size() != 9)
            throw OpenMMException(msg.str()+"the dipole needs 3 and the quadrupole 9 components");
        if (fabs(quadrupole[0]+quadrupole[4]+quadrupole[8]) > tol)
            throw OpenMMException(msg.str()+"the quadrupole must be traceless");
        if (fabs(quadrupole[1]-quadrupole[3]) > tol || fabs(quadrupole[2]-quadrupole[6]) > tol || fabs(quadrupole[5]-quadrupole[7]) > tol)
            throw OpenMMException(msg.str()+"the quadrupole must be symmetric");
        if (alpha <= 0)
            throw OpenMMException(msg.str()+"alpha must be positive");
        if (polarizability < 0)
            throw OpenMMException(msg.str()+"the polarizability must not be negative");
        if (axisType < InteraxMultipoleForce::ZThenX || axisType > InteraxMultipoleForce::NoAxisType)
            throw OpenMMException(msg.str()+"invalid axis type");
        if (axisType != InteraxMultipoleForce::NoAxisType && (atomZ < 0 || atomZ >= numParticles))
            throw OpenMMException(msg.str()+"invalid z axis particle");
        if (axisType != InteraxMultipoleForce::NoAxisType && axisType != InteraxMultipoleForce::ZOnly && (atomX < 0 || atomX >= numParticles))
            throw OpenMMException(msg.str()+"invalid x axis particle");
        if ((axisType == InteraxMultipoleForce::ZBisect || axisType == InteraxMultipoleForce::ThreeFold) && (atomY < 0 || atomY >= numParticles))
            throw OpenMMException(msg.str()+"invalid y axis particle");
    }
    kernel = context.getPlatform().createKernel(CalcInteraxMultipoleForceKernel::Name(), context);
    kernel.getAs<CalcInteraxMultipoleForceKernel>().initialize(context.getSystem(), owner);
}

double InteraxMultipoleForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcInteraxMultipoleForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

map<string, double> InteraxMultipoleForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int i = 0; i < owner.getNumParticles(); i++) {
        int axisType, atomZ, atomX, atomY;
        double charge, coreCharge, alpha, polarizability;
        bool isAlchemical;
        vector<double> dipole, quadrupole;
        owner.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, atomZ, atomX, atomY, isAlchemical);
        if (isAlchemical) {
            parameters[owner.getAlchemicalParameterName()] = 1.0;
            break;
        }
    }
    return parameters;
}

vector<string> InteraxMultipoleForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcInteraxMultipoleForceKernel::Name());
    return names;
}

void InteraxMultipoleForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcInteraxMultipoleForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void InteraxMultipoleForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcInteraxMultipoleForceKernel>().getInducedDipoles(context, dipoles);
}

void InteraxMultipoleForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcInteraxMultipoleForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void InteraxMultipoleForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcInteraxMultipoleForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
