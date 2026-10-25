#ifndef OPENMM_INTERAXMULTIPOLEFORCEIMPL_H_
#define OPENMM_INTERAXMULTIPOLEFORCEIMPL_H_

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

#include "InteraxKernels.h"
#include "InteraxMultipoleForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * This is the internal implementation of InteraxMultipoleForce.
 */

class OPENMM_EXPORT_INTERAX InteraxMultipoleForceImpl : public OpenMM::ForceImpl {
public:
    InteraxMultipoleForceImpl(const InteraxMultipoleForce& owner);
    void initialize(OpenMM::ContextImpl& context);
    const InteraxMultipoleForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters();
    std::vector<std::string> getKernelNames();
    void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    const InteraxMultipoleForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXMULTIPOLEFORCEIMPL_H_*/
