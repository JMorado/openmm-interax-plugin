#ifndef OPENMM_INTERAXPAIRFORCEIMPL_H_
#define OPENMM_INTERAXPAIRFORCEIMPL_H_

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

#include "InteraxKernels.h"
#include "InteraxPairForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * Check that a pair force is consistent with the System it belongs to (number of particles, exclusions, cutoff,
 * switching distance and periodic box), and throw an OpenMMException if it is not.
 */
OPENMM_EXPORT_INTERAX void checkPairForce(const InteraxPairForce& force, const OpenMM::System& system, const std::string& forceName);

/**
 * The context parameters a pair force defines: its alchemical parameter (default 1) if any particle is alchemical.
 */
OPENMM_EXPORT_INTERAX std::map<std::string, double> getPairForceDefaultParameters(const InteraxPairForce& force);

/**
 * This is the internal implementation shared by the InteraxFF pair forces. ForceType is the Force class and
 * KernelType the kernel that computes it.
 */

template <class ForceType, class KernelType>
class InteraxPairForceImpl : public OpenMM::ForceImpl {
public:
    InteraxPairForceImpl(const ForceType& owner, const std::string& forceName) : owner(owner), forceName(forceName) {
    }
    void initialize(OpenMM::ContextImpl& context) {
        checkPairForce(owner, context.getSystem(), forceName);
        kernel = context.getPlatform().createKernel(KernelType::Name(), context);
        kernel.getAs<KernelType>().initialize(context.getSystem(), owner);
    }
    const ForceType& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<owner.getForceGroup())) != 0)
            return kernel.getAs<KernelType>().execute(context, includeForces, includeEnergy);
        return 0.0;
    }
    std::map<std::string, double> getDefaultParameters() {
        return getPairForceDefaultParameters(owner);
    }
    std::vector<std::string> getKernelNames() {
        std::vector<std::string> names;
        names.push_back(KernelType::Name());
        return names;
    }
    void updateParametersInContext(OpenMM::ContextImpl& context) {
        kernel.getAs<KernelType>().copyParametersToContext(context, owner);
        context.systemChanged();
    }
private:
    const ForceType& owner;
    std::string forceName;
    OpenMM::Kernel kernel;
};

/**
 * This is the internal implementation of InteraxDispersionForce. It also computes the long range correction, which
 * every platform uses.
 */
class OPENMM_EXPORT_INTERAX InteraxDispersionForceImpl : public InteraxPairForceImpl<InteraxDispersionForce, CalcInteraxDispersionForceKernel> {
public:
    InteraxDispersionForceImpl(const InteraxDispersionForce& owner) :
            InteraxPairForceImpl<InteraxDispersionForce, CalcInteraxDispersionForceKernel>(owner, "InteraxDispersionForce") {
    }
    /**
     * Compute the coefficients of the long range correction: its energy is (coefficient + lambda*alchemicalCoefficient)/V,
     * with V the volume of the periodic box and lambda the alchemical parameter. alchemicalCoefficient collects the
     * pairs with exactly one alchemical particle. Both are zero unless the force uses periodic boundary conditions
     * and the long range correction is enabled.
     */
    static void calcLongRangeCorrection(const InteraxDispersionForce& force, double& coefficient, double& alchemicalCoefficient);
};

/**
 * This is the internal implementation of InteraxExchangeForce.
 */
class OPENMM_EXPORT_INTERAX InteraxExchangeForceImpl : public InteraxPairForceImpl<InteraxExchangeForce, CalcInteraxExchangeForceKernel> {
public:
    InteraxExchangeForceImpl(const InteraxExchangeForce& owner) :
            InteraxPairForceImpl<InteraxExchangeForce, CalcInteraxExchangeForceKernel>(owner, "InteraxExchangeForce") {
    }
};

/**
 * This is the internal implementation of InteraxShortRangeForce.
 */
class OPENMM_EXPORT_INTERAX InteraxShortRangeForceImpl : public InteraxPairForceImpl<InteraxShortRangeForce, CalcInteraxShortRangeForceKernel> {
public:
    InteraxShortRangeForceImpl(const InteraxShortRangeForce& owner) :
            InteraxPairForceImpl<InteraxShortRangeForce, CalcInteraxShortRangeForceKernel>(owner, "InteraxShortRangeForce") {
    }
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXPAIRFORCEIMPL_H_*/
