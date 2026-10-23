#ifndef REFERENCE_INTERAX_KERNELS_H_
#define REFERENCE_INTERAX_KERNELS_H_

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
#include "internal/InteraxPairTerms.h"
#include "ReferenceInteraxPairIxn.h"
#include "ReferenceInteraxThreeBodyIxn.h"
#include "openmm/Platform.h"
#include <memory>
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * This kernel is invoked by InteraxDispersionForce to calculate the forces acting on the system and the energy of the
 * system.
 */
class ReferenceCalcInteraxDispersionForceKernel : public CalcInteraxDispersionForceKernel {
public:
    ReferenceCalcInteraxDispersionForceKernel(std::string name, const OpenMM::Platform& platform) : CalcInteraxDispersionForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxDispersionForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const InteraxDispersionForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxDispersionForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxDispersionForce& force);
private:
    void setParameters(const InteraxDispersionForce& force);
    struct ParticleParameters {
        double m1, m2, m3, alpha, width;
    };
    std::unique_ptr<ReferenceInteraxPairIxn> ixn;
    std::vector<ParticleParameters> particles;
    double dampingScale, longRangeCoefficient, alchemicalLongRangeCoefficient;
    std::string alchemicalParameter;
    bool hasAlchemicalParticles;
};

/**
 * This kernel is invoked by InteraxExchangeForce to calculate the forces acting on the system and the energy of the
 * system.
 */
class ReferenceCalcInteraxExchangeForceKernel : public CalcInteraxExchangeForceKernel {
public:
    ReferenceCalcInteraxExchangeForceKernel(std::string name, const OpenMM::Platform& platform) : CalcInteraxExchangeForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxExchangeForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const InteraxExchangeForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxExchangeForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxExchangeForce& force);
private:
    void setParameters(const InteraxExchangeForce& force);
    struct ParticleParameters {
        double amplitude, width;
    };
    std::unique_ptr<ReferenceInteraxPairIxn> ixn;
    std::vector<ParticleParameters> particles;
    std::string alchemicalParameter;
    bool hasAlchemicalParticles;
    int overlapMethod;
    InteraxPairTerms::SlaterTaylorScheme taylorScheme;
};

/**
 * This kernel is invoked by InteraxShortRangeForce to calculate the forces acting on the system and the energy of the
 * system.
 */
class ReferenceCalcInteraxShortRangeForceKernel : public CalcInteraxShortRangeForceKernel {
public:
    ReferenceCalcInteraxShortRangeForceKernel(std::string name, const OpenMM::Platform& platform) : CalcInteraxShortRangeForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxShortRangeForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const InteraxShortRangeForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxShortRangeForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxShortRangeForce& force);
private:
    void setParameters(const InteraxShortRangeForce& force);
    struct ParticleParameters {
        double k, l, width;
    };
    std::unique_ptr<ReferenceInteraxPairIxn> ixn;
    std::vector<ParticleParameters> particles;
    std::string alchemicalParameter;
    bool hasAlchemicalParticles;
    int overlapMethod;
    InteraxPairTerms::SlaterTaylorScheme taylorScheme;
};

/**
 * This kernel is invoked by InteraxThreeBodyForce to calculate the forces acting on the system and the energy of the
 * system.
 */
class ReferenceCalcInteraxThreeBodyForceKernel : public CalcInteraxThreeBodyForceKernel {
public:
    ReferenceCalcInteraxThreeBodyForceKernel(std::string name, const OpenMM::Platform& platform) : CalcInteraxThreeBodyForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxThreeBodyForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const InteraxThreeBodyForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxThreeBodyForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxThreeBodyForce& force);
private:
    std::unique_ptr<ReferenceInteraxThreeBodyIxn> ixn;
    std::string alchemicalParameter;
    bool hasAlchemicalMolecules;
};

} // namespace InteraxPlugin

#endif /*REFERENCE_INTERAX_KERNELS_H_*/
