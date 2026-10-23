#ifndef INTERAX_KERNELS_H_
#define INTERAX_KERNELS_H_

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

#include "InteraxDispersionForce.h"
#include "InteraxExchangeForce.h"
#include "InteraxShortRangeForce.h"
#include "InteraxThreeBodyForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace InteraxPlugin {

/**
 * This kernel is invoked by InteraxDispersionForce to calculate the forces acting on the system and the energy of
 * the system.
 */
class CalcInteraxDispersionForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcInteraxDispersionForce";
    }
    CalcInteraxDispersionForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxDispersionForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const InteraxDispersionForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxDispersionForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxDispersionForce& force) = 0;
};

/**
 * This kernel is invoked by InteraxExchangeForce to calculate the forces acting on the system and the energy of
 * the system.
 */
class CalcInteraxExchangeForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcInteraxExchangeForce";
    }
    CalcInteraxExchangeForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxExchangeForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const InteraxExchangeForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxExchangeForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxExchangeForce& force) = 0;
};

/**
 * This kernel is invoked by InteraxShortRangeForce to calculate the forces acting on the system and the energy of
 * the system.
 */
class CalcInteraxShortRangeForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcInteraxShortRangeForce";
    }
    CalcInteraxShortRangeForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxShortRangeForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const InteraxShortRangeForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxShortRangeForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxShortRangeForce& force) = 0;
};

/**
 * This kernel is invoked by InteraxThreeBodyForce to calculate the forces acting on the system and the energy of
 * the system.
 */
class CalcInteraxThreeBodyForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcInteraxThreeBodyForce";
    }
    CalcInteraxThreeBodyForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxThreeBodyForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const InteraxThreeBodyForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxThreeBodyForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxThreeBodyForce& force) = 0;
};

} // namespace InteraxPlugin

#endif /*INTERAX_KERNELS_H_*/
