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

#include "CpuInteraxKernelFactory.h"
#include "CpuInteraxKernels.h"
#include "openmm/cpu/CpuPlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <exception>

using namespace InteraxPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CPU");
        CpuInteraxKernelFactory* factory = new CpuInteraxKernelFactory();
        platform.registerKernelFactory(CalcInteraxDispersionForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcInteraxExchangeForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcInteraxShortRangeForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcInteraxThreeBodyForceKernel::Name(), factory);
    }
    catch (std::exception& ex) {
        // The CPU platform is not available.
    }
}

extern "C" OPENMM_EXPORT void registerInteraxCpuKernelFactories() {
    try {
        Platform::getPlatformByName("CPU");
    }
    catch (...) {
        Platform::registerPlatform(new CpuPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CpuInteraxKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CpuPlatform::PlatformData& data = CpuPlatform::getPlatformData(context);
    if (name == CalcInteraxDispersionForceKernel::Name())
        return new CpuCalcInteraxDispersionForceKernel(name, platform, data);
    if (name == CalcInteraxExchangeForceKernel::Name())
        return new CpuCalcInteraxExchangeForceKernel(name, platform, data);
    if (name == CalcInteraxShortRangeForceKernel::Name())
        return new CpuCalcInteraxShortRangeForceKernel(name, platform, data);
    if (name == CalcInteraxThreeBodyForceKernel::Name())
        return new CpuCalcInteraxThreeBodyForceKernel(name, platform, data);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
