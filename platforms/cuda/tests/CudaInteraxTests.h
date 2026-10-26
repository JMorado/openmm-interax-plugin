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

/**
 * The platform of the InteraxFF tests on the CUDA platform: the global "platform" that the tests use and
 * setupKernels(), which registers the InteraxFF CUDA kernels and loads the InteraxFF Reference plugin (with which
 * some tests compare) and OpenMM's AMOEBA plugins (HippoNonbondedForce). The plugins are loaded rather than linked
 * because every plugin library defines registerKernelFactories(). The precision ("single", "mixed" or "double") and
 * the device index may be given as arguments, as for OpenMM's CUDA tests.
 */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/Platform.h"
#include "openmm/cuda/CudaPlatform.h"
#include <string>

extern "C" OPENMM_EXPORT void registerInteraxCudaKernelFactories();

OpenMM::CudaPlatform platform;

void initializeTests(int argc, char* argv[]) {
    if (argc > 1)
        platform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        platform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
}

void setupKernels(int argc, char* argv[]) {
    registerInteraxCudaKernelFactories();
    OpenMM::Platform::loadPluginLibrary(INTERAX_REFERENCE_PLUGIN);
    OpenMM::Platform::loadPluginLibrary(OPENMM_AMOEBA_REFERENCE_PLUGIN);
    OpenMM::Platform::loadPluginLibrary(OPENMM_AMOEBA_CUDA_PLUGIN);
    platform = dynamic_cast<OpenMM::CudaPlatform&>(OpenMM::Platform::getPlatformByName("CUDA"));
    initializeTests(argc, argv);
}
