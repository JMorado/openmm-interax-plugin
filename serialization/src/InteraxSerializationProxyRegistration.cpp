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

#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "OpenMMInterax.h"
#include "InteraxDispersionForceProxy.h"
#include "InteraxExchangeForceProxy.h"
#include "InteraxShortRangeForceProxy.h"
#include "InteraxThreeBodyForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_INTERAX void registerInteraxSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerInteraxSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerInteraxSerializationProxies();
#endif

using namespace InteraxPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_INTERAX void registerInteraxSerializationProxies() {
    SerializationProxy::registerProxy(typeid(InteraxDispersionForce), new InteraxDispersionForceProxy());
    SerializationProxy::registerProxy(typeid(InteraxExchangeForce), new InteraxExchangeForceProxy());
    SerializationProxy::registerProxy(typeid(InteraxShortRangeForce), new InteraxShortRangeForceProxy());
    SerializationProxy::registerProxy(typeid(InteraxThreeBodyForce), new InteraxThreeBodyForceProxy());
}
