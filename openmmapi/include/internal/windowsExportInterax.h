#ifndef OPENMM_WINDOWSEXPORTINTERAX_H_
#define OPENMM_WINDOWSEXPORTINTERAX_H_

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

/*
 * Shared libraries are messy in Visual Studio. We have to distinguish three
 * cases:
 *   (1) this header is being used to build the InteraxFF plugin shared library
 *       (dllexport)
 *   (2) this header is being used by a *client* of the shared library
 *       (dllimport)
 *   (3) we are building the static library, or the client is being compiled
 *       with the expectation of linking with the static library (nothing
 *       special needed)
 * In the CMake script for building this library, we have to define one of the
 * symbols INTERAX_BUILDING_SHARED_LIBRARY or INTERAX_BUILDING_STATIC_LIBRARY.
 * Client code normally has no special symbol defined, in which case we'll
 * assume it wants to use the shared library.
 */

#ifdef _MSC_VER
    #if defined(INTERAX_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_INTERAX __declspec(dllexport)
    #elif defined(INTERAX_BUILDING_STATIC_LIBRARY) || defined(INTERAX_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_INTERAX
    #else
        #define OPENMM_EXPORT_INTERAX __declspec(dllimport)   // i.e., a client of a shared library
    #endif
#else
    #define OPENMM_EXPORT_INTERAX // Linux, Mac
#endif

#endif // OPENMM_WINDOWSEXPORTINTERAX_H_
