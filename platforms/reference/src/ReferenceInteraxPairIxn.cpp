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

#include "ReferenceInteraxPairIxn.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

ReferenceInteraxPairIxn::ReferenceInteraxPairIxn(const InteraxPairForce& force) {
    int numParticles = force.getNumParticles();
    nonbondedMethod = force.getNonbondedMethod();
    cutoff = force.getCutoffDistance();
    useSwitch = (nonbondedMethod != InteraxPairForce::NoCutoff && force.getUseSwitchingFunction());
    switchingDistance = force.getSwitchingDistance();
    softcoreRadius = force.getSoftcoreRadius();
    exclusions.resize(numParticles);
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
    }
    isAlchemical.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        isAlchemical[i] = force.getParticleIsAlchemical(i);

    // Without a cutoff the interacting pairs never change, so record them once.

    if (nonbondedMethod == InteraxPairForce::NoCutoff)
        for (int i = 0; i < numParticles; i++)
            for (int j = i+1; j < numParticles; j++)
                if (exclusions[i].find(j) == exclusions[i].end())
                    neighborList.push_back(make_pair(i, j));
}
