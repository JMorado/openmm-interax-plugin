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

#include "ReferenceInteraxThreeBodyIxn.h"
#include "internal/InteraxThreeBodyForceImpl.h"
#include "internal/InteraxThreeBodyTerms.h"
#include <cmath>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

ReferenceInteraxThreeBodyIxn::ReferenceInteraxThreeBodyIxn(const InteraxThreeBodyForce& force) {
    int numSites = force.getNumSites();
    siteParticles.resize(numSites);
    molecule.resize(numSites);
    isAlchemical.resize(numSites);
    for (int i = 0; i < numSites; i++) {
        double a, w, s;
        bool alchemical;
        force.getSiteParameters(i, siteParticles[i], a, w, s, molecule[i], alchemical);
        if (molecule[i] < 0)
            molecule[i] = -1-i; // A molecule of its own
        isAlchemical[i] = alchemical;
    }
    cutoff = force.getCutoffDistance();
    periodic = force.usesPeriodicBoundaryConditions();
    useDamping = (force.getDampingMethod() == InteraxThreeBodyForce::TangToennies);
    setParameters(force);
}

void ReferenceInteraxThreeBodyIxn::setParameters(const InteraxThreeBodyForce& force) {
    int numSites = force.getNumSites();
    alpha.resize(numSites);
    frequency.resize(numSites);
    width.resize(numSites);
    for (int i = 0; i < numSites; i++) {
        vector<int> particles;
        int m;
        bool alchemical;
        force.getSiteParameters(i, particles, alpha[i], frequency[i], width[i], m, alchemical);
    }
    scaleFactor = force.getScaleFactor();
    dampingScale = force.getDampingScale();
}

Vec3 ReferenceInteraxThreeBodyIxn::getDelta(const Vec3& from, const Vec3& to, const Vec3* boxVectors) const {
    Vec3 delta = to-from;
    if (periodic) {
        delta -= boxVectors[2]*floor(delta[2]/boxVectors[2][2]+0.5);
        delta -= boxVectors[1]*floor(delta[1]/boxVectors[1][1]+0.5);
        delta -= boxVectors[0]*floor(delta[0]/boxVectors[0][0]+0.5);
    }
    return delta;
}

double ReferenceInteraxThreeBodyIxn::calculateIxn(const vector<Vec3>& positions, const Vec3* boxVectors, double lambda, vector<Vec3>& forces) const {
    // Compute the position of each site.

    int numSites = siteParticles.size();
    vector<Vec3> sites(numSites);
    for (int i = 0; i < numSites; i++) {
        for (int particle : siteParticles[i])
            sites[i] += positions[particle];
        sites[i] /= siteParticles[i].size();
    }

    // Find the pairs of sites of different molecules within the cutoff.

    double cutoff2 = cutoff*cutoff;
    vector<vector<bool> > isNeighbor(numSites, vector<bool>(numSites, false));
    for (int i = 0; i < numSites; i++)
        for (int j = i+1; j < numSites; j++) {
            Vec3 delta = getDelta(sites[i], sites[j], boxVectors);
            isNeighbor[i][j] = isNeighbor[j][i] = (molecule[i] != molecule[j] && delta.dot(delta) < cutoff2);
        }

    // Loop over the triples whose three pairs are all neighbors.

    vector<Vec3> siteForces(numSites);
    double energy = 0;
    for (int i = 0; i < numSites; i++)
        for (int j = i+1; j < numSites; j++) {
            if (!isNeighbor[i][j])
                continue;
            for (int k = j+1; k < numSites; k++) {
                if (!isNeighbor[i][k] || !isNeighbor[j][k])
                    continue;
                double c9 = scaleFactor*InteraxThreeBodyForceImpl::computeC9(alpha[i], frequency[i], alpha[j], frequency[j], alpha[k], frequency[k]);
                if (isAlchemical[i] || isAlchemical[j] || isAlchemical[k])
                    c9 *= lambda;
                double rates[3] = {2*dampingScale/(width[i]+width[j]), 2*dampingScale/(width[i]+width[k]), 2*dampingScale/(width[j]+width[k])};
                energy += InteraxThreeBodyTerms::computeTriple(getDelta(sites[i], sites[j], boxVectors), getDelta(sites[i], sites[k], boxVectors),
                        getDelta(sites[j], sites[k], boxVectors), c9, useDamping ? rates : NULL, siteForces[i], siteForces[j], siteForces[k]);
            }
        }

    // Share each site force equally among its particles.

    for (int i = 0; i < numSites; i++) {
        Vec3 force = siteForces[i]/siteParticles[i].size();
        for (int particle : siteParticles[i])
            forces[particle] += force;
    }
    return energy;
}
