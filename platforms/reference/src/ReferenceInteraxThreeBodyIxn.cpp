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
#include <cmath>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

ReferenceInteraxThreeBodyIxn::ReferenceInteraxThreeBodyIxn(const InteraxThreeBodyForce& force) {
    int numMolecules = force.getNumMolecules();
    moleculeParticles.resize(numMolecules);
    isAlchemical.resize(numMolecules);
    for (int i = 0; i < numMolecules; i++) {
        double a, w;
        bool alchemical;
        force.getMoleculeParameters(i, moleculeParticles[i], a, w, alchemical);
        isAlchemical[i] = alchemical;
    }
    cutoff = force.getCutoffDistance();
    periodic = force.usesPeriodicBoundaryConditions();
    setParameters(force);
}

void ReferenceInteraxThreeBodyIxn::setParameters(const InteraxThreeBodyForce& force) {
    int numMolecules = force.getNumMolecules();
    alpha.resize(numMolecules);
    frequency.resize(numMolecules);
    for (int i = 0; i < numMolecules; i++) {
        vector<int> particles;
        bool alchemical;
        force.getMoleculeParameters(i, particles, alpha[i], frequency[i], alchemical);
    }
    scaleFactor = force.getScaleFactor();
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
    // Compute the centroid of each molecule.

    int numMolecules = moleculeParticles.size();
    vector<Vec3> centroids(numMolecules);
    for (int i = 0; i < numMolecules; i++) {
        for (int particle : moleculeParticles[i])
            centroids[i] += positions[particle];
        centroids[i] /= moleculeParticles[i].size();
    }

    // Find the pairs of molecules within the cutoff.

    double cutoff2 = cutoff*cutoff;
    vector<vector<bool> > isNeighbor(numMolecules, vector<bool>(numMolecules, false));
    for (int i = 0; i < numMolecules; i++)
        for (int j = i+1; j < numMolecules; j++) {
            Vec3 delta = getDelta(centroids[i], centroids[j], boxVectors);
            isNeighbor[i][j] = isNeighbor[j][i] = (delta.dot(delta) < cutoff2);
        }

    // Loop over triples whose three pairs are all within the cutoff. The energy is written in terms of the minimum
    // image vectors a = AB, b = AC, c = BC, so that the angle at each vertex is the angle between the nearest images
    // of its two neighbors (as in CustomCentroidBondForce, also when the three images do not close into a triangle).
    // With u, v, w their squared lengths, P = uvw, and p = a.b, q = a.c, s = b.c, the product of the three cosines
    // is -pqs/P, so E = C9 [P^(-3/2) - 3 pqs P^(-5/2)].

    vector<Vec3> centroidForces(numMolecules);
    double energy = 0;
    for (int i = 0; i < numMolecules; i++)
        for (int j = i+1; j < numMolecules; j++) {
            if (!isNeighbor[i][j])
                continue;
            for (int k = j+1; k < numMolecules; k++) {
                if (!isNeighbor[i][k] || !isNeighbor[j][k])
                    continue;
                Vec3 a = getDelta(centroids[i], centroids[j], boxVectors);
                Vec3 b = getDelta(centroids[i], centroids[k], boxVectors);
                Vec3 c = getDelta(centroids[j], centroids[k], boxVectors);
                double u = a.dot(a), v = b.dot(b), w = c.dot(c);
                double p = a.dot(b), q = a.dot(c), s = b.dot(c);
                double invP = 1/(u*v*w);
                double invP32 = invP*sqrt(invP);
                double invP52 = invP32*invP;
                double c9 = scaleFactor*InteraxThreeBodyForceImpl::computeC9(alpha[i], frequency[i], alpha[j], frequency[j], alpha[k], frequency[k]);
                if (isAlchemical[i] || isAlchemical[j] || isAlchemical[k])
                    c9 *= lambda;
                double pqs = p*q*s;
                energy += c9*(invP32 - 3*pqs*invP52);

                // Gradients with respect to a, b and c.

                double radial = c9*(-3*invP32 + 15*pqs*invP52);
                double cross = -3*c9*invP52;
                Vec3 dEda = a*(radial/u) + (b*(q*s) + c*(p*s))*cross;
                Vec3 dEdb = b*(radial/v) + (a*(q*s) + c*(p*q))*cross;
                Vec3 dEdc = c*(radial/w) + (a*(p*s) + b*(p*q))*cross;
                centroidForces[i] += dEda + dEdb;
                centroidForces[j] += dEdc - dEda;
                centroidForces[k] -= dEdb + dEdc;
            }
        }

    // Share each centroid force equally among the particles of its molecule.

    for (int i = 0; i < numMolecules; i++) {
        Vec3 force = centroidForces[i]/moleculeParticles[i].size();
        for (int particle : moleculeParticles[i])
            forces[particle] += force;
    }
    return energy;
}
