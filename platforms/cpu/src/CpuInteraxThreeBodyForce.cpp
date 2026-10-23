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

#include "CpuInteraxThreeBodyForce.h"
#include "internal/InteraxThreeBodyForceImpl.h"
#include "internal/InteraxThreeBodyTerms.h"
#include <atomic>
#include <cmath>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

CpuInteraxThreeBodyForce::CpuInteraxThreeBodyForce(const InteraxThreeBodyForce& force, int numThreads) {
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
    centroids.resize(numMolecules);
    neighbors.resize(numMolecules);
    threadCentroidForces.resize(numThreads, vector<Vec3>(numMolecules));
    threadMarks.resize(numThreads, vector<int>(numMolecules, -1));
    setParameters(force);
}

void CpuInteraxThreeBodyForce::setParameters(const InteraxThreeBodyForce& force) {
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

Vec3 CpuInteraxThreeBodyForce::getDelta(const Vec3& from, const Vec3& to, const Vec3* boxVectors) const {
    Vec3 delta = to-from;
    if (periodic) {
        delta -= boxVectors[2]*floor(delta[2]/boxVectors[2][2]+0.5);
        delta -= boxVectors[1]*floor(delta[1]/boxVectors[1][1]+0.5);
        delta -= boxVectors[0]*floor(delta[0]/boxVectors[0][0]+0.5);
    }
    return delta;
}

double CpuInteraxThreeBodyForce::calculateIxn(const vector<Vec3>& positions, const Vec3* boxVectors, double lambda,
        vector<Vec3>& forces, ThreadPool& threads) {
    int numMolecules = moleculeParticles.size();
    int numThreads = threads.getNumThreads();
    double cutoff2 = cutoff*cutoff;
    vector<double> threadEnergy(numThreads, 0.0);

    // Compute the centroids, then the neighbors B > A of each molecule A.

    threads.execute([&] (ThreadPool& threads, int threadIndex) {
        for (int i = threadIndex; i < numMolecules; i += numThreads) {
            Vec3 centroid;
            for (int particle : moleculeParticles[i])
                centroid += positions[particle];
            centroids[i] = centroid/moleculeParticles[i].size();
        }
    });
    threads.waitForThreads();
    threads.execute([&] (ThreadPool& threads, int threadIndex) {
        for (int i = threadIndex; i < numMolecules; i += numThreads) {
            neighbors[i].clear();
            for (int j = i+1; j < numMolecules; j++) {
                Vec3 delta = getDelta(centroids[i], centroids[j], boxVectors);
                if (delta.dot(delta) < cutoff2)
                    neighbors[i].push_back(j);
            }
        }
    });
    threads.waitForThreads();

    // Compute the triples. A molecule C is a neighbor of both A and B < C when it is marked with A and appears in the
    // neighbors of B.

    atomic<int> atomicCounter(0);
    threads.execute([&] (ThreadPool& threads, int threadIndex) {
        vector<Vec3>& centroidForces = threadCentroidForces[threadIndex];
        vector<int>& marks = threadMarks[threadIndex];
        for (Vec3& force : centroidForces)
            force = Vec3();
        double energy = 0;
        while (true) {
            int a = atomicCounter++;
            if (a >= numMolecules)
                break;
            for (int b : neighbors[a])
                marks[b] = a;
            for (int b : neighbors[a])
                for (int c : neighbors[b]) {
                    if (marks[c] != a)
                        continue;
                    double c9 = scaleFactor*InteraxThreeBodyForceImpl::computeC9(alpha[a], frequency[a], alpha[b], frequency[b], alpha[c], frequency[c]);
                    if (isAlchemical[a] || isAlchemical[b] || isAlchemical[c])
                        c9 *= lambda;
                    energy += InteraxThreeBodyTerms::computeTriple(getDelta(centroids[a], centroids[b], boxVectors), getDelta(centroids[a], centroids[c], boxVectors),
                            getDelta(centroids[b], centroids[c], boxVectors), c9, centroidForces[a], centroidForces[b], centroidForces[c]);
                }
        }
        threadEnergy[threadIndex] = energy;
    });
    threads.waitForThreads();

    // Share each centroid force equally among the particles of its molecule.

    for (int i = 0; i < numMolecules; i++) {
        Vec3 force;
        for (int t = 0; t < numThreads; t++)
            force += threadCentroidForces[t][i];
        force /= moleculeParticles[i].size();
        for (int particle : moleculeParticles[i])
            forces[particle] += force;
    }
    double energy = 0;
    for (double e : threadEnergy)
        energy += e;
    return energy;
}
