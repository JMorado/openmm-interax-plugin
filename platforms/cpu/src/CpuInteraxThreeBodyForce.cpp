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
    sites.resize(numSites);
    neighbors.resize(numSites);
    threadSiteForces.resize(numThreads, vector<Vec3>(numSites));
    threadMarks.resize(numThreads, vector<int>(numSites, -1));
    setParameters(force);
}

void CpuInteraxThreeBodyForce::setParameters(const InteraxThreeBodyForce& force) {
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
    dampingRadius = force.getCentroidDampingRadius();
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
    int numSites = siteParticles.size();
    int numThreads = threads.getNumThreads();
    double cutoff2 = cutoff*cutoff;
    vector<double> threadEnergy(numThreads, 0.0);

    // Compute the positions of the sites, then the neighbors B > A of each site A: the sites of other molecules within
    // the cutoff.

    threads.execute([&] (ThreadPool& threads, int threadIndex) {
        for (int i = threadIndex; i < numSites; i += numThreads) {
            Vec3 site;
            for (int particle : siteParticles[i])
                site += positions[particle];
            sites[i] = site/siteParticles[i].size();
        }
    });
    threads.waitForThreads();
    threads.execute([&] (ThreadPool& threads, int threadIndex) {
        for (int i = threadIndex; i < numSites; i += numThreads) {
            neighbors[i].clear();
            for (int j = i+1; j < numSites; j++) {
                if (molecule[i] == molecule[j])
                    continue;
                Vec3 delta = getDelta(sites[i], sites[j], boxVectors);
                if (delta.dot(delta) < cutoff2)
                    neighbors[i].push_back(j);
            }
        }
    });
    threads.waitForThreads();

    // Compute the triples. A site C is a neighbor of both A and B < C when it is marked with A and appears in the
    // neighbors of B.

    atomic<int> atomicCounter(0);
    threads.execute([&] (ThreadPool& threads, int threadIndex) {
        vector<Vec3>& siteForces = threadSiteForces[threadIndex];
        vector<int>& marks = threadMarks[threadIndex];
        for (Vec3& force : siteForces)
            force = Vec3();
        double energy = 0;
        while (true) {
            int a = atomicCounter++;
            if (a >= numSites)
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
                    double rates[3] = {2*dampingScale/(width[a]+width[b]), 2*dampingScale/(width[a]+width[c]), 2*dampingScale/(width[b]+width[c])};
                    energy += InteraxThreeBodyTerms::computeTriple(getDelta(sites[a], sites[b], boxVectors), getDelta(sites[a], sites[c], boxVectors),
                            getDelta(sites[b], sites[c], boxVectors), c9, useDamping ? rates : NULL, siteForces[a], siteForces[b], siteForces[c], dampingRadius);
                }
        }
        threadEnergy[threadIndex] = energy;
    });
    threads.waitForThreads();

    // Share each site force equally among its particles.

    for (int i = 0; i < numSites; i++) {
        Vec3 force;
        for (int t = 0; t < numThreads; t++)
            force += threadSiteForces[t][i];
        force /= siteParticles[i].size();
        for (int particle : siteParticles[i])
            forces[particle] += force;
    }
    double energy = 0;
    for (double e : threadEnergy)
        energy += e;
    return energy;
}
