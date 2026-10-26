#ifndef OPENMM_CPU_INTERAX_THREE_BODY_FORCE_H_
#define OPENMM_CPU_INTERAX_THREE_BODY_FORCE_H_

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

#include "InteraxThreeBodyForce.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ThreadPool.h"
#include <atomic>
#include <vector>

namespace InteraxPlugin {

/**
 * This class implements InteraxThreeBodyForce on the CPU platform.  Each site is the centroid of a group of particles.
 * The threads take sites A in turn and compute every triple A < B < C of sites of different molecules with all three
 * distances within the cutoff: each thread marks the neighbors of A, then for each neighbor B of A visits the neighbors
 * C of B that are also marked, so the cost is the sum over sites of the number of pairs of neighbors.
 */
class CpuInteraxThreeBodyForce {
public:
    /**
     * Create a new CpuInteraxThreeBodyForce.
     *
     * @param force      the InteraxThreeBodyForce to create it for
     * @param threads    the thread pool to use
     */
    CpuInteraxThreeBodyForce(const InteraxThreeBodyForce& force, OpenMM::ThreadPool& threads);
    /**
     * Copy the polarizabilities, frequencies and widths of the sites, the scale factor, the damping scale and the
     * centroid damping radius from a force.
     *
     * @param force      the InteraxThreeBodyForce to copy the parameters from
     */
    void setParameters(const InteraxThreeBodyForce& force);
    /**
     * Calculate the interaction.
     *
     * @param positions      the particle positions
     * @param boxVectors     the vectors defining the periodic box
     * @param lambda         the value of the alchemical parameter
     * @param forces         force array (forces added)
     * @param totalEnergy    the energy is added to this
     */
    void calculateIxn(const std::vector<OpenMM::Vec3>& positions, const OpenMM::Vec3* boxVectors, double lambda,
            std::vector<OpenMM::Vec3>& forces, double& totalEnergy);
private:
    std::vector<std::vector<int> > siteParticles;
    std::vector<int> molecule;
    std::vector<double> alpha, frequency, width;
    std::vector<bool> isAlchemical;
    double cutoff, scaleFactor, dampingScale, dampingRadius;
    bool periodic, useDamping;
    OpenMM::ThreadPool& threads;
    std::vector<OpenMM::Vec3> sites;
    std::vector<std::vector<int> > neighbors;
    std::vector<std::vector<OpenMM::Vec3> > threadSiteForces;
    std::vector<std::vector<int> > threadMarks;
    std::vector<double> threadEnergy;
    // The following variables are used to make information accessible to the individual threads.  The threads copy
    // them before the loop over triples, so the cache line of atomicCounter is only touched by the increments.
    const OpenMM::Vec3* positions;
    const OpenMM::Vec3* boxVectors;
    double lambda;
    std::atomic<int> atomicCounter;

    /**
     * This routine contains the code executed by each thread.
     */
    void threadComputeForce(OpenMM::ThreadPool& threads, int threadIndex);

    /**
     * Compute the displacement between two points, optionally using periodic boundary conditions.
     */
    OpenMM::Vec3 computeDelta(const OpenMM::Vec3& from, const OpenMM::Vec3& to, const OpenMM::Vec3* boxVectors) const;
};

} // namespace InteraxPlugin

#endif /*OPENMM_CPU_INTERAX_THREE_BODY_FORCE_H_*/
