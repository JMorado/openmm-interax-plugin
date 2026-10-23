#ifndef REFERENCE_INTERAX_PAIR_IXN_H_
#define REFERENCE_INTERAX_PAIR_IXN_H_

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

#include "InteraxPairForce.h"
#include "openmm/Vec3.h"
#include "openmm/reference/ReferenceForce.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <cmath>
#include <set>
#include <vector>

namespace InteraxPlugin {

/**
 * This class evaluates the interactions of an InteraxFF pair force on the Reference platform: it finds the
 * interacting pairs (all non-excluded pairs, or those within the cutoff), applies periodic boundary conditions, the
 * switching function and the alchemical scaling, and accumulates the forces. The energy of a single pair is
 * supplied by the caller.
 */
class ReferenceInteraxPairIxn {
public:
    /**
     * Create a ReferenceInteraxPairIxn for the cutoff, switching, exclusions and alchemical flags of a pair force.
     */
    explicit ReferenceInteraxPairIxn(const InteraxPairForce& force);
    /**
     * Compute the interactions.
     *
     * @param positions      the particle positions
     * @param boxVectors     the periodic box vectors
     * @param lambda         the value of the alchemical parameter
     * @param forces         the forces are added to this
     * @param pairEnergy     a function (i, j, r, energy, dEdR) that sets the energy of the pair of particles i and j at
     *                       distance r and its derivative with respect to r, and returns false if the pair does not
     *                       interact
     * @return the energy
     */
    template <class PairEnergy>
    double calculateIxn(const std::vector<OpenMM::Vec3>& positions, const OpenMM::Vec3* boxVectors, double lambda,
            std::vector<OpenMM::Vec3>& forces, const PairEnergy& pairEnergy);
private:
    InteraxPairForce::NonbondedMethod nonbondedMethod;
    double cutoff, switchingDistance;
    bool useSwitch;
    std::vector<std::set<int> > exclusions;
    std::vector<bool> isAlchemical;
    OpenMM::NeighborList neighborList;
};

template <class PairEnergy>
double ReferenceInteraxPairIxn::calculateIxn(const std::vector<OpenMM::Vec3>& positions, const OpenMM::Vec3* boxVectors, double lambda,
        std::vector<OpenMM::Vec3>& forces, const PairEnergy& pairEnergy) {
    bool periodic = (nonbondedMethod == InteraxPairForce::CutoffPeriodic);
    if (nonbondedMethod != InteraxPairForce::NoCutoff)
        OpenMM::computeNeighborListVoxelHash(neighborList, positions.size(), positions, exclusions, boxVectors, periodic, cutoff, 0.0);
    double totalEnergy = 0;
    for (auto& pair : neighborList) {
        int i = pair.first;
        int j = pair.second;
        double deltaR[OpenMM::ReferenceForce::LastDeltaRIndex];
        if (periodic)
            OpenMM::ReferenceForce::getDeltaRPeriodic(positions[i], positions[j], boxVectors, deltaR);
        else
            OpenMM::ReferenceForce::getDeltaR(positions[i], positions[j], deltaR);
        double r = deltaR[OpenMM::ReferenceForce::RIndex];
        if (nonbondedMethod != InteraxPairForce::NoCutoff && r >= cutoff)
            continue;
        double energy, dEdR;
        if (!pairEnergy(i, j, r, energy, dEdR))
            continue;
        if (useSwitch && r > switchingDistance) {
            double t = (r-switchingDistance)/(cutoff-switchingDistance);
            double switchValue = 1+t*t*t*(-10+t*(15-t*6));
            double switchDeriv = t*t*(-30+t*(60-t*30))/(cutoff-switchingDistance);
            dEdR = dEdR*switchValue + energy*switchDeriv;
            energy *= switchValue;
        }
        double scale = (isAlchemical[i] != isAlchemical[j] ? lambda : 1.0);
        totalEnergy += scale*energy;
        double forceScale = scale*dEdR/r;
        OpenMM::Vec3 force(deltaR[OpenMM::ReferenceForce::XIndex]*forceScale, deltaR[OpenMM::ReferenceForce::YIndex]*forceScale,
                deltaR[OpenMM::ReferenceForce::ZIndex]*forceScale);
        forces[i] += force;
        forces[j] -= force;
    }
    return totalEnergy;
}

} // namespace InteraxPlugin

#endif /*REFERENCE_INTERAX_PAIR_IXN_H_*/
