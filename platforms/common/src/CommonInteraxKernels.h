#ifndef COMMON_INTERAX_KERNELS_H_
#define COMMON_INTERAX_KERNELS_H_

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

#include "InteraxKernels.h"
#include "internal/InteraxPairTerms.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * This is the common (CUDA and OpenCL) implementation of the InteraxFF pair forces. Each force adds its interaction
 * to the nonbonded kernel of the context (NonbondedUtilities), so all pair forces with the same cutoff and
 * exclusions are evaluated together, in one pass over the neighbor list.
 */
class CommonInteraxPairForceKernel {
public:
    CommonInteraxPairForceKernel(OpenMM::ComputeContext& cc) : cc(cc), hasAlchemicalParticles(false), lastLambda(-1) {
    }
protected:
    /**
     * Add the interaction of a pair force to the nonbonded kernel.
     *
     * @param system        the System the force belongs to
     * @param force         the pair force
     * @param interaction   the code that computes pairEnergy and pairDerivative for a pair (a kernel source with the
     *                      placeholders PARAMS1, PARAMS2, EXTRA1, EXTRA2, LAMBDA, DAMPING_SCALE, SLATER_OVERLAP)
     * @param gate          the condition for a pair to interact, in terms of the same placeholders
     * @param isAlchemical  the placeholder expressions of the alchemical flags of particles 1 and 2 (e.g. "EXTRA1.y")
     * @param overlapWidths the placeholder expressions of the two widths of the Slater overlap, or empty if the
     *                      interaction does not use it
     * @param overlapName   the name of the variable that receives the overlap integral
     * @param taylorScheme  the Taylor scheme of the overlap integral (unused without it)
     * @param params        the per-particle parameters (float4)
     * @param extra         additional per-particle parameters (float2), or an uninitialized array
     */
    void addInteraction(const OpenMM::System& system, const InteraxPairForce& force, const std::string& interaction, const std::string& gate,
            const std::vector<std::string>& isAlchemical, const std::vector<std::string>& overlapWidths, const std::string& overlapName,
            const InteraxPairTerms::SlaterTaylorScheme& taylorScheme, OpenMM::ComputeArray& params, OpenMM::ComputeArray& extra);
    /**
     * Upload the global parameters (the alchemical parameter and one force-specific value) if they have changed.
     */
    void updateGlobals(OpenMM::ContextImpl& context, double value);
    OpenMM::ComputeContext& cc;
    OpenMM::ComputeArray globals;
    std::string alchemicalParameter;
    bool hasAlchemicalParticles;
    double lastLambda, lastValue;
};

/**
 * This kernel is invoked by InteraxDispersionForce to calculate the forces acting on the system and the energy of
 * the system.
 */
class CommonCalcInteraxDispersionForceKernel : public CalcInteraxDispersionForceKernel, public CommonInteraxPairForceKernel {
public:
    CommonCalcInteraxDispersionForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc) :
            CalcInteraxDispersionForceKernel(name, platform), CommonInteraxPairForceKernel(cc) {
    }
    void initialize(const OpenMM::System& system, const InteraxDispersionForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxDispersionForce& force);
private:
    class ForceInfo;
    void setParameters(const InteraxDispersionForce& force);
    OpenMM::ComputeArray params, extra;
    double dampingScale, longRangeCoefficient, alchemicalLongRangeCoefficient;
};

/**
 * This kernel is invoked by InteraxExchangeForce to calculate the forces acting on the system and the energy of
 * the system.
 */
class CommonCalcInteraxExchangeForceKernel : public CalcInteraxExchangeForceKernel, public CommonInteraxPairForceKernel {
public:
    CommonCalcInteraxExchangeForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc) :
            CalcInteraxExchangeForceKernel(name, platform), CommonInteraxPairForceKernel(cc) {
    }
    void initialize(const OpenMM::System& system, const InteraxExchangeForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxExchangeForce& force);
private:
    class ForceInfo;
    void setParameters(const InteraxExchangeForce& force);
    OpenMM::ComputeArray params, extra;
    int overlapMethod;
};

/**
 * This kernel is invoked by InteraxShortRangeForce to calculate the forces acting on the system and the energy of
 * the system.
 */
class CommonCalcInteraxShortRangeForceKernel : public CalcInteraxShortRangeForceKernel, public CommonInteraxPairForceKernel {
public:
    CommonCalcInteraxShortRangeForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc) :
            CalcInteraxShortRangeForceKernel(name, platform), CommonInteraxPairForceKernel(cc) {
    }
    void initialize(const OpenMM::System& system, const InteraxShortRangeForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxShortRangeForce& force);
private:
    class ForceInfo;
    void setParameters(const InteraxShortRangeForce& force);
    OpenMM::ComputeArray params, extra;
    int overlapMethod;
};

/**
 * This kernel is invoked by InteraxThreeBodyForce to calculate the forces acting on the system and the energy of
 * the system. It computes the sites, a neighbor list of the sites, the triples (each from its three sites, without
 * atomic additions per triple), and then distributes the site forces to the particles.
 */
class CommonCalcInteraxThreeBodyForceKernel : public CalcInteraxThreeBodyForceKernel {
public:
    CommonCalcInteraxThreeBodyForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc) :
            CalcInteraxThreeBodyForceKernel(name, platform), cc(cc), numSites(0), numEvaluations(0), hasComputed(false), lastLambda(-1) {
    }
    void initialize(const OpenMM::System& system, const InteraxThreeBodyForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxThreeBodyForce& force);
private:
    class ForceInfo;
    class NeighborListPostComputation;
    // The triples kernel runs one block of TripleBlockSize threads per site, over tiles of TileSize neighbors.
    static const int TripleBlockSize = 128;
    static const int TileSize = 64;
    void setParameters(const InteraxThreeBodyForce& force);
    void checkNeighborListSize();
    OpenMM::ComputeContext& cc;
    int numSites, maxNeighbors, numEvaluations;
    bool periodic, hasAlchemicalSites, hasComputed;
    double scaleFactor, dampingScale, lastLambda;
    std::string alchemicalParameter;
    OpenMM::ComputeArray siteOffsets, siteParticles, siteMolecules, siteParams, sites, neighbors, neighborDeltas, numNeighbors, maxNeighborCount,
            siteForce, globals;
    OpenMM::ComputeKernel sitesKernel, neighborsKernel, triplesKernel, applyForcesKernel;
};

} // namespace InteraxPlugin

#endif /*COMMON_INTERAX_KERNELS_H_*/
