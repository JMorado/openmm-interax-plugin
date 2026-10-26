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

#include "CommonInteraxKernels.h"
#include "CommonInteraxKernelSources.h"
#include "internal/InteraxPairForceImpl.h"
#include "internal/InteraxPairTerms.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

/**
 * Create an array of per-particle or per-molecule parameters in the precision of the context. The values are always
 * given in double precision (mm_double4, mm_double2 or double) and converted on upload in single precision.
 */
static void initializeParameters(ComputeContext& cc, ComputeArray& array, int size, int components, const string& name) {
    int elementSize = components*(cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    array.initialize(cc, size, elementSize, name);
}

/**
 * The component type of the parameter arrays in the kernels.
 */
static string getParameterType(ComputeContext& cc) {
    return (cc.getUseDoublePrecision() ? "double" : "float");
}

/* -------------------------------------------------------------------------- *
 *                     Parameters of the pair forces                          *
 * -------------------------------------------------------------------------- */

// All parameters of a particle, alchemical flag included, as compared by the ForceInfo classes.

static vector<double> getParticleParameters(const InteraxDispersionForce& force, int index) {
    vector<double> p(6);
    bool isAlchemical;
    force.getParticleParameters(index, p[0], p[1], p[2], p[3], p[4], isAlchemical);
    p[5] = isAlchemical;
    return p;
}

static vector<double> getParticleParameters(const InteraxExchangeForce& force, int index) {
    vector<double> p(3);
    bool isAlchemical;
    force.getParticleParameters(index, p[0], p[1], isAlchemical);
    p[2] = isAlchemical;
    return p;
}

static vector<double> getParticleParameters(const InteraxShortRangeForce& force, int index) {
    vector<double> p(4);
    bool isAlchemical;
    force.getParticleParameters(index, p[0], p[1], p[2], isAlchemical);
    p[3] = isAlchemical;
    return p;
}

/**
 * The ComputeForceInfo of a pair force: particles are interchangeable if all their parameters are equal, and the
 * excluded pairs are the groups, as for CustomNonbondedForce.
 */
template <class ForceType>
class PairForceInfo : public ComputeForceInfo {
public:
    PairForceInfo(const ForceType& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        return getParticleParameters(force, particle1) == getParticleParameters(force, particle2);
    }
    int getNumParticleGroups() {
        return force.getNumExclusions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        force.getExclusionParticles(index, particle1, particle2);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const ForceType& force;
};

void CommonInteraxPairForceKernel::addInteraction(const System& system, const InteraxPairForce& force, const string& interaction, const string& gate,
        const vector<string>& isAlchemical, const vector<string>& overlapWidths, const string& overlapName,
        const InteraxPairTerms::SlaterTaylorScheme& taylorScheme, ComputeArray& params, ComputeArray& extra) {
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "interax"+cc.intToString(forceIndex)+"_";
    alchemicalParameter = force.getAlchemicalParameterName();
    for (int i = 0; i < force.getNumParticles(); i++)
        hasAlchemicalParticles |= force.getParticleIsAlchemical(i);

    // Every particle excludes itself and its excluded partners.

    int numParticles = force.getNumParticles();
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++)
        exclusionList[i].push_back(i);
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusionList[particle1].push_back(particle2);
        exclusionList[particle2].push_back(particle1);
    }

    // Assemble the code: the start of the pair loop body, the interaction, and the switching, alchemical scaling and
    // accumulation. First expand the snippets, then name the parameters of this force.

    bool useCutoff = (force.getNonbondedMethod() != InteraxPairForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() == InteraxPairForce::CutoffPeriodic);
    bool useSwitch = (useCutoff && force.getUseSwitchingFunction());
    map<string, string> snippets;
    snippets["PAIR_GATE"] = gate;
    snippets["IS_ALCHEMICAL1"] = isAlchemical[0];
    snippets["IS_ALCHEMICAL2"] = isAlchemical[1];
    if (!overlapWidths.empty()) {
        map<string, string> overlap;
        overlap["OVERLAP_WIDTH1"] = overlapWidths[0];
        overlap["OVERLAP_WIDTH2"] = overlapWidths[1];
        overlap["OVERLAP_DERIV"] = overlapName+"Deriv";
        overlap["OVERLAP"] = overlapName;
        overlap["TAYLOR_ORDER"] = cc.intToString(taylorScheme.order);
        overlap["TAYLOR_ABOUT_SMALLER_WIDTH"] = (taylorScheme.aboutSmallerWidth ? "1" : "0");
        overlap["TAYLOR_MAX_DELTA"] = cc.doubleToString(taylorScheme.maxDelta);
        overlap["TAYLOR_MAX_T"] = cc.doubleToString(taylorScheme.maxT);
        overlap["TAYLOR_MAX_TX"] = cc.doubleToString(min(taylorScheme.maxTX, 1e30)); // No bound for the InteraxFF schemes
        overlap["ATOM_INDEX"] = prefix+"atomIndex";
        snippets["SLATER_OVERLAP"] = cc.replaceStrings(CommonInteraxKernelSources::interaxSlaterOverlap, overlap);
    }
    string source = CommonInteraxKernelSources::interaxPairBegin+interaction+CommonInteraxKernelSources::interaxPairEnd;
    source = cc.replaceStrings(source, snippets);
    map<string, string> names;
    names["PARAMS1"] = prefix+"params1";
    names["PARAMS2"] = prefix+"params2";
    names["EXTRA1"] = prefix+"extra1";
    names["EXTRA2"] = prefix+"extra2";
    names["LAMBDA"] = prefix+"globals[0]";
    names["DAMPING_SCALE"] = prefix+"globals[1]";
    names["USE_SWITCH"] = (useSwitch ? "1" : "0");
    double softcoreRadius = force.getSoftcoreRadius();
    names["USE_SOFTCORE"] = (softcoreRadius > 0 && hasAlchemicalParticles ? "1" : "0");
    names["SOFTCORE_RADIUS6"] = cc.doubleToString(pow(softcoreRadius, 6.0));
    if (useSwitch) {
        names["SWITCH_CUTOFF"] = cc.doubleToString(force.getSwitchingDistance());
        names["INV_SWITCH_WIDTH"] = cc.doubleToString(1/(force.getCutoffDistance()-force.getSwitchingDistance()));
    }
    source = cc.replaceStrings(source, names);

    // Add the interaction, its per-particle parameters and the global parameters to the nonbonded kernel.

    NonbondedUtilities& nonbonded = cc.getNonbondedUtilities();
    nonbonded.addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup());
    nonbonded.addParameter(ComputeParameterInfo(params, prefix+"params", getParameterType(cc), 4));
    if (extra.isInitialized())
        nonbonded.addParameter(ComputeParameterInfo(extra, prefix+"extra", getParameterType(cc), 2));
    if (!overlapWidths.empty() && !taylorScheme.aboutSmallerWidth) {
        // The InteraxFF Taylor schemes need the original indices of the atoms, which the context may reorder.
        nonbonded.addArgument(ComputeParameterInfo(cc.getAtomIndexArray(), prefix+"atomIndex", "int", 1));
    }
    initializeParameters(cc, globals, 2, 1, "interaxGlobals");
    nonbonded.addArgument(ComputeParameterInfo(globals, prefix+"globals", getParameterType(cc), 1));
}

void CommonInteraxPairForceKernel::updateGlobals(ContextImpl& context, double value) {
    double lambda = (hasAlchemicalParticles ? context.getParameter(alchemicalParameter) : 1.0);
    if (lambda != lastLambda || value != lastValue) {
        ContextSelector selector(cc);
        vector<double> values = {lambda, value};
        globals.upload(values, true);
        lastLambda = lambda;
        lastValue = value;
    }
}

/* -------------------------------------------------------------------------- *
 *                         InteraxDispersionForce                             *
 * -------------------------------------------------------------------------- */

class CommonCalcInteraxDispersionForceKernel::ForceInfo : public PairForceInfo<InteraxDispersionForce> {
public:
    ForceInfo(const InteraxDispersionForce& force) : PairForceInfo<InteraxDispersionForce>(force) {
    }
};

void CommonCalcInteraxDispersionForceKernel::initialize(const System& system, const InteraxDispersionForce& force) {
    ContextSelector selector(cc);
    initializeParameters(cc, params, cc.getPaddedNumAtoms(), 4, "interaxDispersionParams");
    initializeParameters(cc, extra, cc.getPaddedNumAtoms(), 2, "interaxDispersionExtra");
    setParameters(force);
    addInteraction(system, force, CommonInteraxKernelSources::interaxDispersion, "PARAMS1.w*PARAMS2.w != 0", {"EXTRA1.y", "EXTRA2.y"},
            {}, "", InteraxPairTerms::SlaterTaylorScheme::accurate(true), params, extra);
    cc.addForce(new ForceInfo(force));
}

double CommonCalcInteraxDispersionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    updateGlobals(context, dampingScale);
    if (cc.getContextIndex() != 0 || (longRangeCoefficient == 0 && alchemicalLongRangeCoefficient == 0))
        return 0.0;
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    return (longRangeCoefficient+lastLambda*alchemicalLongRangeCoefficient)/(a[0]*b[1]*c[2]);
}

void CommonCalcInteraxDispersionForceKernel::copyParametersToContext(ContextImpl& context, const InteraxDispersionForce& force) {
    ContextSelector selector(cc);
    if (force.getNumParticles() != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    setParameters(force);
    cc.invalidateMolecules();
}

void CommonCalcInteraxDispersionForceKernel::setParameters(const InteraxDispersionForce& force) {
    vector<mm_double4> paramVector(cc.getPaddedNumAtoms(), mm_double4(0, 0, 0, 0));
    vector<mm_double2> extraVector(cc.getPaddedNumAtoms(), mm_double2(0, 0));
    for (int i = 0; i < force.getNumParticles(); i++) {
        double m1, m2, m3, alpha, width;
        bool isAlchemical;
        force.getParticleParameters(i, m1, m2, m3, alpha, width, isAlchemical);
        paramVector[i] = mm_double4(m1, m2, m3, alpha);
        extraVector[i] = mm_double2(width, isAlchemical ? 1.0 : 0.0);
    }
    params.upload(paramVector, true);
    extra.upload(extraVector, true);
    dampingScale = force.getDampingScale();
    InteraxDispersionForceImpl::calcLongRangeCorrection(force, longRangeCoefficient, alchemicalLongRangeCoefficient);
}

/* -------------------------------------------------------------------------- *
 *                          InteraxExchangeForce                              *
 * -------------------------------------------------------------------------- */

class CommonCalcInteraxExchangeForceKernel::ForceInfo : public PairForceInfo<InteraxExchangeForce> {
public:
    ForceInfo(const InteraxExchangeForce& force) : PairForceInfo<InteraxExchangeForce>(force) {
    }
};

void CommonCalcInteraxExchangeForceKernel::initialize(const System& system, const InteraxExchangeForce& force) {
    ContextSelector selector(cc);
    overlapMethod = force.getOverlapMethod();
    initializeParameters(cc, params, cc.getPaddedNumAtoms(), 4, "interaxExchangeParams");
    setParameters(force);
    addInteraction(system, force, CommonInteraxKernelSources::interaxExchange, "PARAMS1.y*PARAMS2.y != 0", {"PARAMS1.z", "PARAMS2.z"},
            {"PARAMS1.y", "PARAMS2.y"}, "exchangeOverlap", getSlaterTaylorScheme(force, cc.getUseDoublePrecision()), params, extra);
    cc.addForce(new ForceInfo(force));
}

double CommonCalcInteraxExchangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    updateGlobals(context, 0.0);
    return 0.0;
}

void CommonCalcInteraxExchangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxExchangeForce& force) {
    ContextSelector selector(cc);
    if (force.getNumParticles() != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getOverlapMethod() != overlapMethod)
        throw OpenMMException("updateParametersInContext: The overlap method has changed");
    setParameters(force);
    cc.invalidateMolecules();
}

void CommonCalcInteraxExchangeForceKernel::setParameters(const InteraxExchangeForce& force) {
    vector<mm_double4> paramVector(cc.getPaddedNumAtoms(), mm_double4(0, 0, 0, 0));
    for (int i = 0; i < force.getNumParticles(); i++) {
        double amplitude, width;
        bool isAlchemical;
        force.getParticleParameters(i, amplitude, width, isAlchemical);
        paramVector[i] = mm_double4(amplitude, width, isAlchemical ? 1.0 : 0.0, 0.0);
    }
    params.upload(paramVector, true);
}

/* -------------------------------------------------------------------------- *
 *                         InteraxShortRangeForce                             *
 * -------------------------------------------------------------------------- */

class CommonCalcInteraxShortRangeForceKernel::ForceInfo : public PairForceInfo<InteraxShortRangeForce> {
public:
    ForceInfo(const InteraxShortRangeForce& force) : PairForceInfo<InteraxShortRangeForce>(force) {
    }
};

void CommonCalcInteraxShortRangeForceKernel::initialize(const System& system, const InteraxShortRangeForce& force) {
    ContextSelector selector(cc);
    overlapMethod = force.getOverlapMethod();
    initializeParameters(cc, params, cc.getPaddedNumAtoms(), 4, "interaxShortRangeParams");
    setParameters(force);
    addInteraction(system, force, CommonInteraxKernelSources::interaxShortRange, "PARAMS1.z*PARAMS2.z != 0", {"PARAMS1.w", "PARAMS2.w"},
            {"PARAMS1.z", "PARAMS2.z"}, "shortRangeOverlap", getSlaterTaylorScheme(force, cc.getUseDoublePrecision()), params, extra);
    cc.addForce(new ForceInfo(force));
}

double CommonCalcInteraxShortRangeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    updateGlobals(context, 0.0);
    return 0.0;
}

void CommonCalcInteraxShortRangeForceKernel::copyParametersToContext(ContextImpl& context, const InteraxShortRangeForce& force) {
    ContextSelector selector(cc);
    if (force.getNumParticles() != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getOverlapMethod() != overlapMethod)
        throw OpenMMException("updateParametersInContext: The overlap method has changed");
    setParameters(force);
    cc.invalidateMolecules();
}

void CommonCalcInteraxShortRangeForceKernel::setParameters(const InteraxShortRangeForce& force) {
    vector<mm_double4> paramVector(cc.getPaddedNumAtoms(), mm_double4(0, 0, 0, 0));
    for (int i = 0; i < force.getNumParticles(); i++) {
        double k, l, width;
        bool isAlchemical;
        force.getParticleParameters(i, k, l, width, isAlchemical);
        paramVector[i] = mm_double4(k, l, width, isAlchemical ? 1.0 : 0.0);
    }
    params.upload(paramVector, true);
}

/* -------------------------------------------------------------------------- *
 *                          InteraxThreeBodyForce                             *
 * -------------------------------------------------------------------------- */

/**
 * The sites are the groups: two sites are interchangeable if they have the same number of particles and the same
 * parameters. (Their molecules need not be compared: the context only swaps whole identical molecules, which keeps
 * every pair of sites in the same molecule.)
 */
class CommonCalcInteraxThreeBodyForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const InteraxThreeBodyForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumSites();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force.getSiteParameters(index, particles, alpha, frequency, width, molecule, isAlchemical);
    }
    bool areGroupsIdentical(int group1, int group2) {
        vector<int> particles1, particles2;
        double alpha1, alpha2, frequency1, frequency2, width1, width2;
        int molecule1, molecule2;
        bool isAlchemical1, isAlchemical2;
        force.getSiteParameters(group1, particles1, alpha1, frequency1, width1, molecule1, isAlchemical1);
        force.getSiteParameters(group2, particles2, alpha2, frequency2, width2, molecule2, isAlchemical2);
        return (particles1.size() == particles2.size() && alpha1 == alpha2 && frequency1 == frequency2 && width1 == width2 &&
                isAlchemical1 == isAlchemical2);
    }
private:
    const InteraxThreeBodyForce& force;
};

/**
 * After the forces have been computed, check now and then whether the neighbor lists of the sites overflowed, and
 * enlarge them if they did. The forces are correct either way (the triples of a site whose list overflowed are found
 * by scanning all the sites, which is slower), so this check only keeps the lists large enough. Reading the count
 * waits for the GPU, so it is done in the first evaluations and then only every CheckInterval evaluations.
 */
class CommonCalcInteraxThreeBodyForceKernel::NeighborListPostComputation : public ComputeContext::ForcePostComputation {
public:
    NeighborListPostComputation(CommonCalcInteraxThreeBodyForceKernel& owner) : owner(owner) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if (owner.hasComputed) {
            owner.hasComputed = false;
            if (owner.numEvaluations < 10 || owner.numEvaluations%CheckInterval == 0)
                owner.checkNeighborListSize();
            owner.numEvaluations++;
        }
        return 0.0;
    }
private:
    static const int CheckInterval = 100;
    CommonCalcInteraxThreeBodyForceKernel& owner;
};

void CommonCalcInteraxThreeBodyForceKernel::initialize(const System& system, const InteraxThreeBodyForce& force) {
    ContextSelector selector(cc);
    numSites = force.getNumSites();
    alchemicalParameter = force.getAlchemicalParameterName();
    periodic = force.usesPeriodicBoundaryConditions();
    cc.addForce(new ForceInfo(force));
    if (numSites == 0)
        return;

    // Record the particles and the molecule of each site.

    vector<int> offsetVector(numSites+1, 0), particleVector, moleculeVector(numSites);
    hasAlchemicalSites = false;
    for (int i = 0; i < numSites; i++) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force.getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        particleVector.insert(particleVector.end(), particles.begin(), particles.end());
        offsetVector[i+1] = particleVector.size();
        moleculeVector[i] = (molecule < 0 ? -1-i : molecule); // A negative molecule is a molecule of its own
        hasAlchemicalSites |= isAlchemical;
    }
    siteOffsets.initialize<int>(cc, numSites+1, "interaxSiteOffsets");
    siteParticles.initialize<int>(cc, particleVector.size(), "interaxSiteParticles");
    siteMolecules.initialize<int>(cc, numSites, "interaxSiteMolecules");
    siteOffsets.upload(offsetVector);
    siteParticles.upload(particleVector);
    siteMolecules.upload(moleculeVector);
    initializeParameters(cc, siteParams, numSites, 4, "interaxSiteParams");
    initializeParameters(cc, globals, 4, 1, "interaxThreeBodyGlobals");
    setParameters(force);

    // The neighbor lists (each site lists all its neighbors) start small and grow when they overflow
    // (checkNeighborListSize()).

    maxNeighbors = min(numSites, 64);
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
    sites.initialize(cc, numSites, elementSize, "interaxSites");
    neighbors.initialize<int>(cc, numSites*maxNeighbors, "interaxNeighbors");
    neighborDeltas.initialize(cc, numSites*maxNeighbors, elementSize, "interaxNeighborDeltas");
    numNeighbors.initialize<int>(cc, numSites, "interaxNumNeighbors");
    maxNeighborCount.initialize<int>(cc, 1, "interaxMaxNeighborCount");
    siteForce.initialize<long long>(cc, 3*numSites, "interaxSiteForce");
    siteForce.upload(vector<long long>(3*numSites, 0));

    // Create the kernels.

    map<string, string> defines;
    defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    if (periodic)
        defines["USE_PERIODIC"] = "1";
    if (force.getDampingMethod() == InteraxThreeBodyForce::TangToennies)
        defines["USE_DAMPING"] = "1";
    useCentroidDamping = (force.getCentroidDampingRadius() > 0);
    if (useCentroidDamping)
        defines["USE_CENTROID_DAMPING"] = "1";
    defines["TILE_SIZE"] = cc.intToString(TileSize);
    defines["TRIPLE_BLOCK_SIZE"] = cc.intToString(TripleBlockSize);
    ComputeProgram program = cc.compileProgram(CommonInteraxKernelSources::interaxThreeBody, defines);
    sitesKernel = program->createKernel("computeSites");
    sitesKernel->addArg(numSites);
    sitesKernel->addArg(cc.getPosq());
    sitesKernel->addArg(siteOffsets);
    sitesKernel->addArg(siteParticles);
    sitesKernel->addArg(sites);
    sitesKernel->addArg(maxNeighborCount);
    neighborsKernel = program->createKernel("findNeighbors");
    neighborsKernel->addArg(numSites);
    neighborsKernel->addArg(maxNeighbors);
    neighborsKernel->addArg(sites);
    neighborsKernel->addArg(siteMolecules);
    neighborsKernel->addArg(neighbors);
    neighborsKernel->addArg(neighborDeltas);
    neighborsKernel->addArg(numNeighbors);
    neighborsKernel->addArg(maxNeighborCount);
    for (int i = 0; i < 5; i++)
        neighborsKernel->addArg(); // Periodic box size arguments are set when the kernel is executed.
    triplesKernel = program->createKernel("computeTriples");
    triplesKernel->addArg(numSites);
    triplesKernel->addArg(maxNeighbors);
    triplesKernel->addArg(sites);
    triplesKernel->addArg(siteParams);
    triplesKernel->addArg(siteMolecules);
    triplesKernel->addArg(neighbors);
    triplesKernel->addArg(neighborDeltas);
    triplesKernel->addArg(numNeighbors);
    triplesKernel->addArg(siteForce);
    triplesKernel->addArg(cc.getEnergyBuffer());
    triplesKernel->addArg(globals);
    for (int i = 0; i < 5; i++)
        triplesKernel->addArg(); // Periodic box size arguments are set when the kernel is executed.
    applyForcesKernel = program->createKernel("applySiteForces");
    applyForcesKernel->addArg(numSites);
    applyForcesKernel->addArg(siteForce);
    applyForcesKernel->addArg(cc.getLongForceBuffer());
    applyForcesKernel->addArg(siteOffsets);
    applyForcesKernel->addArg(siteParticles);
    cc.addPostComputation(new NeighborListPostComputation(*this));
}

double CommonCalcInteraxThreeBodyForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (numSites == 0 || cc.getContextIndex() != 0)
        return 0.0;
    ContextSelector selector(cc);
    double lambda = (hasAlchemicalSites ? context.getParameter(alchemicalParameter) : 1.0);
    if (lambda != lastLambda) {
        vector<double> values = {lambda, scaleFactor, dampingScale, useCentroidDamping ? 1/centroidDampingRadius : 0.0};
        globals.upload(values, true);
        lastLambda = lambda;
    }
    setPeriodicBoxArgs(cc, neighborsKernel, 8);
    setPeriodicBoxArgs(cc, triplesKernel, 11);
    sitesKernel->execute(numSites);
    neighborsKernel->execute(numSites);
    triplesKernel->execute(numSites*TripleBlockSize, TripleBlockSize);
    applyForcesKernel->execute(numSites);
    hasComputed = true;
    return 0.0;
}

void CommonCalcInteraxThreeBodyForceKernel::checkNeighborListSize() {
    // This waits for the kernels to finish, at the end of the force computation.

    vector<int> count;
    maxNeighborCount.download(count);
    if (count[0] <= maxNeighbors)
        return;
    maxNeighbors = min(numSites, (int) (1.2*count[0])+16);
    neighbors.resize(numSites*maxNeighbors);
    neighborDeltas.resize(numSites*maxNeighbors);
    neighborsKernel->setArg(1, maxNeighbors);
    neighborsKernel->setArg(4, neighbors);
    neighborsKernel->setArg(5, neighborDeltas);
    triplesKernel->setArg(1, maxNeighbors);
    triplesKernel->setArg(5, neighbors);
    triplesKernel->setArg(6, neighborDeltas);
}

void CommonCalcInteraxThreeBodyForceKernel::copyParametersToContext(ContextImpl& context, const InteraxThreeBodyForce& force) {
    ContextSelector selector(cc);
    if (force.getNumSites() != numSites)
        throw OpenMMException("updateParametersInContext: The number of sites has changed");
    if (numSites > 0 && (force.getCentroidDampingRadius() > 0) != useCentroidDamping)
        throw OpenMMException("updateParametersInContext: The centroid damping cannot be switched on or off");
    if (numSites > 0)
        setParameters(force);
    cc.invalidateMolecules();
}

void CommonCalcInteraxThreeBodyForceKernel::setParameters(const InteraxThreeBodyForce& force) {
    vector<mm_double4> paramVector(numSites);
    for (int i = 0; i < numSites; i++) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force.getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        paramVector[i] = mm_double4(alpha, frequency, width, isAlchemical ? 1.0 : 0.0);
    }
    siteParams.upload(paramVector, true);
    scaleFactor = force.getScaleFactor();
    dampingScale = force.getDampingScale();
    centroidDampingRadius = force.getCentroidDampingRadius();
    lastLambda = -1; // Force the global parameters to be uploaded with the new values.
}
