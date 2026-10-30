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

/**
 * Tests of what the InteraxFF pair forces (InteraxDispersionForce, InteraxExchangeForce and InteraxShortRangeForce)
 * have in common: the pair energy itself, exclusions, cutoffs, periodic boundary conditions (rectangular and
 * triclinic), the switching function, alchemical scaling, the soft-core distance, updateParametersInContext() and
 * the checks of the settings. Every expected value is computed here, in long double, from the formulas of
 * docs/SPEC.md and InteraxPairForce.h.
 *
 * The tests are templates over a model of the force, a class with these static members:
 *
 *   typedef ... Force;                       the force class
 *   NUM_TYPES                                the number of particle types
 *   Force* createForce()                     create a force with its force-wide parameters set
 *   void addParticle(Force&, int type, bool isAlchemical)
 *   bool computePairEnergy(const Force&, int i, int j, Real r, Real& energy)
 *                                            the energy of a pair of particles at distance r, before switching
 *                                            and alchemical scaling, or false if the pair does not interact
 *   void changeParameters(Force&)            change the parameters that updateParametersInContext() copies
 */

#include "InteraxTestUtilities.h"
#include <set>
#include <utility>

double energyTol, forceTol, consistencyTol;

/**
 * The energy of a pair of particles at distance r with every option of the force applied: cutoff, soft-core
 * distance, switching function and alchemical scaling.
 */
template <class Model>
Real computeReferencePairEnergy(const typename Model::Force& force, int i, int j, Real r, Real lambda, bool applySwitch=true) {
    bool useCutoff = (force.getNonbondedMethod() != InteraxPairForce::NoCutoff);
    if (useCutoff && r >= force.getCutoffDistance())
        return 0;
    bool alchemicalPair = (force.getParticleIsAlchemical(i) != force.getParticleIsAlchemical(j));
    Real rEff = r;
    Real radius = force.getSoftcoreRadius();
    if (alchemicalPair && radius > 0)
        rEff = powl(powl(r, 6) + (1-lambda)*powl(radius, 6), 1/6.0L);
    Real energy;
    if (!Model::computePairEnergy(force, i, j, rEff, energy))
        return 0;
    if (useCutoff && force.getUseSwitchingFunction() && applySwitch)
        energy *= switchingFunction(r, force.getSwitchingDistance(), force.getCutoffDistance());
    if (alchemicalPair)
        energy *= lambda;
    return energy;
}

/**
 * Compute the energy and forces of a pair force by looping over all pairs of particles. The force of each pair is
 * the radial derivative of its energy, from a five-point finite difference in long double.
 *
 * @param box             the periodic box vectors (used with CutoffPeriodic)
 * @param lambda          the value of the alchemical parameter
 * @param[out] forces     the forces on the particles
 * @param[out] absEnergy  if not NULL, the sum of the magnitudes of the pair energies without the switching function
 *                        (the scale of rounding errors)
 * @return the energy
 */
template <class Model>
Real computeReferenceEnergy(const typename Model::Force& force, const vector<Vec3>& positions, const Vec3* box, Real lambda,
        vector<Vec3>& forces, Real* absEnergy=NULL) {
    int numParticles = force.getNumParticles();
    set<pair<int, int> > exclusions;
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusions.insert(make_pair(min(particle1, particle2), max(particle1, particle2)));
    }
    bool periodic = (force.getNonbondedMethod() == InteraxPairForce::CutoffPeriodic);
    forces.assign(numParticles, Vec3());
    Real energy = 0, sumAbs = 0;
    for (int i = 0; i < numParticles; i++)
        for (int j = i+1; j < numParticles; j++) {
            if (exclusions.find(make_pair(i, j)) != exclusions.end())
                continue;
            Vec3 delta = positions[j]-positions[i];
            if (periodic)
                delta = getMinimumImage(delta, box);
            Real r = sqrtl((Real) delta[0]*delta[0] + (Real) delta[1]*delta[1] + (Real) delta[2]*delta[2]);
            auto pairEnergy = [&] (Real distance) {
                return computeReferencePairEnergy<Model>(force, i, j, distance, lambda);
            };
            Real e = pairEnergy(r);
            if (e == 0)
                continue;
            energy += e;
            sumAbs += fabsl(computeReferencePairEnergy<Model>(force, i, j, r, lambda, false));
            Real dEdR = fivePointDerivative(pairEnergy, r, 1e-5L);
            Vec3 f = delta*(double) (dEdR/r);
            forces[i] += f;
            forces[j] -= f;
        }
    if (absEnergy != NULL)
        *absEnergy = sumAbs;
    return energy;
}

/**
 * Compare the energy and forces of a Context with the reference values for the force.
 */
template <class Model>
void compareWithReference(Context& context, const typename Model::Force& force, Real lambda, double energyTolerance, double forceTolerance) {
    State state = context.getState(State::Positions | State::Energy | State::Forces);
    Vec3 box[3];
    state.getPeriodicBoxVectors(box[0], box[1], box[2]);
    vector<Vec3> expectedForces;
    Real absEnergy;
    Real expectedEnergy = computeReferenceEnergy<Model>(force, state.getPositions(), box, lambda, expectedForces, &absEnergy);
    double energy = state.getPotentialEnergy();
    if (!(fabsl(energy-expectedEnergy) <= energyTolerance*absEnergy)) {
        stringstream details;
        details.precision(17);
        details << "Expected energy " << (double) expectedEnergy << ", found " << energy << " (sum of |pair energies| " << (double) absEnergy << ")";
        throwException(__FILE__, __LINE__, details.str());
    }
    assertForcesEqual(expectedForces, state.getForces(), forceTolerance);
}

/**
 * Create a System with a force of the model and the given particle types.
 */
template <class Model>
typename Model::Force* createSystem(System& system, const vector<int>& types, const vector<bool>& isAlchemical=vector<bool>()) {
    typename Model::Force* force = Model::createForce();
    for (int i = 0; i < types.size(); i++) {
        system.addParticle(1.0);
        Model::addParticle(*force, types[i], isAlchemical.empty() ? false : isAlchemical[i]);
    }
    system.addForce(force);
    return force;
}

/**
 * Test the energy and forces of every pair of particle types over a range of distances.
 */
template <class Model>
void testPairEnergies() {
    const double distances[] = {0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7};
    for (int type1 = 0; type1 < Model::NUM_TYPES; type1++)
        for (int type2 = type1; type2 < Model::NUM_TYPES; type2++) {
            System system;
            typename Model::Force* force = createSystem<Model>(system, {type1, type2});
            VerletIntegrator integrator(0.001);
            Context context(system, integrator, platform);
            for (double r : distances) {
                // Put the pair along a general direction, so that every force component is tested.
                Vec3 direction = Vec3(0.3, -0.5, 0.8)/sqrt(0.98);
                context.setPositions({Vec3(0.1, 0.2, -0.1), Vec3(0.1, 0.2, -0.1)+direction*r});
                Real expected = computeReferencePairEnergy<Model>(*force, 0, 1, r, 1);
                State state = getEnergyAndForces(context);
                ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
                Real dEdR = fivePointDerivative([&] (Real x) {return computeReferencePairEnergy<Model>(*force, 0, 1, x, 1);}, (Real) r, 1e-5L);
                Vec3 expectedForce = direction*(double) dEdR;
                assertForcesEqual({expectedForce, -expectedForce}, state.getForces(), forceTol);
            }
        }
}

/**
 * Test that excluded pairs do not interact, whichever order their particles are given in.
 */
template <class Model>
void testExclusions() {
    System system;
    typename Model::Force* force = createSystem<Model>(system, {0, 1, 2, 3, 0});
    force->addExclusion(0, 1);
    force->addExclusion(3, 2);
    force->addExclusion(4, 0);
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.25, 0, 0), Vec3(0, 0.3, 0), Vec3(0.1, 0.1, 0.2), Vec3(-0.2, 0.05, 0.1)};
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    compareWithReference<Model>(context, *force, 1, energyTol, forceTol);

    // Make sure the excluded pairs were really omitted, by comparing with the sum of the interacting pairs.

    Real expected = 0;
    for (int i = 0; i < 5; i++)
        for (int j = i+1; j < 5; j++)
            if (!((i == 0 && j == 1) || (i == 2 && j == 3) || (i == 0 && j == 4)))
                expected += computeReferencePairEnergy<Model>(*force, i, j, sqrt((positions[i]-positions[j]).dot(positions[i]-positions[j])), 1);
    ASSERT_EQUAL_REL(expected, getEnergyAndForces(context).getPotentialEnergy(), energyTol);
}

/**
 * Test a non-periodic cutoff.
 */
template <class Model>
void testCutoff() {
    System system;
    typename Model::Force* force = createSystem<Model>(system, {0, 1, 2});
    force->setNonbondedMethod(InteraxPairForce::CutoffNonPeriodic);
    force->setCutoffDistance(0.35);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    // The pairs 0-1 and 1-2 are inside the cutoff, the pair 0-2 is outside it.

    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.2, 0, 0), Vec3(0.45, 0, 0)};
    context.setPositions(positions);
    State state = getEnergyAndForces(context);
    Real expected = computeReferencePairEnergy<Model>(*force, 0, 1, 0.2, 1) + computeReferencePairEnergy<Model>(*force, 1, 2, 0.25, 1);
    ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
    compareWithReference<Model>(context, *force, 1, energyTol, forceTol);

    // With all pairs outside the cutoff, there is no interaction.

    context.setPositions({Vec3(0, 0, 0), Vec3(0.36, 0, 0), Vec3(0, 0.36, 0.36)});
    state = getEnergyAndForces(context);
    ASSERT_EQUAL(0.0, state.getPotentialEnergy());
    for (int i = 0; i < 3; i++)
        ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[i], 0);
}

/**
 * Test periodic boundary conditions: particles interact with the nearest image of each other.
 */
template <class Model>
void testPeriodic() {
    System system;
    typename Model::Force* force = createSystem<Model>(system, {0, 1, 3});
    force->setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
    force->setCutoffDistance(0.5);
    system.setDefaultPeriodicBoxVectors(Vec3(1.2, 0, 0), Vec3(0, 1.1, 0), Vec3(0, 0, 1.3));
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0.05, 0.5, 0.6), Vec3(1.1, 0.55, 0.6), Vec3(0.1, 1.05, 2.6+0.05)});
    State state = getEnergyAndForces(context);

    // The nearest images: 0-1 at (-0.15, 0.05, 0); 0-2 at (0.05, -0.55, -0.55) and 1-2 at (0.2, 0.5, -0.55) are
    // outside the cutoff.

    Real expected = computeReferencePairEnergy<Model>(*force, 0, 1, sqrtl(0.15L*0.15L+0.05L*0.05L), 1);
    ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
    compareWithReference<Model>(context, *force, 1, energyTol, forceTol);
}

/**
 * Test periodic boundary conditions in a triclinic box.
 */
template <class Model>
void testTriclinic() {
    const int numParticles = 50;
    Vec3 box[3] = {Vec3(2.0, 0, 0), Vec3(0.4, 2.1, 0), Vec3(-0.5, 0.7, 1.9)};
    System system;
    system.setDefaultPeriodicBoxVectors(box[0], box[1], box[2]);
    vector<int> types;
    for (int i = 0; i < numParticles; i++)
        types.push_back(i%Model::NUM_TYPES);
    typename Model::Force* force = createSystem<Model>(system, types);
    force->setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
    force->setCutoffDistance(0.75);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(5, sfmt);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        // Keep the particles more than 0.12 nm apart, and spread them over several periodic copies of the box.
        bool accepted = false;
        while (!accepted) {
            positions[i] = box[0]*(3*random01(sfmt)-1) + box[1]*(3*random01(sfmt)-1) + box[2]*(3*random01(sfmt)-1);
            accepted = true;
            for (int j = 0; j < i; j++) {
                Vec3 delta = getMinimumImage(positions[i]-positions[j], box);
                if (delta.dot(delta) < 0.12*0.12)
                    accepted = false;
            }
        }
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    compareWithReference<Model>(context, *force, 1, energyTol, forceTol);
}

/**
 * Test the switching function: the energy is S(r) E(r) between the switching distance and the cutoff.
 */
template <class Model>
void testSwitchingFunction(InteraxPairForce::NonbondedMethod method) {
    System system;
    typename Model::Force* force = createSystem<Model>(system, {0, 2});
    force->setNonbondedMethod(method);
    force->setCutoffDistance(0.5);
    force->setUseSwitchingFunction(true);
    force->setSwitchingDistance(0.3);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    for (double r = 0.25; r < 0.55; r += 0.02) {
        context.setPositions({Vec3(0, 0, 0), Vec3(0, r, 0)});
        State state = getEnergyAndForces(context);
        Real expected = 0, dEdR = 0, unswitched = 0, unswitchedDerivative = 0;
        if (r < 0.5) {
            auto energy = [&] (Real x) {Real e; Model::computePairEnergy(*force, 0, 1, x, e); return e;};
            auto switchedEnergy = [&] (Real x) {return energy(x)*switchingFunction(x, 0.3, 0.5);};
            unswitched = energy(r);
            unswitchedDerivative = fivePointDerivative(energy, (Real) r, 1e-5L);
            expected = switchedEnergy(r);
            dEdR = fivePointDerivative(switchedEnergy, (Real) r, 1e-5L);
        }
        // Near the cutoff S(r) is small and its rounding error in single precision is ~1e-7, so the errors are
        // relative to the unswitched energy and force.
        double energyScale = fabsl(unswitched);
        double forceScale = fabsl(unswitchedDerivative)+energyScale/0.2;
        ASSERT(fabsl(expected-state.getPotentialEnergy()) <= energyTol*energyScale);
        ASSERT(fabsl(dEdR+state.getForces()[1][1]) <= forceTol*forceScale+forceResolution);
        ASSERT(fabs(state.getForces()[1][0])+fabs(state.getForces()[1][2]) <= forceTol*forceScale+2*forceResolution);
        ASSERT_EQUAL_VEC(-state.getForces()[1], state.getForces()[0], 1e-6);
    }
}

/**
 * Test the alchemical scaling: pairs with exactly one alchemical particle are multiplied by the alchemical parameter,
 * whose default value is 1.
 */
template <class Model>
void testAlchemical() {
    System system;
    typename Model::Force* force = createSystem<Model>(system, {0, 1, 2, 3, 1}, {true, true, false, false, false});
    force->setAlchemicalParameterName("lambda_test");
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.2, 0, 0), Vec3(0, 0.25, 0), Vec3(0.15, 0.2, 0.15), Vec3(-0.1, 0.1, 0.25)};
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    ASSERT_EQUAL(1.0, context.getParameter("lambda_test"));
    compareWithReference<Model>(context, *force, 1, energyTol, forceTol);
    double fullEnergy = getEnergyAndForces(context).getPotentialEnergy();

    // Compute the alchemical (one particle alchemical) and the unscaled (none or both alchemical) parts of the energy.

    Real alchemicalEnergy = 0, unscaledEnergy = 0;
    for (int i = 0; i < 5; i++)
        for (int j = i+1; j < 5; j++) {
            Real r = sqrt((positions[i]-positions[j]).dot(positions[i]-positions[j]));
            Real e;
            ASSERT(Model::computePairEnergy(*force, i, j, r, e));
            if (force->getParticleIsAlchemical(i) != force->getParticleIsAlchemical(j))
                alchemicalEnergy += e;
            else
                unscaledEnergy += e;
        }
    ASSERT_EQUAL_REL(alchemicalEnergy+unscaledEnergy, fullEnergy, energyTol);
    for (double lambda : {0.0, 0.3, 0.8}) {
        context.setParameter("lambda_test", lambda);
        State state = getEnergyAndForces(context);
        ASSERT_EQUAL_TOL(unscaledEnergy+lambda*alchemicalEnergy, state.getPotentialEnergy(), energyTol);
        compareWithReference<Model>(context, *force, lambda, energyTol, forceTol);
    }
}

/**
 * Test the soft-core distance: a pair with exactly one alchemical particle is evaluated at
 * r_eff = (r^6 + (1 - lambda) R^6)^(1/6) and scaled by lambda. Pairs without or with two alchemical particles are
 * unaffected.
 */
template <class Model>
void testSoftcore() {
    System system;
    typename Model::Force* force = createSystem<Model>(system, {0, 1, 2, 3}, {true, true, false, false});
    force->setSoftcoreRadius(0.2);
    force->setNonbondedMethod(InteraxPairForce::CutoffNonPeriodic);
    force->setCutoffDistance(0.6);
    force->setUseSwitchingFunction(true);
    force->setSwitchingDistance(0.4);
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.18, 0, 0), Vec3(0.02, 0.1, 0), Vec3(0.3, 0.25, 0.15)};
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    for (double lambda : {1.0, 0.7, 0.2, 0.0}) {
        context.setParameter(force->getAlchemicalParameterName(), lambda);
        compareWithReference<Model>(context, *force, lambda, energyTol, forceTol);

        // Check one alchemical pair explicitly: 0-2 at r = sqrt(0.02^2 + 0.1^2).

        Real r = sqrtl(0.02L*0.02L + 0.1L*0.1L);
        Real rEff = powl(powl(r, 6) + (1-lambda)*powl(0.2L, 6), 1/6.0L);
        Real e;
        ASSERT(Model::computePairEnergy(*force, 0, 2, rEff, e));
        ASSERT_EQUAL_REL(lambda*e, computeReferencePairEnergy<Model>(*force, 0, 2, r, lambda), 1e-15);
    }

    // At lambda = 1 the soft-core radius has no effect.

    context.setParameter(force->getAlchemicalParameterName(), 1.0);
    double energy = getEnergyAndForces(context).getPotentialEnergy();
    force->setSoftcoreRadius(0.0);
    context.reinitialize(true);
    ASSERT_EQUAL_REL(getEnergyAndForces(context).getPotentialEnergy(), energy, energyTol);
}

/**
 * Test a larger periodic system with every option of the force (exclusions, switching, alchemical particles and
 * the soft-core distance) against the reference values, and check its forces against its energy.
 */
template <class Model>
void testLargeSystem() {
    const int numMolecules = 60;
    const double boxSize = 2.3;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    vector<int> types;
    vector<bool> isAlchemical;
    for (int i = 0; i < numMolecules; i++)
        for (int j = 0; j < 3; j++) {
            types.push_back((i+j)%Model::NUM_TYPES);
            isAlchemical.push_back(i < 4);
        }
    typename Model::Force* force = createSystem<Model>(system, types, isAlchemical);
    force->setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
    force->setCutoffDistance(0.9);
    force->setUseSwitchingFunction(true);
    force->setSwitchingDistance(0.75);
    force->setSoftcoreRadius(0.15);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions;
    for (int i = 0; i < numMolecules; i++) {
        // Molecules of three particles, 0.1 nm apart, more than 0.2 nm from every other molecule.
        vector<Vec3> molecule(3);
        bool accepted = false;
        while (!accepted) {
            molecule[0] = Vec3(random01(sfmt), random01(sfmt), random01(sfmt))*boxSize;
            molecule[1] = molecule[0]+Vec3(0.1, 0, 0);
            molecule[2] = molecule[0]+Vec3(0, 0.1, 0);
            accepted = true;
            Vec3 box[3] = {Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize)};
            for (const Vec3& p : positions)
                for (const Vec3& q : molecule) {
                    Vec3 delta = getMinimumImage(p-q, box);
                    if (delta.dot(delta) < 0.2*0.2)
                        accepted = false;
                }
        }
        for (int j = 0; j < 3; j++) {
            positions.push_back(molecule[j]);
            for (int k = 0; k < j; k++)
                force->addExclusion(3*i+k, 3*i+j);
        }
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    for (double lambda : {1.0, 0.4}) {
        context.setParameter(force->getAlchemicalParameterName(), lambda);
        compareWithReference<Model>(context, *force, lambda, energyTol, forceTol);
        checkForceEnergyConsistency(context, consistencyTol);
    }
}

/**
 * Test updateParametersInContext().
 */
template <class Model>
void testChangingParameters() {
    System system;
    typename Model::Force* force = createSystem<Model>(system, {0, 1, 2, 3}, {false, false, true, false});
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.2, 0, 0), Vec3(0, 0.22, 0), Vec3(0.15, 0.2, 0.15)};
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setParameter(force->getAlchemicalParameterName(), 0.5);
    double energy1 = getEnergyAndForces(context).getPotentialEnergy();

    // Change the parameters and make sure the results change and agree with the reference.

    Model::changeParameters(*force);
    force->updateParametersInContext(context);
    State state2 = getEnergyAndForces(context);
    ASSERT(fabs(energy1-state2.getPotentialEnergy()) > 1e-3*fabs(energy1));
    compareWithReference<Model>(context, *force, 0.5, energyTol, forceTol);

    // A new context with the altered parameters agrees.

    VerletIntegrator integrator2(0.001);
    Context context2(system, integrator2, platform);
    context2.setPositions(positions);
    context2.setParameter(force->getAlchemicalParameterName(), 0.5);
    State state3 = getEnergyAndForces(context2);
    ASSERT_EQUAL_REL(state3.getPotentialEnergy(), state2.getPotentialEnergy(), energyTol);
    assertForcesEqual(state3.getForces(), state2.getForces(), forceTol);
}

/**
 * Test that invalid settings are rejected.
 */
template <class Model>
void testIllegalSettings() {
    // The force must have as many particles as the System.
    {
        System system;
        createSystem<Model>(system, {0, 1});
        system.addParticle(1.0);
        assertContextCreationFails(system, platform, "a missing particle");
    }
    // Exclusions must refer to valid particles.
    {
        System system;
        createSystem<Model>(system, {0, 1})->addExclusion(0, 2);
        assertContextCreationFails(system, platform, "an exclusion of an invalid particle");
    }
    // The switching distance must be less than the cutoff.
    {
        System system;
        typename Model::Force* force = createSystem<Model>(system, {0, 1});
        force->setNonbondedMethod(InteraxPairForce::CutoffNonPeriodic);
        force->setCutoffDistance(0.5);
        force->setUseSwitchingFunction(true);
        force->setSwitchingDistance(0.5);
        assertContextCreationFails(system, platform, "a switching distance equal to the cutoff");
    }
    // The cutoff cannot exceed half the box.
    {
        System system;
        typename Model::Force* force = createSystem<Model>(system, {0, 1});
        force->setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
        force->setCutoffDistance(1.1);
        system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
        assertContextCreationFails(system, platform, "a cutoff larger than half the box");
    }
    // The soft-core radius cannot be negative.
    {
        unique_ptr<typename Model::Force> force(Model::createForce());
        bool failed = false;
        try {
            force->setSoftcoreRadius(-0.1);
        }
        catch (const OpenMMException& e) {
            failed = true;
        }
        ASSERT(failed);
    }
}

/**
 * Test the overlap methods of the forces built on the Slater overlap integral (InteraxExchangeForce and
 * InteraxShortRangeForce) against the exact integral, for equal, nearly equal and different widths. The model must
 * also have
 *
 *   void addParticleWithWidth(Force&, double width, bool isAlchemical)
 *   void setOverlapMethod(Force&, int method)
 *
 * Exact is the exact integral to ~1e-10 in double precision and ~1e-6 in single precision (docs/SPEC.md: the
 * energies are compared to 3e-5, since the short-range correction amplifies the error); in single precision the
 * closed form loses more below about one width (up to 2e-4 at r = 0.1 width just above the switch to
 * it at t = 0.2), which is not tested. The InteraxFF schemes carry the truncation errors of the export: they are
 * compared where they are accurate (order 6 where t x < 1, to 1e-6; order 2 where it uses its expansion and
 * t x < 0.12, to 1e-5) and exact elsewhere.
 */
template <class Model>
void testOverlapMethods() {
    bool doublePrecision = (getPlatformPrecision(platform, false) == "double");
    const double width = 0.02;
    const double relativeDifferences[] = {0.0, 2e-4, 1e-3, 0.005, 0.01, 0.05, 0.12, 0.3, 0.5};
    const double distances[] = {0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3};
    const int methods[] = {0, 2, 6};
    for (int method : methods)
        for (double difference : relativeDifferences) {
            // Particle 0 is the narrower one in half of the cases, so that the scheme expands about either width.
            for (int order = 0; order < 2; order++) {
                System system;
                addParticles(system, 2);
                typename Model::Force* force = Model::createForce();
                double widths[2] = {width, width*(1+difference)};
                Model::addParticleWithWidth(*force, widths[order], false);
                Model::addParticleWithWidth(*force, widths[1-order], false);
                Model::setOverlapMethod(*force, method);
                system.addForce(force);
                VerletIntegrator integrator(0.001);
                Context context(system, integrator, platform);
                for (double r : distances) {
                    // The InteraxFF schemes expand about the width of particle 0.
                    Real t = fabs(widths[1-order]-widths[order])/widths[order], x = r/widths[order];
                    double tolerance;
                    if (method == 0)
                        tolerance = (doublePrecision ? 1e-9 : (r > 0.025 ? 3e-5 : -1));
                    else if (method == 6 && t < 0.2)
                        tolerance = (t*x < 1 ? 1e-6 : -1);
                    else if (method == 2 && t*widths[order] < 0.01*0.0529177210903)
                        tolerance = (t*x < 0.12 ? 1e-5 : -1);
                    else
                        tolerance = 1e-9;
                    if (!doublePrecision && method != 0)
                        tolerance = (method == 2 || r <= 0.025 ? -1 : max(tolerance, 3e-5));
                    if (tolerance < 0)
                        continue;
                    context.setPositions({Vec3(0, 0, 0), Vec3(0, 0, r)});
                    Real expected;
                    ASSERT(Model::computePairEnergy(*force, 0, 1, r, expected));
                    State state = getEnergyAndForces(context);
                    Real dEdR = fivePointDerivative([&] (Real s) {Real e; Model::computePairEnergy(*force, 0, 1, s, e); return e;}, (Real) r, 1e-6L);
                    try {
                        ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), tolerance);
                        ASSERT(fabsl(dEdR+state.getForces()[1][2]) <= 10*tolerance*fabsl(dEdR)+forceResolution);
                    }
                    catch (...) {
                        cout << "Overlap method " << method << ", widths " << widths[0] << " and " << widths[1] << ", r = " << r << endl;
                        throw;
                    }
                }
            }
        }
}

/**
 * Test, on a platform other than Reference, that each overlap method gives the same results as on the Reference
 * platform, in a periodic system of identical molecules with nearly equal widths. The platform may reorder the
 * molecules, and the InteraxFF schemes must still expand about the width of the particle with the lower original
 * index.
 */
template <class Model>
void testOverlapMethodsAgainstReference() {
    if (platform.getName() == "Reference")
        return;
    bool doublePrecision = (getPlatformPrecision(platform, false) == "double");
    const int numMolecules = 150;
    const double boxSize = 2.0;
    const double widths[] = {0.02, 0.02*(1+0.004), 0.02*(1+0.1), 0.026};
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(1, sfmt);
    vector<Vec3> positions;
    Vec3 box[3] = {Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize)};
    while (positions.size() < 4*numMolecules) {
        // Molecules of four particles, more than 0.12 nm from the other molecules.
        Vec3 origin = Vec3(random01(sfmt), random01(sfmt), random01(sfmt))*boxSize;
        vector<Vec3> molecule = {origin, origin+Vec3(0.1, 0, 0), origin+Vec3(0, 0.1, 0), origin+Vec3(0, 0, 0.1)};
        bool accepted = true;
        for (const Vec3& p : positions)
            for (const Vec3& q : molecule) {
                Vec3 delta = getMinimumImage(p-q, box);
                accepted &= (delta.dot(delta) > 0.12*0.12);
            }
        if (accepted)
            positions.insert(positions.end(), molecule.begin(), molecule.end());
    }
    for (int method : {0, 2, 6}) {
        if (!doublePrecision && method == 2)
            continue;
        System system;
        system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
        addParticles(system, 4*numMolecules);
        typename Model::Force* force = Model::createForce();
        for (int i = 0; i < numMolecules; i++) {
            for (int j = 0; j < 4; j++)
                Model::addParticleWithWidth(*force, widths[j], false);
            for (int j = 0; j < 4; j++)
                for (int k = j+1; k < 4; k++)
                    force->addExclusion(4*i+j, 4*i+k);
        }
        force->setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
        force->setCutoffDistance(0.6);
        Model::setOverlapMethod(*force, method);
        system.addForce(force);
        VerletIntegrator integrator(0.001), referenceIntegrator(0.001);
        Context context(system, integrator, platform);
        Context referenceContext(system, referenceIntegrator, Platform::getPlatformByName("Reference"));
        context.setPositions(positions);
        referenceContext.setPositions(positions);
        State state = getEnergyAndForces(context);
        State referenceState = getEnergyAndForces(referenceContext);
        // The errors are relative to the sum of the magnitudes of the pair energies.
        vector<Vec3> forces;
        Real absEnergy;
        computeReferenceEnergy<Model>(*force, positions, box, 1, forces, &absEnergy);
        double tolerance = (doublePrecision ? 1e-9 : 2e-5);
        ASSERT(fabs(referenceState.getPotentialEnergy()-state.getPotentialEnergy()) <= tolerance*absEnergy);
        assertForcesEqual(referenceState.getForces(), state.getForces(), 10*tolerance);
    }
}

/**
 * Run all the tests of InteraxPairForce for a model.
 */
template <class Model>
void runPairForceTests() {
    testPairEnergies<Model>();
    testExclusions<Model>();
    testCutoff<Model>();
    testPeriodic<Model>();
    testTriclinic<Model>();
    testSwitchingFunction<Model>(InteraxPairForce::CutoffNonPeriodic);
    testSwitchingFunction<Model>(InteraxPairForce::CutoffPeriodic);
    testAlchemical<Model>();
    testSoftcore<Model>();
    testLargeSystem<Model>();
    testChangingParameters<Model>();
    testIllegalSettings<Model>();
}

/**
 * Set the tolerances of the pair force tests for the precision of the platform.
 */
void initializePairForceTolerances() {
    setForceResolution(platform);
    if (getPlatformPrecision(platform, false) == "double") {
        energyTol = 1e-10;
        forceTol = 1e-9;
        consistencyTol = 1e-5;
    }
    else {
        energyTol = 2e-5;
        forceTol = 2e-4;
        consistencyTol = 2e-3;
    }
}
