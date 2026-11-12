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
 * This tests InteraxShortRangeForce: the short-range correction
 *
 *   E = sum_{pairs} [ (K_i + K_j)/2 St + (L_i + L_j)/2 St^2 ],   St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j)
 *
 * with S the overlap integral of normalized Slater densities (docs/SPEC.md), its overlap methods, everything it
 * shares with the other pair forces (TestInteraxPairForce.h), and the three pair forces together in one System.
 */

#include "TestInteraxPairForce.h"

/**
 * The model of InteraxShortRangeForce used by the pair force tests. The particle types are InteraxFF atoms of the
 * test systems in tests/data, with K of both signs.
 */
struct ShortRangeModel {
    typedef InteraxShortRangeForce Force;
    static const int NUM_TYPES = 4;
    static Force* createForce() {
        return new Force();
    }
    static void addParticle(Force& force, int type, bool isAlchemical) {
        static const double parameters[NUM_TYPES][3] = {
            {-1102.281886852146, 5953.779066676866, 0.017966632940037853},
            {1236.5568512540217, 80470.88690318535, 0.026044362132579005},
            {-1579.591012686484, 70439.46596886561, 0.027056173026181088},
            {-1702.373363056105, 47202.476121472, 0.01880253891316055}
        };
        force.addParticle(parameters[type][0], parameters[type][1], parameters[type][2], isAlchemical);
    }
    static void addParticleWithWidth(Force& force, double width, bool isAlchemical) {
        force.addParticle(-1500.0, 60000.0, width, isAlchemical);
    }
    static void setOverlapMethod(Force& force, int method) {
        force.setOverlapMethod((Force::OverlapMethod) method);
    }
    static bool computePairEnergy(const Force& force, int i, int j, Real r, Real& energy) {
        double ki, li, widthi, kj, lj, widthj;
        bool alchemical;
        force.getParticleParameters(i, ki, li, widthi, alchemical);
        force.getParticleParameters(j, kj, lj, widthj, alchemical);
        if (widthi*widthj == 0)
            return false;
        Real normalizedOverlap = 8*PI_L*powl((Real) widthi+widthj, 3)*slaterOverlap(r, widthi, widthj);
        energy = 0.5L*((Real) ki+kj)*normalizedOverlap + 0.5L*((Real) li+lj)*normalizedOverlap*normalizedOverlap;
        return true;
    }
    static void changeParameters(Force& force) {
        force.setParticleParameters(1, 900.0, 50000.0, 0.024, false);
        force.setParticleParameters(3, -1000.0, 30000.0, 0.02, false);
    }
};

/**
 * Test the normalization of the overlap: for equal widths at r -> 0, St = 1 and the energy is (K_i + K_j)/2 +
 * (L_i + L_j)/2.
 */
void testNormalization() {
    System system;
    addParticles(system, 2);
    InteraxShortRangeForce* force = new InteraxShortRangeForce();
    force->addParticle(-1000.0, 40000.0, 0.02);
    force->addParticle(-2000.0, 60000.0, 0.02);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(1e-6, 0, 0)});
    double x = 1e-6/0.02;
    double st = exp(-x)*(1+x+x*x/3);
    ASSERT_EQUAL_REL(-1500.0*st + 50000.0*st*st, getEnergyAndForces(context).getPotentialEnergy(), max(energyTol, 1e-6));
}

/**
 * Test that a particle of zero width (a lone pair site) does not interact.
 */
void testZeroWidth() {
    System system;
    addParticles(system, 3);
    InteraxShortRangeForce* force = new InteraxShortRangeForce();
    force->addParticle(-1000.0, 40000.0, 0.02);
    force->addParticle(-2000.0, 60000.0, 0.0);
    force->addParticle(500.0, 20000.0, 0.025);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(0.03, 0, 0), Vec3(0, 0.12, 0)});
    State state = getEnergyAndForces(context);
    Real expected;
    ShortRangeModel::computePairEnergy(*force, 0, 2, 0.12, expected);
    ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[1], 0);
}

/**
 * Test the three pair forces in one System, in the same and in different force groups, against each force alone.
 * On the GPU platforms forces with identical exclusions and cutoffs share one nonbonded kernel.
 */
void testCombinedPairForces() {
    const int numMolecules = 40;
    const double boxSize = 1.8;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(2, sfmt);
    vector<Vec3> positions;
    Vec3 box[3] = {Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize)};
    while (positions.size() < 2*numMolecules) {
        // Molecules of two particles 0.1 nm apart, more than 0.2 nm from the other molecules.
        Vec3 origin = Vec3(random01(sfmt), random01(sfmt), random01(sfmt))*boxSize;
        bool accepted = true;
        for (const Vec3& p : positions)
            for (const Vec3& q : {origin, origin+Vec3(0.1, 0, 0)}) {
                Vec3 delta = getMinimumImage(p-q, box);
                accepted &= (delta.dot(delta) > 0.2*0.2);
            }
        if (accepted) {
            positions.push_back(origin);
            positions.push_back(origin+Vec3(0.1, 0, 0));
        }
    }
    auto createForces = [&] (System& system, bool sameGroup) {
        system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
        addParticles(system, 2*numMolecules);
        InteraxDispersionForce* dispersion = new InteraxDispersionForce();
        InteraxExchangeForce* exchange = new InteraxExchangeForce();
        InteraxShortRangeForce* shortRange = new InteraxShortRangeForce();
        for (int i = 0; i < numMolecules; i++) {
            bool isAlchemical = (i == 0);
            dispersion->addParticle(2.0412491330659432, 0.07991696226413432, 0.003917538949072952, 0.0020393931939312773, 0.026844933830306183, isAlchemical);
            dispersion->addParticle(0.47385034770767653, 0.0062664158577877644, 0.00015411387998271836, 0.00023498508736915282, 0.020351567997950105, isAlchemical);
            exchange->addParticle(10.326424719450545, 0.0248302193400194, isAlchemical);
            exchange->addParticle(1.0254403205662432, 0.019763296864870033, isAlchemical);
            ShortRangeModel::addParticle(*shortRange, 0, isAlchemical);
            ShortRangeModel::addParticle(*shortRange, 3, isAlchemical);
        }
        vector<InteraxPairForce*> forces = {dispersion, exchange, shortRange};
        for (int k = 0; k < 3; k++) {
            InteraxPairForce* force = forces[k];
            for (int i = 0; i < numMolecules; i++)
                force->addExclusion(2*i, 2*i+1);
            force->setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
            force->setCutoffDistance(0.8);
            force->setUseSwitchingFunction(true);
            force->setSwitchingDistance(0.7);
            force->setSoftcoreRadius(0.1);
            force->setForceGroup(sameGroup ? 0 : k);
            system.addForce(force);
        }
        dispersion->setUseLongRangeCorrection(true);
    };
    System separate, combined;
    createForces(separate, false);
    createForces(combined, true);
    VerletIntegrator integrator1(0.001), integrator2(0.001);
    Context context1(separate, integrator1, platform);
    Context context2(combined, integrator2, platform);
    context1.setPositions(positions);
    context2.setPositions(positions);
    context1.setParameter("lambda_sterics", 0.6);
    context2.setParameter("lambda_sterics", 0.6);
    State total1 = getEnergyAndForces(context1);
    State total2 = getEnergyAndForces(context2);
    ASSERT_EQUAL_TOL(total1.getPotentialEnergy(), total2.getPotentialEnergy(), energyTol);
    assertForcesEqual(total1.getForces(), total2.getForces(), forceTol);

    // Each force group of the first context is one force: compare it with a System that has only that force.

    for (int k = 0; k < 3; k++) {
        System single;
        createForces(single, false);
        for (int j = 2; j >= 0; j--)
            if (j != k)
                single.removeForce(j);
        VerletIntegrator integrator(0.001);
        Context context(single, integrator, platform);
        context.setPositions(positions);
        context.setParameter("lambda_sterics", 0.6);
        State expected = getEnergyAndForces(context);
        State found = getEnergyAndForces(context1, 1<<k);
        ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), found.getPotentialEnergy(), energyTol);
        assertForcesEqual(expected.getForces(), found.getForces(), forceTol);
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        initializePairForceTolerances();
        runPairForceTests<ShortRangeModel>();
        testNormalization();
        testZeroWidth();
        testOverlapMethods<ShortRangeModel>();
        testOverlapMethodsAgainstReference<ShortRangeModel>();
        testCombinedPairForces();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
