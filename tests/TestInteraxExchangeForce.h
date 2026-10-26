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
 * This tests InteraxExchangeForce: the Slater-overlap exchange repulsion
 *
 *   E = sum_{pairs} A_i A_j S(r; s_i, s_j)
 *
 * with S the overlap integral of normalized Slater densities (docs/SPEC.md), its overlap methods, and everything it
 * shares with the other pair forces (TestInteraxPairForce.h).
 */

#include "TestInteraxPairForce.h"

/**
 * The model of InteraxExchangeForce used by the pair force tests. The particle types are InteraxFF atoms of the test
 * systems in tests/data; their widths differ by 10% or more.
 */
struct ExchangeModel {
    typedef InteraxExchangeForce Force;
    static const int NUM_TYPES = 4;
    static Force* createForce() {
        return new Force();
    }
    static void addParticle(Force& force, int type, bool isAlchemical) {
        static const double parameters[NUM_TYPES][2] = {
            {10.326424719450545, 0.0248302193400194},
            {1.0254403205662432, 0.019763296864870033},
            {13.231601566169132, 0.02761867008817038},
            {1.5623921585722147, 0.021706271973167678}
        };
        force.addParticle(parameters[type][0], parameters[type][1], isAlchemical);
    }
    static void addParticleWithWidth(Force& force, double width, bool isAlchemical) {
        force.addParticle(3.0, width, isAlchemical);
    }
    static void setOverlapMethod(Force& force, int method) {
        force.setOverlapMethod((Force::OverlapMethod) method);
    }
    static bool computePairEnergy(const Force& force, int i, int j, Real r, Real& energy) {
        double amplitudei, widthi, amplitudej, widthj;
        bool alchemical;
        force.getParticleParameters(i, amplitudei, widthi, alchemical);
        force.getParticleParameters(j, amplitudej, widthj, alchemical);
        if (widthi*widthj == 0)
            return false;
        energy = (Real) amplitudei*amplitudej*slaterOverlap(r, widthi, widthj);
        return true;
    }
    static void changeParameters(Force& force) {
        force.setParticleParameters(1, 2.0, 0.023, false);
        force.setParticleParameters(3, 5.0, 0.026, false);
    }
};

/**
 * Check the closed form of the Slater overlap integral used by these tests (slaterOverlap()) against the integral
 * computed numerically from the Fourier transforms of the densities. This involves no platform.
 */
void testOverlapIntegral() {
    const double cases[][3] = {{0.02, 0.02, 0.02}, {0.05, 0.02, 0.02}, {0.1, 0.02, 0.025}, {0.15, 0.0198, 0.0276},
                               {0.05, 0.03, 0.03003}, {0.2, 0.025, 0.0248}, {0.01, 0.018, 0.031}};
    for (auto& c : cases) {
        Real expected = slaterOverlapByQuadrature(c[0], c[1], c[2]);
        ASSERT_EQUAL_REL(expected, slaterOverlap(c[0], c[1], c[2]), 1e-9);
        ASSERT_EQUAL_REL(expected, slaterOverlap(c[0], c[2], c[1]), 1e-9);
    }

    // For nearly equal widths, compare with the closed form evaluated in 50 digit arithmetic (mpmath), and check
    // the continuity with the limit of equal widths.

    ASSERT_EQUAL_REL(5371.8136182115909371938804306761/9, slaterOverlap(0.01L, 0.02L, 0.02L*(1+0.0002L)), 1e-15);
    ASSERT_EQUAL_REL(slaterOverlap(0.05L, 0.02L, 0.02L), slaterOverlap(0.05L, 0.02L, 0.02L*(1+1e-12L)), 1e-11);

    // At r = 0 the normalized overlap 8 pi (a + b)^3 S is 1 for equal widths.

    Real a = 0.02;
    ASSERT_EQUAL_REL(1.0, 8*PI_L*powl(2*a, 3)*slaterOverlap(1e-12L, a, a), 1e-9);
}

/**
 * Test the energy of a pair against the formula A_i A_j S written out for equal widths.
 */
void testEqualWidths() {
    System system;
    addParticles(system, 2);
    InteraxExchangeForce* force = new InteraxExchangeForce();
    force->addParticle(2.0, 0.025);
    force->addParticle(3.0, 0.025);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    for (double r : {0.02, 0.1, 0.3}) {
        context.setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
        double x = r/0.025;
        double expected = 6.0*exp(-x)*(1+x+x*x/3)/(64*M_PI*pow(0.025, 3));
        double dEdR = -6.0*exp(-x)*x*(1+x)/3/(64*M_PI*pow(0.025, 4));
        State state = getEnergyAndForces(context);
        ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
        ASSERT_EQUAL_VEC(Vec3(-dEdR, 0, 0), state.getForces()[1], forceTol);
    }
}

/**
 * Test that a particle of zero width (a lone pair site) does not interact.
 */
void testZeroWidth() {
    System system;
    addParticles(system, 3);
    InteraxExchangeForce* force = new InteraxExchangeForce();
    force->addParticle(2.0, 0.025);
    force->addParticle(3.0, 0.0);
    force->addParticle(1.5, 0.02);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(0.03, 0, 0), Vec3(0, 0.2, 0)});
    State state = getEnergyAndForces(context);
    ASSERT_EQUAL_REL((double) (3.0*slaterOverlap(0.2, 0.025, 0.02)), state.getPotentialEnergy(), energyTol);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[1], 0);
}

/**
 * Test that updateParametersInContext() refuses a change of the overlap method.
 */
void testChangingOverlapMethod() {
    System system;
    addParticles(system, 2);
    InteraxExchangeForce* force = new InteraxExchangeForce();
    force->addParticle(2.0, 0.025);
    force->addParticle(3.0, 0.02);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    force->setOverlapMethod(InteraxExchangeForce::InteraxOrder6);
    bool failed = false;
    try {
        force->updateParametersInContext(context);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    ASSERT(failed);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        initializePairForceTolerances();
        testOverlapIntegral();
        runPairForceTests<ExchangeModel>();
        testEqualWidths();
        testZeroWidth();
        testOverlapMethods<ExchangeModel>();
        testOverlapMethodsAgainstReference<ExchangeModel>();
        testChangingOverlapMethod();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
