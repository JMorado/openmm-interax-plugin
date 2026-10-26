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
 * This tests InteraxDispersionForce: the damped C6/C8/C10 pair dispersion
 *
 *   E = - sum_{pairs} sum_{n = 6, 8, 10} f_n(x) C_n / r^n,   x = beta 2 r / (s_i + s_j)
 *
 * with the coefficients of docs/SPEC.md, its long range correction, and everything it shares with the other
 * pair forces (TestInteraxPairForce.h).
 */

#include "TestInteraxPairForce.h"

/**
 * The model of InteraxDispersionForce used by the pair force tests. The particle types are InteraxFF atoms (an
 * oxygen, two carbons and a hydrogen of the test systems in tests/data).
 */
struct DispersionModel {
    typedef InteraxDispersionForce Force;
    static const int NUM_TYPES = 4;
    static Force* createForce() {
        Force* force = new Force();
        force->setDampingScale(0.7155241370201111);
        return force;
    }
    static void addParticle(Force& force, int type, bool isAlchemical) {
        static const double parameters[NUM_TYPES][5] = {
            {2.0412491330659432, 0.07991696226413432, 0.003917538949072952, 0.0020393931939312773, 0.026844933830306183},
            {2.179116829570064, 0.05077216863583387, 0.001421435052885423, 0.0012613225562556841, 0.021090413137365274},
            {0.47385034770767653, 0.0062664158577877644, 0.00015411387998271836, 0.00023498508736915282, 0.020351567997950105},
            {3.9405044694974447, 0.18787610923101286, 0.010735264902054127, 0.004221894603968141, 0.03096091246647922}
        };
        const double* p = parameters[type];
        force.addParticle(p[0], p[1], p[2], p[3], p[4], isAlchemical);
    }
    static bool computePairEnergy(const Force& force, int i, int j, Real r, Real& energy) {
        double m1i, m2i, m3i, alphai, widthi, m1j, m2j, m3j, alphaj, widthj;
        bool alchemical;
        force.getParticleParameters(i, m1i, m2i, m3i, alphai, widthi, alchemical);
        force.getParticleParameters(j, m1j, m2j, m3j, alphaj, widthj, alchemical);
        if (alphai*alphaj == 0)
            return false;
        Real denominator = (Real) alphai*m1j + (Real) alphaj*m1i;
        Real alpha2 = (Real) alphai*alphaj;
        Real c6 = alpha2*m1i*m1j/denominator;
        Real c8 = 1.5L*alpha2*((Real) m1i*m2j + (Real) m2i*m1j)/denominator;
        Real c10 = (2*alpha2*((Real) m1i*m3j + (Real) m3i*m1j) + 4.2L*alpha2*m2i*m2j)/denominator;
        Real x = force.getDampingScale()*2*r/((Real) widthi+widthj);
        energy = -(tangToennies(6, x)*c6/powl(r, 6) + tangToennies(8, x)*c8/powl(r, 8) + tangToennies(10, x)*c10/powl(r, 10));
        return true;
    }
    static void changeParameters(Force& force) {
        force.setParticleParameters(1, 2.5, 0.06, 0.002, 0.0015, 0.023, false);
        force.setDampingScale(0.8);
    }
};

/**
 * Test the coefficients: at large distances, where the damping functions are 1, the energy is
 * -(C6/r^6 + C8/r^8 + C10/r^10) with the coefficients computed by hand for two particles.
 */
void testCoefficients() {
    System system;
    addParticles(system, 2);
    InteraxDispersionForce* force = new InteraxDispersionForce();
    force->addParticle(2.0, 0.1, 0.005, 0.002, 0.025);
    force->addParticle(0.5, 0.01, 0.0002, 0.0003, 0.02);
    force->setDampingScale(1.0);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    double r = 2.0;   // x = 2r/(0.045 nm) = 89, so f_n = 1 to double precision
    context.setPositions({Vec3(0, 0, 0), Vec3(0, 0, r)});
    double denominator = 0.002*0.5 + 0.0003*2.0;
    double c6 = 0.002*0.0003*2.0*0.5/denominator;
    double c8 = 1.5*0.002*0.0003*(2.0*0.01 + 0.1*0.5)/denominator;
    double c10 = (2*0.002*0.0003*(2.0*0.0002 + 0.005*0.5) + 4.2*0.002*0.0003*0.1*0.01)/denominator;
    double expected = -(c6/pow(r, 6) + c8/pow(r, 8) + c10/pow(r, 10));
    State state = getEnergyAndForces(context);
    ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
    double dEdR = 6*c6/pow(r, 7) + 8*c8/pow(r, 9) + 10*c10/pow(r, 11);
    ASSERT_EQUAL_VEC(Vec3(0, 0, -dEdR), state.getForces()[1], forceTol);
    ASSERT_EQUAL_VEC(Vec3(0, 0, dEdR), state.getForces()[0], forceTol);
}

/**
 * Test the Tang-Toennies damping on both sides of x = (n+1)/2, where the force switches between the closed form and
 * the series of f_n, and at short range, where the damped energy stays finite.
 */
void testDamping() {
    System system;
    addParticles(system, 2);
    InteraxDispersionForce* force = DispersionModel::createForce();
    DispersionModel::addParticle(*force, 0, false);
    DispersionModel::addParticle(*force, 3, false);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    double widthSum = 0.026844933830306183+0.03096091246647922;
    for (double x : {0.01, 0.5, 2.0, 3.49, 3.51, 4.49, 4.51, 5.49, 5.51, 8.0, 20.0}) {
        double r = x*widthSum/(2*force->getDampingScale());
        context.setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
        State state = getEnergyAndForces(context);
        Real expected;
        DispersionModel::computePairEnergy(*force, 0, 1, r, expected);
        ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
        Real dEdR = fivePointDerivative([&] (Real s) {Real e; DispersionModel::computePairEnergy(*force, 0, 1, s, e); return e;}, (Real) r, 1e-3L*r);
        assertForcesEqual({Vec3(dEdR, 0, 0), Vec3(-dEdR, 0, 0)}, state.getForces(), forceTol);
    }
}

/**
 * Test that a particle with zero polarizability (a lone pair site) does not interact.
 */
void testZeroPolarizability() {
    System system;
    addParticles(system, 3);
    InteraxDispersionForce* force = DispersionModel::createForce();
    DispersionModel::addParticle(*force, 0, false);
    force->addParticle(0.0, 0.0, 0.0, 0.0, 0.0);
    DispersionModel::addParticle(*force, 1, false);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(0.05, 0, 0), Vec3(0, 0.3, 0)});
    State state = getEnergyAndForces(context);
    Real expected;
    DispersionModel::computePairEnergy(*force, 0, 2, 0.3, expected);
    ASSERT_EQUAL_REL(expected, state.getPotentialEnergy(), energyTol);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[1], 0);
}

/**
 * Integrate a function over (a, b) with Simpson's rule.
 */
template <class Function>
Real integrate(const Function& function, Real a, Real b, int numIntervals=4000) {
    Real h = (b-a)/numIntervals;
    Real sum = function(a)+function(b);
    for (int i = 1; i < numIntervals; i++)
        sum += (i%2 == 1 ? 4 : 2)*function(a+i*h);
    return sum*h/3;
}

/**
 * Compute the long range correction 2 pi N^2 <I> / V, with <I> the average over the N(N+1)/2 pairs of particles
 * (self pairs included) of the integral of r^2 E(r) beyond the cutoff and of r^2 (1 - S(r)) E(r) over the switching
 * interval. Pairs that involve an alchemical particle are scaled by lambda.
 */
Real computeLongRangeCorrection(const InteraxDispersionForce& force, double volume, Real lambda) {
    int numParticles = force.getNumParticles();
    Real cutoff = force.getCutoffDistance();
    Real sum = 0;
    for (int i = 0; i < numParticles; i++)
        for (int j = i; j < numParticles; j++) {
            auto energy = [&] (Real r) {
                Real e;
                return (DispersionModel::computePairEnergy(force, i, j, r, e) ? e : 0);
            };
            // Integrate beyond the cutoff with r = r_c/u.
            Real integral = integrate([&] (Real u) {return (u == 0 ? 0 : energy(cutoff/u)*cutoff*cutoff*cutoff/(u*u*u*u));}, 0, 1);
            if (force.getUseSwitchingFunction()) {
                Real switchingDistance = force.getSwitchingDistance();
                integral += integrate([&] (Real r) {return (1-switchingFunction(r, switchingDistance, cutoff))*energy(r)*r*r;}, switchingDistance, cutoff);
            }
            bool alchemical = (force.getParticleIsAlchemical(i) || force.getParticleIsAlchemical(j));
            sum += (alchemical ? lambda : 1)*integral;
        }
    return 2*PI_L*numParticles*numParticles*sum/(0.5L*numParticles*(numParticles+1))/volume;
}

/**
 * Test the long range correction, with and without switching, its dependence on the box volume, its alchemical
 * scaling, and its update by updateParametersInContext(). The particles are further apart than the cutoff, so the
 * energy is the correction alone.
 */
void testLongRangeCorrection() {
    const int numParticles = 8;
    System system;
    addParticles(system, numParticles);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    InteraxDispersionForce* force = DispersionModel::createForce();
    for (int i = 0; i < numParticles; i++)
        DispersionModel::addParticle(*force, i%DispersionModel::NUM_TYPES, i < 2);
    force->setNonbondedMethod(InteraxPairForce::CutoffPeriodic);
    force->setCutoffDistance(0.6);
    force->setUseLongRangeCorrection(true);
    system.addForce(force);
    vector<Vec3> positions;
    for (int i = 0; i < numParticles; i++)
        positions.push_back(Vec3(i%2, (i/2)%2, i/4)+Vec3(0.1, 0.2, 0.3));
    for (bool useSwitch : {false, true}) {
        force->setUseSwitchingFunction(useSwitch);
        force->setSwitchingDistance(0.45);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions(positions);
        ASSERT_EQUAL_REL(computeLongRangeCorrection(*force, 8.0, 1), getEnergyAndForces(context).getPotentialEnergy(), 1e-9);

        // The correction is proportional to 1/V.

        context.setPeriodicBoxVectors(Vec3(2.2, 0, 0), Vec3(0, 2.1, 0), Vec3(0, 0, 2.3));
        ASSERT_EQUAL_REL(computeLongRangeCorrection(*force, 2.2*2.1*2.3, 1), getEnergyAndForces(context).getPotentialEnergy(), 1e-9);
        context.setPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));

        // Every pair that involves an alchemical particle, the self pairs and the pair of the two alchemical particles
        // included, is scaled by lambda.

        context.setParameter("lambda_sterics", 0.3);
        ASSERT_EQUAL_REL(computeLongRangeCorrection(*force, 8.0, 0.3), getEnergyAndForces(context).getPotentialEnergy(), 1e-9);
        context.setParameter("lambda_sterics", 1.0);

        // updateParametersInContext() updates the correction.

        double m1, m2, m3, alpha, width, dampingScale = force->getDampingScale();
        bool isAlchemical;
        force->getParticleParameters(3, m1, m2, m3, alpha, width, isAlchemical);
        force->setParticleParameters(3, 2.5, 0.06, 0.002, 0.0015, 0.023, false);
        force->setDampingScale(0.8);
        force->updateParametersInContext(context);
        double expected = computeLongRangeCorrection(*force, 8.0, 1);
        ASSERT_EQUAL_REL(expected, getEnergyAndForces(context).getPotentialEnergy(), 1e-9);
        force->setParticleParameters(3, m1, m2, m3, alpha, width, false);
        force->setDampingScale(dampingScale);
        ASSERT(fabs(expected-computeLongRangeCorrection(*force, 8.0, 1)) > 1e-3*fabs(expected));
    }

    // Without periodic boundary conditions, or when it is disabled, there is no correction.

    force->setUseLongRangeCorrection(false);
    {
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions(positions);
        ASSERT_EQUAL(0.0, getEnergyAndForces(context).getPotentialEnergy());
    }
    force->setUseLongRangeCorrection(true);
    force->setNonbondedMethod(InteraxPairForce::CutoffNonPeriodic);
    {
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions(positions);
        ASSERT_EQUAL(0.0, getEnergyAndForces(context).getPotentialEnergy());
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        initializePairForceTolerances();
        runPairForceTests<DispersionModel>();
        testCoefficients();
        testDamping();
        testZeroPolarizability();
        testLongRangeCorrection();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
