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
 * This tests InteraxThreeBodyForce, the Axilrod-Teller-Muto three-body dispersion of InteraxFF:
 *
 *   E = scale * sum_{triples} C9 (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC r_BC)^3 * f_AB f_AC f_BC
 *   C9 = 3/2 a_A a_B a_C w_A w_B w_C (w_A + w_B + w_C) / ((w_A + w_B)(w_B + w_C)(w_C + w_A))
 *
 * over the triples of sites (centroids of particles) of three different molecules whose three distances are within
 * the cutoff, with the optional Tang-Toennies and centroid damping functions f, periodic boundary conditions
 * (each distance the minimum image), alchemical scaling and updateParametersInContext(). Every expected value is
 * computed here, in long double, from the formulas of docs/SPEC.md and InteraxThreeBodyForce.h.
 */

#include "InteraxTestUtilities.h"
#include <map>

double energyTol, forceTol, consistencyTol;

/**
 * The C9 coefficient of a triple.
 */
Real computeC9(Real alphaA, Real wA, Real alphaB, Real wB, Real alphaC, Real wC) {
    return 1.5L*alphaA*alphaB*alphaC*wA*wB*wC*(wA+wB+wC)/((wA+wB)*(wB+wC)*(wC+wA));
}

/**
 * Get the displacement from one site to another: under periodic boundary conditions the minimum image in a
 * rectangular box, which the force uses even when the cutoff exceeds half the box.
 */
static Vec3 getDisplacement(const Vec3& from, const Vec3& to, bool periodic, const Vec3* box) {
    Vec3 delta = to-from;
    if (periodic)
        for (int k = 0; k < 3; k++)
            delta[k] -= box[k][k]*floor(delta[k]/box[k][k]+0.5);
    return delta;
}

/**
 * Compute the energy of the force by looping over all triples of sites.
 *
 * @param[out] absEnergy  if not NULL, the sum of the magnitudes of the triple energies
 */
Real computeReferenceEnergy(const InteraxThreeBodyForce& force, const vector<Vec3>& positions, const Vec3* box, Real lambda, Real* absEnergy=NULL) {
    int numSites = force.getNumSites();
    vector<Vec3> sites(numSites);
    vector<double> alpha(numSites), frequency(numSites), width(numSites);
    vector<int> molecule(numSites);
    vector<bool> isAlchemical(numSites);
    for (int i = 0; i < numSites; i++) {
        vector<int> particles;
        bool alchemical;
        force.getSiteParameters(i, particles, alpha[i], frequency[i], width[i], molecule[i], alchemical);
        isAlchemical[i] = alchemical;
        if (molecule[i] < 0)
            molecule[i] = -1-i;
        for (int p : particles)
            sites[i] += positions[p];
        sites[i] /= particles.size();
    }
    bool periodic = force.usesPeriodicBoundaryConditions();
    Real cutoff = force.getCutoffDistance();
    Real radius = force.getCentroidDampingRadius();
    bool useTangToennies = (force.getDampingMethod() == InteraxThreeBodyForce::TangToennies);
    auto damping = [&] (int i, int j, Real r) {
        Real f = 1;
        if (useTangToennies)
            f *= tangToennies(3, force.getDampingScale()*2*r/((Real) width[i]+width[j]));
        if (radius > 0)
            f *= 1-expl(-powl(r/radius, 6));
        return f;
    };
    Real energy = 0, sumAbs = 0;
    for (int i = 0; i < numSites; i++)
        for (int j = i+1; j < numSites; j++)
            for (int k = j+1; k < numSites; k++) {
                if (molecule[i] == molecule[j] || molecule[i] == molecule[k] || molecule[j] == molecule[k])
                    continue;
                Vec3 a = getDisplacement(sites[i], sites[j], periodic, box);
                Vec3 b = getDisplacement(sites[i], sites[k], periodic, box);
                Vec3 c = getDisplacement(sites[j], sites[k], periodic, box);
                Real ra = sqrtl((Real) a.dot(a)), rb = sqrtl((Real) b.dot(b)), rc = sqrtl((Real) c.dot(c));
                if (ra >= cutoff || rb >= cutoff || rc >= cutoff)
                    continue;

                // The angle at each vertex is between the vectors to its two neighbors.

                Real cosA = a.dot(b)/(ra*rb);
                Real cosB = -a.dot(c)/(ra*rc);
                Real cosC = b.dot(c)/(rb*rc);
                Real c9 = force.getScaleFactor()*computeC9(alpha[i], frequency[i], alpha[j], frequency[j], alpha[k], frequency[k]);
                Real e = c9*(1+3*cosA*cosB*cosC)/powl(ra*rb*rc, 3)*damping(i, j, ra)*damping(i, k, rb)*damping(j, k, rc);
                if (isAlchemical[i] || isAlchemical[j] || isAlchemical[k])
                    e *= lambda;
                energy += e;
                sumAbs += fabsl(e);
            }
    if (absEnergy != NULL)
        *absEnergy = sumAbs;
    return energy;
}

/**
 * Compute the forces as five-point finite differences of the reference energy.
 */
vector<Vec3> computeReferenceForces(const InteraxThreeBodyForce& force, const vector<Vec3>& positions, const Vec3* box, Real lambda) {
    vector<Vec3> forces(positions.size());
    for (int i = 0; i < positions.size(); i++)
        for (int k = 0; k < 3; k++) {
            auto energy = [&] (Real x) {
                vector<Vec3> displaced = positions;
                displaced[i][k] = (double) x;
                return computeReferenceEnergy(force, displaced, box, lambda);
            };
            // The step is a power of 2, so that the displaced coordinates are exact.
            forces[i][k] = -(double) fivePointDerivative(energy, (Real) positions[i][k], 1.0L/(1<<17));
        }
    return forces;
}

/**
 * Compare the energy and forces of a Context with the reference values.
 */
void compareWithReference(Context& context, const InteraxThreeBodyForce& force, Real lambda) {
    State state = context.getState(State::Positions | State::Energy | State::Forces);
    Vec3 box[3];
    state.getPeriodicBoxVectors(box[0], box[1], box[2]);
    Real absEnergy;
    Real expected = computeReferenceEnergy(force, state.getPositions(), box, lambda, &absEnergy);
    if (!(fabsl(state.getPotentialEnergy()-expected) <= energyTol*absEnergy)) {
        stringstream details;
        details.precision(17);
        details << "Expected energy " << (double) expected << ", found " << state.getPotentialEnergy() << " (sum of |triple energies| " << (double) absEnergy << ")";
        throwException(__FILE__, __LINE__, details.str());
    }
    assertForcesEqual(computeReferenceForces(force, state.getPositions(), box, lambda), state.getForces(), forceTol);
}

/**
 * Create a force whose sites are single particles, with the parameters of InteraxFF molecular sites.
 */
InteraxThreeBodyForce* createSingleParticleSites(System& system, int numSites) {
    static const double parameters[3][3] = {{0.004087456746801944, 911.0681141195369, 0.03}, {0.0016246535729804365, 1262.664320774599, 0.025},
                                            {0.008817709933796472, 747.5256543931184, 0.035}};
    InteraxThreeBodyForce* force = new InteraxThreeBodyForce();
    for (int i = 0; i < numSites; i++) {
        system.addParticle(1.0);
        const double* p = parameters[i%3];
        force->addSite({i}, p[0], p[1], p[2]);
    }
    system.addForce(force);
    return force;
}

/**
 * Test a single triple against the formula written out with the law of cosines.
 */
void testTriple() {
    System system;
    InteraxThreeBodyForce* force = createSingleParticleSites(system, 3);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions = {Vec3(0.1, -0.2, 0.05), Vec3(0.45, 0.1, -0.1), Vec3(-0.05, 0.3, 0.2)};
    context.setPositions(positions);
    double rAB = sqrt((positions[1]-positions[0]).dot(positions[1]-positions[0]));
    double rAC = sqrt((positions[2]-positions[0]).dot(positions[2]-positions[0]));
    double rBC = sqrt((positions[2]-positions[1]).dot(positions[2]-positions[1]));
    double cosA = (rAB*rAB+rAC*rAC-rBC*rBC)/(2*rAB*rAC);
    double cosB = (rAB*rAB+rBC*rBC-rAC*rAC)/(2*rAB*rBC);
    double cosC = (rAC*rAC+rBC*rBC-rAB*rAB)/(2*rAC*rBC);
    double aA = 0.004087456746801944, wA = 911.0681141195369;
    double aB = 0.0016246535729804365, wB = 1262.664320774599;
    double aC = 0.008817709933796472, wC = 747.5256543931184;
    double c9 = 1.5*aA*aB*aC*wA*wB*wC*(wA+wB+wC)/((wA+wB)*(wB+wC)*(wC+wA));
    double expected = c9*(1+3*cosA*cosB*cosC)/pow(rAB*rAC*rBC, 3);
    ASSERT_EQUAL_REL(expected, getEnergyAndForces(context).getPotentialEnergy(), energyTol);
    compareWithReference(context, *force, 1);

    // The C6 of a site is 3/4 alpha^2 w; for three identical sites C9 = 3/4 alpha C6.

    force->setSiteParameters(1, {1}, aA, wA, 0.03, -1, false);
    force->setSiteParameters(2, {2}, aA, wA, 0.03, -1, false);
    force->updateParametersInContext(context);
    double c6 = 0.75*aA*aA*wA;
    expected = 0.75*aA*c6*(1+3*cosA*cosB*cosC)/pow(rAB*rAC*rBC, 3);
    ASSERT_EQUAL_REL(expected, getEnergyAndForces(context).getPotentialEnergy(), energyTol);

    // The scale factor multiplies every triple.

    force->setScaleFactor(0.7);
    force->updateParametersInContext(context);
    ASSERT_EQUAL_REL(0.7*expected, getEnergyAndForces(context).getPotentialEnergy(), energyTol);
}

/**
 * Test sites that are the centroids of several particles: the force on a site is shared equally among them.
 */
void testCentroidSites() {
    const int numMolecules = 4;
    System system;
    InteraxThreeBodyForce* force = new InteraxThreeBodyForce();
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(3, sfmt);
    for (int i = 0; i < numMolecules; i++) {
        // Molecules of 1, 2, 3 and 4 particles.
        vector<int> particles;
        Vec3 center(0.4*cos(2*M_PI*i/numMolecules), 0.4*sin(2*M_PI*i/numMolecules), 0.1*i);
        for (int j = 0; j <= i; j++) {
            particles.push_back(system.addParticle(1.0));
            positions.push_back(center+Vec3(random01(sfmt)-0.5, random01(sfmt)-0.5, random01(sfmt)-0.5)*0.15);
        }
        force->addSite(particles, 0.004+0.001*i, 900.0-50*i);
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    compareWithReference(context, *force, 1);
}

/**
 * Test that triples with two sites of the same molecule do not interact, and that a negative molecule index puts a
 * site in a molecule of its own.
 */
void testMolecules() {
    System system;
    InteraxThreeBodyForce* force = createSingleParticleSites(system, 5);
    vector<int> molecules = {0, 0, -1, 3, -1};
    for (int i = 0; i < 5; i++) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force->getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        force->setSiteParameters(i, particles, alpha, frequency, width, molecules[i], isAlchemical);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.15, 0, 0), Vec3(0, 0.4, 0), Vec3(0.3, 0.3, 0.2), Vec3(-0.2, 0.1, 0.35)};
    context.setPositions(positions);
    compareWithReference(context, *force, 1);

    // The interacting triples are those of sites {0 or 1, 2, 3, 4} without both 0 and 1: 0-2-3, 0-2-4, 0-3-4,
    // 1-2-3, 1-2-4, 1-3-4 and 2-3-4. Removing site 1 from molecule 0 adds the triples 0-1-2, 0-1-3 and 0-1-4.

    double energy = getEnergyAndForces(context).getPotentialEnergy();
    vector<int> particles;
    double alpha, frequency, width;
    int molecule;
    bool isAlchemical;
    force->getSiteParameters(1, particles, alpha, frequency, width, molecule, isAlchemical);
    force->setSiteParameters(1, particles, alpha, frequency, width, 1, isAlchemical);
    VerletIntegrator integrator2(0.001);
    Context context2(system, integrator2, platform);
    context2.setPositions(positions);
    Vec3 box[3];
    Real expected = computeReferenceEnergy(*force, positions, box, 1);
    ASSERT_EQUAL_REL(expected, getEnergyAndForces(context2).getPotentialEnergy(), energyTol);
    ASSERT(fabs(expected-energy) > 1e-3*fabs(energy));
}

/**
 * Test the cutoff: a triple interacts only if all three of its distances are within it.
 */
void testCutoff() {
    System system;
    InteraxThreeBodyForce* force = createSingleParticleSites(system, 4);
    force->setCutoffDistance(0.5);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    // Sites 0, 1, 2 form a triangle within the cutoff; site 3 is within the cutoff of 2 only, so the energy is that
    // of the triple 0-1-2.

    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.4, 0, 0), Vec3(0.2, 0.3, 0), Vec3(0.2, 0.75, 0)};
    context.setPositions(positions);
    State state = getEnergyAndForces(context);
    System system3;
    createSingleParticleSites(system3, 3);
    VerletIntegrator integrator3(0.001);
    Context context3(system3, integrator3, platform);
    context3.setPositions({positions[0], positions[1], positions[2]});
    ASSERT_EQUAL_REL(getEnergyAndForces(context3).getPotentialEnergy(), state.getPotentialEnergy(), energyTol);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[3], 0);
    compareWithReference(context, *force, 1);

    // Without the cutoff the other triples contribute.

    InteraxThreeBodyForce noCutoff = *force;
    noCutoff.setCutoffDistance(10.0);
    Vec3 box[3];
    ASSERT(fabsl(computeReferenceEnergy(noCutoff, positions, box, 1)-state.getPotentialEnergy()) > 1e-3*fabs(state.getPotentialEnergy()));
}

/**
 * Create a periodic system of random single-particle sites in molecules of two, with a minimum separation.
 */
InteraxThreeBodyForce* createRandomSystem(System& system, int numSites, double boxSize, double cutoff, vector<Vec3>& positions, int seed) {
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    InteraxThreeBodyForce* force = createSingleParticleSites(system, numSites);
    for (int i = 0; i < numSites; i++) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force->getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        force->setSiteParameters(i, particles, alpha, frequency, width, i/2, isAlchemical);
    }
    force->setCutoffDistance(cutoff);
    force->setUsesPeriodicBoundaryConditions(true);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(seed, sfmt);
    Vec3 box[3] = {Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize)};
    positions.clear();
    while (positions.size() < numSites) {
        Vec3 p = Vec3(random01(sfmt), random01(sfmt), random01(sfmt))*boxSize*3 - Vec3(boxSize, boxSize, boxSize);
        bool accepted = true;
        for (const Vec3& q : positions) {
            Vec3 delta = getDisplacement(p, q, true, box);
            accepted &= (delta.dot(delta) > 0.25*0.25);
        }
        if (accepted)
            positions.push_back(p);
    }
    return force;
}

/**
 * Test periodic boundary conditions, including a cutoff larger than half the box (as in the InteraxFF liquid boxes),
 * where the minimum images of the three pairs of a triple need not close into a triangle.
 */
void testPeriodic() {
    for (double boxSize : {3.5, 2.2}) {
        System system;
        vector<Vec3> positions;
        InteraxThreeBodyForce* force = createRandomSystem(system, 40, boxSize, 1.2, positions, 4);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions(positions);
        compareWithReference(context, *force, 1);
    }
}

/**
 * Test the Tang-Toennies damping f_3 of each pair, the centroid damping 1 - exp(-(r/R0)^6), and both together, at
 * distances where they matter.
 */
void testDamping() {
    for (int method = 0; method < 2; method++)
        for (double radius : {0.0, 0.3}) {
            System system;
            InteraxThreeBodyForce* force = createSingleParticleSites(system, 4);
            force->setDampingMethod((InteraxThreeBodyForce::DampingMethod) method);
            force->setDampingScale(0.8);
            force->setCentroidDampingRadius(radius);
            VerletIntegrator integrator(0.001);
            Context context(system, integrator, platform);
            vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.12, 0.02, 0), Vec3(0.05, 0.18, 0.03), Vec3(0.2, 0.2, 0.25)};
            context.setPositions(positions);
            compareWithReference(context, *force, 1);
            if (method == 0 && radius == 0)
                continue;
            unique_ptr<InteraxThreeBodyForce> undamped(new InteraxThreeBodyForce(*force));
            undamped->setDampingMethod(InteraxThreeBodyForce::NoDamping);
            undamped->setCentroidDampingRadius(0);
            Vec3 box[3];
            State state = getEnergyAndForces(context);
            ASSERT(fabsl(computeReferenceEnergy(*undamped, positions, box, 1)-state.getPotentialEnergy()) > 0.01*fabs(state.getPotentialEnergy()));
        }
}

/**
 * Test the alchemical scaling: a triple with at least one alchemical site is multiplied by the alchemical parameter.
 */
void testAlchemical() {
    System system;
    InteraxThreeBodyForce* force = createSingleParticleSites(system, 5);
    for (int i : {0, 1}) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force->getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        force->setSiteParameters(i, particles, alpha, frequency, width, molecule, true);
    }
    force->setAlchemicalParameterName("lambda_test");
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.35, 0, 0), Vec3(0, 0.4, 0), Vec3(0.3, 0.3, 0.2), Vec3(-0.2, 0.1, 0.35)};
    context.setPositions(positions);
    ASSERT_EQUAL(1.0, context.getParameter("lambda_test"));
    Vec3 box[3];
    double fullEnergy = getEnergyAndForces(context).getPotentialEnergy();
    Real scaled = computeReferenceEnergy(*force, positions, box, 1)-computeReferenceEnergy(*force, positions, box, 0);
    Real unscaled = computeReferenceEnergy(*force, positions, box, 0);
    ASSERT_EQUAL_REL(scaled+unscaled, fullEnergy, energyTol);

    // Only the triple 2-3-4 has no alchemical site.

    System system3;
    createSingleParticleSites(system3, 5);
    InteraxThreeBodyForce& force3 = dynamic_cast<InteraxThreeBodyForce&>(system3.getForce(0));
    for (int i : {0, 1}) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force3.getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        force3.setSiteParameters(i, particles, 0.0, frequency, width, molecule, false);
    }
    VerletIntegrator integrator3(0.001);
    Context context3(system3, integrator3, platform);
    context3.setPositions(positions);
    ASSERT_EQUAL_REL(getEnergyAndForces(context3).getPotentialEnergy(), (double) unscaled, energyTol);
    for (double lambda : {0.0, 0.25, 0.9}) {
        context.setParameter("lambda_test", lambda);
        ASSERT_EQUAL_REL(unscaled+lambda*scaled, getEnergyAndForces(context).getPotentialEnergy(), energyTol);
        compareWithReference(context, *force, lambda);
    }
}

/**
 * Test alchemical sites at lambda = 0 that coincide with another site, as a decoupled solute lets water through it:
 * their triples must stay exactly zero, although 1/(u v w)^(5/2) overflows in single precision.
 */
void testAlchemicalOverlap() {
    System system;
    InteraxThreeBodyForce* force = createSingleParticleSites(system, 5);
    for (int i : {0, 1}) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force->getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        force->setSiteParameters(i, particles, alpha, frequency, width, molecule, true);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions = {Vec3(0, 0.4, 1e-9), Vec3(0.35, 0, 0), Vec3(0, 0.4, 0), Vec3(0.3, 0.3, 0.2), Vec3(-0.2, 0.1, 0.35)};
    context.setPositions(positions);
    context.setParameter("lambda_sterics", 0.0);
    State state = getEnergyAndForces(context);
    Vec3 box[3];
    ASSERT_EQUAL_REL((double) computeReferenceEnergy(*force, positions, box, 0), state.getPotentialEnergy(), energyTol);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 3; j++)
            ASSERT(isfinite(state.getForces()[i][j]));
    for (int i : {0, 1})
        ASSERT_EQUAL_VEC(Vec3(), state.getForces()[i], 0.0);
}

/**
 * Test a larger periodic system of molecular sites with damping and alchemical sites, and check the forces against
 * the energy.
 */
void testLargeSystem() {
    const int numMolecules = 50;
    const double boxSize = 2.4;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    InteraxThreeBodyForce* force = new InteraxThreeBodyForce();
    force->setCutoffDistance(1.0);
    force->setUsesPeriodicBoundaryConditions(true);
    force->setCentroidDampingRadius(0.25);
    force->setScaleFactor(1.3);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(6, sfmt);
    vector<Vec3> positions;
    Vec3 box[3] = {Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize)};
    vector<Vec3> centers;
    while (centers.size() < numMolecules) {
        Vec3 p = Vec3(random01(sfmt), random01(sfmt), random01(sfmt))*boxSize;
        bool accepted = true;
        for (const Vec3& q : centers) {
            Vec3 delta = getDisplacement(p, q, true, box);
            accepted &= (delta.dot(delta) > 0.3*0.3);
        }
        if (accepted)
            centers.push_back(p);
    }
    for (int i = 0; i < numMolecules; i++) {
        vector<int> particles;
        for (int j = 0; j < 3; j++) {
            particles.push_back(system.addParticle(1.0));
            positions.push_back(centers[i]+Vec3(random01(sfmt)-0.5, random01(sfmt)-0.5, random01(sfmt)-0.5)*0.1);
        }
        force->addSite(particles, 0.003+0.002*(i%3), 800.0+150*(i%4), 0.03, i, i < 3);
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setParameter("lambda_sterics", 0.6);
    State state = context.getState(State::Energy);
    Real absEnergy;
    Real expected = computeReferenceEnergy(*force, positions, box, 0.6, &absEnergy);
    ASSERT(fabsl(state.getPotentialEnergy()-expected) <= energyTol*absEnergy);
    checkForceEnergyConsistency(context, consistencyTol);
}

/**
 * Test updateParametersInContext().
 */
void testChangingParameters() {
    System system;
    InteraxThreeBodyForce* force = createSingleParticleSites(system, 4);
    force->setDampingMethod(InteraxThreeBodyForce::TangToennies);
    force->setCentroidDampingRadius(0.2);
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(0.25, 0.02, 0), Vec3(0.05, 0.28, 0.03), Vec3(0.2, 0.2, 0.25)};
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    double energy1 = getEnergyAndForces(context).getPotentialEnergy();

    // Change the parameters and make sure the results change and agree with the reference.

    force->setSiteParameters(1, {1}, 0.006, 1000.0, 0.028, -1, false);
    force->setScaleFactor(0.9);
    force->setDampingScale(1.2);
    force->setCentroidDampingRadius(0.25);
    force->updateParametersInContext(context);
    State state2 = getEnergyAndForces(context);
    ASSERT(fabs(energy1-state2.getPotentialEnergy()) > 1e-3*fabs(energy1));
    compareWithReference(context, *force, 1);

    // A new context with the altered parameters agrees.

    VerletIntegrator integrator2(0.001);
    Context context2(system, integrator2, platform);
    context2.setPositions(positions);
    State state3 = getEnergyAndForces(context2);
    ASSERT_EQUAL_REL(state3.getPotentialEnergy(), state2.getPotentialEnergy(), energyTol);
    assertForcesEqual(state3.getForces(), state2.getForces(), forceTol);
}

/**
 * Test that invalid settings are rejected.
 */
void testIllegalSettings() {
    {
        System system;
        InteraxThreeBodyForce* force = createSingleParticleSites(system, 3);
        force->addSite({}, 0.004, 900.0);
        assertContextCreationFails(system, platform, "a site without particles");
    }
    {
        System system;
        InteraxThreeBodyForce* force = createSingleParticleSites(system, 3);
        force->addSite({3}, 0.004, 900.0);
        assertContextCreationFails(system, platform, "a site with an invalid particle");
    }
    {
        System system;
        InteraxThreeBodyForce* force = createSingleParticleSites(system, 3);
        force->setDampingMethod(InteraxThreeBodyForce::TangToennies);
        force->setSiteParameters(2, {2}, 0.004, 900.0, 0.0, -1, false);
        assertContextCreationFails(system, platform, "Tang-Toennies damping without a width");
    }
    {
        System system;
        InteraxThreeBodyForce* force = createSingleParticleSites(system, 3);
        force->setCutoffDistance(0.0);
        assertContextCreationFails(system, platform, "a zero cutoff");
    }
    InteraxThreeBodyForce force;
    bool failed = false;
    try {
        force.setCentroidDampingRadius(-0.1);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    ASSERT(failed);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        setForceResolution(platform);
        if (getPlatformPrecision(platform, true) == "double") {
            energyTol = 1e-10;
            forceTol = 1e-8;
            consistencyTol = 1e-5;
        }
        else {
            energyTol = 1e-5;
            forceTol = 1e-4;
            consistencyTol = 2e-3;
        }
        testTriple();
        testCentroidSites();
        testMolecules();
        testCutoff();
        testPeriodic();
        testDamping();
        testAlchemical();
        testAlchemicalOverlap();
        testLargeSystem();
        testChangingParameters();
        testIllegalSettings();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
