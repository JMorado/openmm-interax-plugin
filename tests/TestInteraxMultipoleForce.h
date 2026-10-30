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
 * This tests InteraxMultipoleForce, the electrostatics and polarization of InteraxFF. The force is HIPPO's
 * multipoles and polarization (HippoNonbondedForce without repulsion, dispersion and charge transfer) with the
 * InteraxFF options, so with the options set to HIPPO's values it must reproduce HippoNonbondedForce, which these
 * tests use as an independent reference (as in OpenMM's TestHippoNonbondedForce.h, on water). The InteraxFF options
 * are tested against values computed here from their definitions:
 *
 *   core damping         the permanent field on an atom is multiplied by f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r),
 *                        a = eta*alpha of the polarized atom
 *   mutual damping scale the induced dipole couplings use the exponents kappa*alpha
 *   alchemical particles their multipoles and polarizabilities are multiplied by the alchemical parameter
 *
 * and the own GPU kernel against the default one. HIPPO is only compared where it is correct: not for forces on
 * particles with a ZOnly frame on the Reference and CPU platforms, nor with grids of different sizes in x and z on
 * the CUDA platform (docs/upstream).
 */

#include "InteraxTestUtilities.h"
#include "openmm/HippoNonbondedForce.h"
#include "openmm/NonbondedForce.h"
#include <map>

/**
 * Coulomb's constant in OpenMM's units (kJ/mol nm/e^2).
 */
const double COULOMB_CONSTANT = 138.93545764438198;

double energyTol, forceTol, dipoleTol, consistencyTol, consistencyStep;

/**
 * The parameters of a particle, shared by InteraxMultipoleForce and HippoNonbondedForce.
 */
struct MultipoleParticle {
    double charge;
    vector<double> dipole, quadrupole;
    double coreCharge, alpha, polarizability;
    int axisType, atomZ, atomX, atomY;
};

/**
 * The scale factors of an exception.
 */
struct MultipoleException {
    int particle1, particle2;
    double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale;
};

/**
 * A test system: particles, exceptions, positions, and the settings of the force.
 */
struct MultipoleSystem {
    vector<MultipoleParticle> particles;
    vector<MultipoleException> exceptions;
    vector<Vec3> positions;
    bool periodic = false;
    double boxSize = 0, cutoff = 1.0, pmeAlpha = 0;
    int pmeGrid = 0;
    vector<double> extrapolationCoefficients = {0.042, 0.635, 0.414};
};

/**
 * Add a water molecule with the HIPPO water parameters of OpenMM's TestHippoNonbondedForce.h.
 */
void addWater(MultipoleSystem& s, const Vec3& oxygen, const Vec3& hydrogen1, const Vec3& hydrogen2) {
    int o = s.particles.size();
    double bohr = 0.52917720859;
    double ds = 0.1*bohr;
    double qs = 0.01*bohr*bohr/3.0;
    s.particles.push_back({-0.38280, {0.0, 0.0, ds*0.05477}, {qs*0.69866, 0.0, 0.0, 0.0, qs*-0.60471, 0.0, 0.0, 0.0, qs*-0.09395}, 6.0,
                           10*4.7075, 0.001*0.795, InteraxMultipoleForce::Bisector, o+1, o+2, -1});
    for (int h : {1, 2})
        s.particles.push_back({0.19140, {0.0, 0.0, ds*-0.20097}, {qs*0.03881, 0.0, 0.0, 0.0, qs*0.02214, 0.0, 0.0, 0.0, qs*-0.06095}, 1.0,
                               10*4.7909, 0.001*0.341, InteraxMultipoleForce::ZThenX, o, o+3-h, -1});
    s.exceptions.push_back({o, o+1, 0.0, 0.0, 0.2});
    s.exceptions.push_back({o, o+2, 0.0, 0.0, 0.2});
    s.exceptions.push_back({o+1, o+2, 0.0, 0.0, 1.0});
    s.positions.push_back(oxygen);
    s.positions.push_back(hydrogen1);
    s.positions.push_back(hydrogen2);
}

/**
 * The water dimer of OpenMM's TestHippoNonbondedForce.h.
 */
MultipoleSystem createWaterDimer() {
    MultipoleSystem s;
    addWater(s, 0.1*Vec3(1.505434, 0.0, -0.065656), 0.1*Vec3(0.553912, 0.0, 0.057710), 0.1*Vec3(1.907155, 0.0, 0.801980));
    addWater(s, 0.1*Vec3(-1.436029, 0.0, 0.060505), 0.1*Vec3(-1.781197, 0.772272, -0.388976), 0.1*Vec3(-1.781197, -0.772272, -0.388976));
    return s;
}

/**
 * A water trimer without symmetry.
 */
MultipoleSystem createWaterTrimer() {
    MultipoleSystem s = createWaterDimer();
    addWater(s, Vec3(0.02, 0.29, 0.05), Vec3(0.08, 0.22, 0.02), Vec3(-0.06, 0.25, 0.09));
    return s;
}

/**
 * Rotate a vector by the unit quaternion (w, x, y, z).
 */
Vec3 rotate(const double q[4], const Vec3& v) {
    Vec3 u(q[1], q[2], q[3]);
    Vec3 t = u.cross(v)*2;
    return v + t*q[0] + u.cross(t);
}

/**
 * A periodic box of water molecules with random orientations on a cubic lattice.
 */
MultipoleSystem createWaterBox(int numPerSide, double boxSize, int seed) {
    MultipoleSystem s;
    s.periodic = true;
    s.boxSize = boxSize;
    s.cutoff = 0.6;
    s.pmeAlpha = 4.0;
    s.pmeGrid = 20;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(seed, sfmt);
    double spacing = boxSize/numPerSide;
    double angle = 104.52*M_PI/180, bond = 0.09572;
    Vec3 h1(bond*sin(angle/2), 0, bond*cos(angle/2)), h2(-bond*sin(angle/2), 0, bond*cos(angle/2));
    for (int i = 0; i < numPerSide; i++)
        for (int j = 0; j < numPerSide; j++)
            for (int k = 0; k < numPerSide; k++) {
                double q[4], norm = 0;
                for (int m = 0; m < 4; m++) {
                    q[m] = random01(sfmt)-0.5;
                    norm += q[m]*q[m];
                }
                for (int m = 0; m < 4; m++)
                    q[m] /= sqrt(norm);
                Vec3 oxygen = Vec3(i+0.5, j+0.5, k+0.5)*spacing + Vec3(random01(sfmt)-0.5, random01(sfmt)-0.5, random01(sfmt)-0.5)*0.02;
                addWater(s, oxygen, oxygen+rotate(q, h1), oxygen+rotate(q, h2));
            }
    return s;
}

/**
 * Options of InteraxMultipoleForce beyond HIPPO.
 */
struct InteraxOptions {
    bool coreDamping = false, useOwnKernel = false;
    double coreDampingScale = 1.0, mutualDampingScale = 1.0;
    vector<bool> isAlchemical;
    string alchemicalParameter = "lambda_electrostatics";
    InteraxMultipoleForce::ChiralityMode chiralityMode = InteraxMultipoleForce::AmoebaChirality;
};

InteraxMultipoleForce* createInteraxForce(const MultipoleSystem& s, const InteraxOptions& options=InteraxOptions()) {
    InteraxMultipoleForce* force = new InteraxMultipoleForce();
    force->setNonbondedMethod(s.periodic ? InteraxMultipoleForce::PME : InteraxMultipoleForce::NoCutoff);
    force->setCutoffDistance(s.cutoff);
    force->setPMEParameters(s.pmeAlpha, s.pmeGrid, s.pmeGrid, s.pmeGrid);
    force->setExtrapolationCoefficients(s.extrapolationCoefficients);
    force->setCoreDamping(options.coreDamping);
    force->setCoreDampingScale(options.coreDampingScale);
    force->setMutualDampingScale(options.mutualDampingScale);
    force->setUseOwnKernel(options.useOwnKernel);
    force->setAlchemicalParameterName(options.alchemicalParameter);
    force->setChiralityMode(options.chiralityMode);
    for (int i = 0; i < s.particles.size(); i++) {
        const MultipoleParticle& p = s.particles[i];
        force->addParticle(p.charge, p.dipole, p.quadrupole, p.coreCharge, p.alpha, p.polarizability, p.axisType, p.atomZ, p.atomX, p.atomY,
                options.isAlchemical.empty() ? false : options.isAlchemical[i]);
    }
    for (const MultipoleException& e : s.exceptions)
        force->addException(e.particle1, e.particle2, e.multipoleMultipoleScale, e.dipoleMultipoleScale, e.dipoleDipoleScale);
    return force;
}

/**
 * Create the HippoNonbondedForce with the same electrostatics and polarization, and no repulsion, dispersion or
 * charge transfer. HIPPO requires positive polarizabilities: zero becomes 1e-14 nm^3.
 */
HippoNonbondedForce* createHippoForce(const MultipoleSystem& s) {
    HippoNonbondedForce* force = new HippoNonbondedForce();
    force->setNonbondedMethod(s.periodic ? HippoNonbondedForce::PME : HippoNonbondedForce::NoCutoff);
    force->setCutoffDistance(s.cutoff);
    force->setSwitchingDistance(0.9*s.cutoff);
    force->setPMEParameters(s.pmeAlpha, s.pmeGrid, s.pmeGrid, s.pmeGrid);
    force->setDPMEParameters(s.pmeAlpha, s.pmeGrid, s.pmeGrid, s.pmeGrid);
    force->setExtrapolationCoefficients(s.extrapolationCoefficients);
    for (const MultipoleParticle& p : s.particles)
        force->addParticle(p.charge, p.dipole, p.quadrupole, p.coreCharge, p.alpha, 0.0, 10*40.0, 0.0, 0.0, 1.0, 10*4.0,
                max(p.polarizability, 1e-14), p.axisType, p.atomZ, p.atomX, p.atomY);
    for (const MultipoleException& e : s.exceptions)
        force->addException(e.particle1, e.particle2, e.multipoleMultipoleScale, e.dipoleMultipoleScale, e.dipoleDipoleScale, 0.0, 0.0, 0.0);
    return force;
}

/**
 * Create a System for a test system and a force.
 */
void createSystem(System& system, const MultipoleSystem& s, Force* force) {
    addParticles(system, s.particles.size());
    if (s.periodic)
        system.setDefaultPeriodicBoxVectors(Vec3(s.boxSize, 0, 0), Vec3(0, s.boxSize, 0), Vec3(0, 0, s.boxSize));
    system.addForce(force);
}

/**
 * The results of a force for a test system.
 */
struct MultipoleResults {
    double energy;
    vector<Vec3> forces, labDipoles, inducedDipoles;
};

/**
 * Compute the energy, forces and dipoles of an InteraxMultipoleForce or HippoNonbondedForce (the force of a System
 * built by createSystem()).
 */
MultipoleResults evaluate(System& system, const vector<Vec3>& positions, Platform& p, const map<string, double>& parameters=map<string, double>()) {
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, p);
    context.setPositions(positions);
    for (auto& parameter : parameters)
        context.setParameter(parameter.first, parameter.second);
    MultipoleResults results;
    State state = context.getState(State::Energy | State::Forces);
    results.energy = state.getPotentialEnergy();
    results.forces = state.getForces();
    if (InteraxMultipoleForce* force = dynamic_cast<InteraxMultipoleForce*>(&system.getForce(0))) {
        force->getLabFramePermanentDipoles(context, results.labDipoles);
        force->getInducedDipoles(context, results.inducedDipoles);
    }
    else if (HippoNonbondedForce* force = dynamic_cast<HippoNonbondedForce*>(&system.getForce(0))) {
        force->getLabFramePermanentDipoles(context, results.labDipoles);
        force->getInducedDipoles(context, results.inducedDipoles);
    }
    return results;
}

/**
 * Compare two sets of results.
 *
 * @param compareForces  whether to compare the forces (not with HIPPO for ZOnly frames on the Reference and CPU
 *                       platforms)
 */
void compareResults(const MultipoleResults& expected, const MultipoleResults& found, double tolerance, bool compareForces=true) {
    ASSERT_EQUAL_TOL(expected.energy, found.energy, tolerance);
    if (compareForces)
        assertForcesEqual(expected.forces, found.forces, 10*tolerance);
    assertForcesEqual(expected.labDipoles, found.labDipoles, tolerance);
    assertForcesEqual(expected.inducedDipoles, found.inducedDipoles, 10*tolerance);
}

/**
 * Compare InteraxMultipoleForce on the platform being tested with HippoNonbondedForce on the Reference platform, and
 * on double precision platforms also with HippoNonbondedForce on the same platform.
 *
 * @param compareReferenceForces  whether to compare the forces with HIPPO on the Reference platform (not for ZOnly
 *                                frames, whose forces are wrong there)
 */
void compareWithHippo(const MultipoleSystem& s, bool compareReferenceForces=true) {
    System interaxSystem, hippoSystem;
    createSystem(interaxSystem, s, createInteraxForce(s));
    createSystem(hippoSystem, s, createHippoForce(s));
    MultipoleResults found = evaluate(interaxSystem, s.positions, platform);
    compareResults(evaluate(hippoSystem, s.positions, Platform::getPlatformByName("Reference")), found, energyTol, compareReferenceForces);
    if (platform.getName() != "Reference" && getPlatformPrecision(platform, true) == "double")
        compareResults(evaluate(hippoSystem, s.positions, platform), found, energyTol, platform.getName() != "CPU");
}

/**
 * Test the water dimer against HIPPO, with mutual and direct polarization and with a non-polarizable atom.
 */
void testWaterDimer() {
    MultipoleSystem s = createWaterDimer();
    compareWithHippo(s);
    s.extrapolationCoefficients = {1.0};
    compareWithHippo(s);
    s.extrapolationCoefficients = {0.0, 0.3, 0.7};
    compareWithHippo(s);
    s.particles[4].polarizability = 0.0;
    compareWithHippo(s);
}

/**
 * Test a water trimer with intermolecular exceptions of all three kinds against HIPPO.
 */
void testExceptions() {
    MultipoleSystem s = createWaterTrimer();
    s.exceptions.push_back({0, 3, 0.4, 0.6, 0.3});
    s.exceptions.push_back({1, 7, 0.0, 1.0, 0.0});
    s.exceptions.push_back({4, 8, 1.0, 0.0, 0.5});
    s.exceptions.push_back({2, 6, 0.7, 0.2, 1.0});
    compareWithHippo(s);

    // Replacing an exception gives the same result as adding it with the new scales.

    System system;
    createSystem(system, s, createInteraxForce(s));
    InteraxMultipoleForce& force = dynamic_cast<InteraxMultipoleForce&>(system.getForce(0));
    MultipoleResults expected = evaluate(system, s.positions, platform);
    force.setExceptionParameters(force.getNumExceptions()-1, 2, 6, 0.0, 0.0, 0.0);
    int index = force.addException(6, 2, 0.7, 0.2, 1.0, true);
    ASSERT_EQUAL(force.getNumExceptions()-1, index);
    compareResults(expected, evaluate(system, s.positions, platform), energyTol);
    bool failed = false;
    try {
        force.addException(2, 6, 0.7, 0.2, 1.0);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    ASSERT(failed);
}

/**
 * Test a periodic water box with PME against HIPPO, and the PME parameters of the context.
 */
void testWaterBox() {
    MultipoleSystem s = createWaterBox(4, 1.3, 0);
    compareWithHippo(s);
    System system;
    InteraxMultipoleForce* force = createInteraxForce(s);
    createSystem(system, s, force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    double alpha;
    int nx, ny, nz;
    force->getPMEParametersInContext(context, alpha, nx, ny, nz);
    ASSERT_EQUAL_TOL(4.0, alpha, 1e-10);
    ASSERT_EQUAL(20, nx);
    ASSERT_EQUAL(20, ny);
    ASSERT_EQUAL(20, nz);
}

/**
 * Test the mutual damping scale kappa. In a system of point charges (no valence charge) and neutral polarizable
 * atoms, the exponents of the polarizable atoms enter only the damping of the induced dipole couplings, so the
 * force with kappa is HIPPO with the exponents of the polarizable atoms multiplied by kappa.
 */
void testMutualDampingScale() {
    for (bool periodic : {false, true}) {
        MultipoleSystem s;
        s.periodic = periodic;
        s.boxSize = 2.0;
        s.cutoff = 0.8;
        s.pmeAlpha = 3.5;
        s.pmeGrid = 24;
        OpenMM_SFMT::SFMT sfmt;
        init_gen_rand(7, sfmt);
        for (int i = 0; i < 12; i++) {
            Vec3 position = Vec3(i%2, (i/2)%2, i/4)*0.25 + Vec3(random01(sfmt), random01(sfmt), random01(sfmt))*0.05;
            if (i%3 == 0)
                s.particles.push_back({0.5-(i%2), {0, 0, 0}, vector<double>(9, 0.0), 0.5-(i%2), 30.0, 0.0, InteraxMultipoleForce::NoAxisType, -1, -1, -1});
            else
                s.particles.push_back({0.0, {0, 0, 0}, vector<double>(9, 0.0), 0.0, 35.0+3*i, 0.001+0.0002*i, InteraxMultipoleForce::NoAxisType, -1, -1, -1});
            s.positions.push_back(position);
        }
        const double kappa = 0.8;
        System interaxSystem, hippoSystem;
        InteraxOptions options;
        options.mutualDampingScale = kappa;
        createSystem(interaxSystem, s, createInteraxForce(s, options));
        MultipoleSystem scaled = s;
        for (MultipoleParticle& p : scaled.particles)
            if (p.polarizability > 0)
                p.alpha *= kappa;
        createSystem(hippoSystem, scaled, createHippoForce(scaled));
        MultipoleResults expected = evaluate(hippoSystem, s.positions, Platform::getPlatformByName("Reference"));
        MultipoleResults found = evaluate(interaxSystem, s.positions, platform);
        compareResults(expected, found, energyTol);

        // Without the scale the result is different.

        System unscaledSystem;
        createSystem(unscaledSystem, s, createInteraxForce(s));
        ASSERT(fabs(evaluate(unscaledSystem, s.positions, platform).energy-found.energy) > 1e-4*fabs(found.energy));
    }
}

/**
 * Test core damping in the simplest case it applies to: a point charge q and a neutral atom of polarizability
 * alpha_p and exponent alpha at distance r, with direct polarization. The induced dipole is alpha_p q f(r)/r^2 along
 * the separation and the energy is -k alpha_p q^2 f(r)^2/(2 r^4), f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r) with
 * a = eta*alpha (f = 1 without core damping, HIPPO). A dipole-multipole exception scale s multiplies the field by s.
 */
void testCoreDamping() {
    const double q = 0.8, polarizability = 0.0012, alpha = 42.0;
    for (int variant = 0; variant < 4; variant++) {
        bool coreDamping = (variant > 0);
        double eta = (variant == 2 ? 0.7 : 1.0);
        double scale = (variant == 3 ? 0.6 : 1.0);
        MultipoleSystem s;
        s.extrapolationCoefficients = {1.0};
        s.particles.push_back({q, {0, 0, 0}, vector<double>(9, 0.0), q, 30.0, 0.0, InteraxMultipoleForce::NoAxisType, -1, -1, -1});
        s.particles.push_back({0.0, {0, 0, 0}, vector<double>(9, 0.0), 0.0, alpha, polarizability, InteraxMultipoleForce::NoAxisType, -1, -1, -1});
        if (scale != 1.0)
            s.exceptions.push_back({0, 1, 1.0, scale, 1.0});
        s.positions = {Vec3(0, 0, 0), Vec3(0, 0, 0)};
        System system;
        InteraxOptions options;
        options.coreDamping = coreDamping;
        options.coreDampingScale = eta;
        createSystem(system, s, createInteraxForce(s, options));
        Vec3 direction = Vec3(0.2, -0.4, 0.5)/sqrt(0.45);
        for (double r : {0.03, 0.08, 0.15, 0.3}) {
            auto damping = [&] (double x) {
                double ax = eta*alpha*x;
                return (coreDamping ? 1-(1+ax+0.5*ax*ax)*exp(-ax) : 1.0);
            };
            auto energy = [&] (double x) {
                double field = scale*q*damping(x)/(x*x);
                return -0.5*COULOMB_CONSTANT*polarizability*field*field;
            };
            s.positions[1] = direction*r;
            MultipoleResults results = evaluate(system, s.positions, platform);
            ASSERT_EQUAL_REL(energy(r), results.energy, energyTol);
            ASSERT_EQUAL_VEC(direction*(polarizability*scale*q*damping(r)/(r*r)), results.inducedDipoles[1], dipoleTol);
            double h = 1e-5*r;
            double dEdR = (energy(r-2*h)-8*energy(r-h)+8*energy(r+h)-energy(r+2*h))/(12*h);
            assertForcesEqual({direction*dEdR, -direction*dEdR}, results.forces, forceTol);
        }
    }
}

/**
 * Test core damping with PME. The force adds (f - 1) times the full permanent field of each pair within the cutoff
 * to the PME field, so for a point charge and a neutral polarizable atom with direct polarization the induced
 * dipoles with and without core damping differ by alpha_p (f - 1) q/r^2 along the separation, and the energies by
 * -k (|mu_on|^2 - |mu_off|^2)/(2 alpha_p).
 */
void testCoreDampingPme() {
    const double q = 0.8, polarizability = 0.0012, alpha = 42.0, eta = 0.7;
    MultipoleSystem s;
    s.periodic = true;
    s.boxSize = 2.0;
    s.cutoff = 0.7;
    s.pmeAlpha = 3.2;
    s.pmeGrid = 24;
    s.extrapolationCoefficients = {1.0};
    s.particles.push_back({q, {0, 0, 0}, vector<double>(9, 0.0), q, 30.0, 0.0, InteraxMultipoleForce::NoAxisType, -1, -1, -1});
    s.particles.push_back({0.0, {0, 0, 0}, vector<double>(9, 0.0), 0.0, alpha, polarizability, InteraxMultipoleForce::NoAxisType, -1, -1, -1});
    Vec3 direction = Vec3(0.2, -0.4, 0.5)/sqrt(0.45);
    for (double r : {0.05, 0.12}) {
        s.positions = {Vec3(0.3, 0.4, 0.5), Vec3(0.3, 0.4, 0.5)+direction*r};
        System systemOff, systemOn;
        InteraxOptions options;
        createSystem(systemOff, s, createInteraxForce(s, options));
        options.coreDamping = true;
        options.coreDampingScale = eta;
        createSystem(systemOn, s, createInteraxForce(s, options));
        MultipoleResults off = evaluate(systemOff, s.positions, platform);
        MultipoleResults on = evaluate(systemOn, s.positions, platform);
        double ar = eta*alpha*r;
        double f = 1-(1+ar+0.5*ar*ar)*exp(-ar);
        Vec3 expectedChange = direction*(polarizability*(f-1)*q/(r*r));
        ASSERT_EQUAL_VEC(expectedChange, on.inducedDipoles[1]-off.inducedDipoles[1], dipoleTol);
        double expectedEnergyChange = -0.5*COULOMB_CONSTANT*(on.inducedDipoles[1].dot(on.inducedDipoles[1])-off.inducedDipoles[1].dot(off.inducedDipoles[1]))/polarizability;
        ASSERT_EQUAL_TOL(expectedEnergyChange, on.energy-off.energy, energyTol);
    }
}

/**
 * Test the forces with all the InteraxFF options (core damping, its scale and the mutual damping scale) against
 * the energy, for the water trimer and the water box.
 */
void testInteraxOptionsForces() {
    InteraxOptions options;
    options.coreDamping = true;
    options.coreDampingScale = 0.7;
    options.mutualDampingScale = 0.8;
    {
        MultipoleSystem s = createWaterTrimer();
        System system;
        createSystem(system, s, createInteraxForce(s, options));
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions(s.positions);
        if (getPlatformPrecision(platform, true) == "double")
            checkFiniteDifferenceForces(context, 1e-6);
        checkForceEnergyConsistency(context, consistencyTol, consistencyStep);
    }
    {
        MultipoleSystem s = createWaterBox(3, 1.25, 1);
        System system;
        createSystem(system, s, createInteraxForce(s, options));
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions(s.positions);
        checkForceEnergyConsistency(context, consistencyTol, consistencyStep);
    }
}

/**
 * Test particles with a ZOnly frame (with multipoles symmetric about z, as the frame requires): forces against the
 * energy, and energies and dipoles against HIPPO (whose forces are wrong for ZOnly on the Reference and CPU
 * platforms).
 */
void testZOnlyFrame() {
    MultipoleSystem s;
    double qs = 0.0001;
    for (int m = 0; m < 3; m++) {
        int a = s.particles.size();
        s.particles.push_back({0.3, {0.0, 0.0, 0.005}, {qs, 0, 0, 0, qs, 0, 0, 0, -2*qs}, 2.0, 40.0, 0.0009, InteraxMultipoleForce::ZOnly, a+1, -1, -1});
        s.particles.push_back({-0.3, {0.0, 0.0, -0.004}, {-qs, 0, 0, 0, -qs, 0, 0, 0, 2*qs}, 4.0, 45.0, 0.0011, InteraxMultipoleForce::ZOnly, a, -1, -1});
        s.exceptions.push_back({a, a+1, 0.0, 0.0, 1.0});
    }
    s.positions = {Vec3(0, 0, 0), Vec3(0.11, 0.02, 0.01), Vec3(0.05, 0.31, 0.02), Vec3(0.02, 0.33, 0.13),
                   Vec3(-0.25, 0.1, 0.12), Vec3(-0.3, 0.05, 0.03)};
    compareWithHippo(s, false);
    System system;
    createSystem(system, s, createInteraxForce(s));
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(s.positions);
    if (getPlatformPrecision(platform, true) == "double")
        checkFiniteDifferenceForces(context, 1e-6);
    checkForceEnergyConsistency(context, consistencyTol, consistencyStep);
}

/**
 * Test exponents that are nearly but not exactly equal, where HIPPO's formulas for unequal exponents lose precision
 * to cancellation. The energy must be a smooth function of the difference d of the exponents: (E(d) - E(0))/d is
 * the same for d from 1e-4 to 1e-7 (relative to the exponent) to within the first order change in d.
 */
void testNearlyEqualExponents() {
    MultipoleSystem s = createWaterDimer();
    for (MultipoleParticle& p : s.particles)
        p.alpha = 45.0;
    auto energy = [&] (double difference) {
        MultipoleSystem nearly = s;
        for (int i = 0; i < nearly.particles.size(); i++)
            nearly.particles[i].alpha = 45.0*(1+difference*(i%3));
        System system;
        createSystem(system, nearly, createInteraxForce(nearly));
        return evaluate(system, s.positions, platform).energy;
    };
    double equal = energy(0.0);
    if (getPlatformPrecision(platform, true) != "double") {
        ASSERT_EQUAL_TOL(equal, energy(1e-7), energyTol);
        return;
    }
    double slope = (energy(1e-4)-equal)/1e-4;
    ASSERT(fabs(slope) > 1e-3*fabs(equal));
    for (double difference : {1e-5, 1e-6, 1e-7})
        ASSERT_EQUAL_TOL(slope, (energy(difference)-equal)/difference, 1e-2);
}

/**
 * Test the alchemical scaling: with the alchemical parameter at lambda, the force is that with the multipoles and
 * polarizabilities of the alchemical particles multiplied by lambda.
 */
void testAlchemical() {
    for (bool periodic : {false, true}) {
        MultipoleSystem s = (periodic ? createWaterBox(3, 1.25, 2) : createWaterTrimer());
        InteraxOptions options;
        options.coreDamping = true;
        options.coreDampingScale = 0.8;
        options.alchemicalParameter = "lambda_test";
        options.isAlchemical.resize(s.particles.size(), false);
        for (int i = 0; i < 3; i++)
            options.isAlchemical[i] = true;
        System system;
        createSystem(system, s, createInteraxForce(s, options));
        {
            VerletIntegrator integrator(0.001);
            Context context(system, integrator, platform);
            ASSERT_EQUAL(1.0, context.getParameter("lambda_test"));
        }
        for (double lambda : {1.0, 0.3, 0.0}) {
            MultipoleSystem scaled = s;
            for (int i = 0; i < 3; i++) {
                MultipoleParticle& p = scaled.particles[i];
                p.charge *= lambda;
                p.coreCharge *= lambda;
                p.polarizability *= lambda;
                for (double& d : p.dipole)
                    d *= lambda;
                for (double& d : p.quadrupole)
                    d *= lambda;
            }
            InteraxOptions scaledOptions = options;
            scaledOptions.isAlchemical.clear();
            System scaledSystem;
            createSystem(scaledSystem, scaled, createInteraxForce(scaled, scaledOptions));
            MultipoleResults expected = evaluate(scaledSystem, s.positions, platform);
            MultipoleResults found = evaluate(system, s.positions, platform, {{"lambda_test", lambda}});
            compareResults(expected, found, energyTol);
        }
    }
}

/**
 * Test the own GPU kernel (setUseOwnKernel()) against the default one, alone and together with a NonbondedForce
 * that uses the default nonbonded kernel, with the same exclusions, in one context.
 */
void testOwnKernel() {
    for (bool periodic : {false, true}) {
        MultipoleSystem s = (periodic ? createWaterBox(3, 1.25, 3) : createWaterTrimer());
        InteraxOptions options;
        options.coreDamping = true;
        options.mutualDampingScale = 0.9;
        System defaultSystem, ownSystem;
        createSystem(defaultSystem, s, createInteraxForce(s, options));
        options.useOwnKernel = true;
        createSystem(ownSystem, s, createInteraxForce(s, options));
        MultipoleResults expected = evaluate(defaultSystem, s.positions, platform);
        compareResults(expected, evaluate(ownSystem, s.positions, platform), energyTol);

        // Add a Lennard-Jones NonbondedForce, whose exclusions are the exceptions of the multipole force.

        NonbondedForce* lj = new NonbondedForce();
        lj->setNonbondedMethod(periodic ? NonbondedForce::PME : NonbondedForce::NoCutoff);
        lj->setCutoffDistance(s.cutoff);
        lj->setUseDispersionCorrection(false);
        for (int i = 0; i < s.particles.size(); i++)
            lj->addParticle(0.0, 0.3, 0.2);
        for (const MultipoleException& e : s.exceptions)
            lj->addException(e.particle1, e.particle2, 0.0, 1.0, 0.0);
        lj->setForceGroup(1);
        ownSystem.addForce(lj);
        VerletIntegrator integrator(0.001);
        Context context(ownSystem, integrator, platform);
        context.setPositions(s.positions);
        State multipole = getEnergyAndForces(context, 1<<0);
        State both = getEnergyAndForces(context);
        System ljSystem;
        addParticles(ljSystem, s.particles.size());
        if (periodic)
            ljSystem.setDefaultPeriodicBoxVectors(Vec3(s.boxSize, 0, 0), Vec3(0, s.boxSize, 0), Vec3(0, 0, s.boxSize));
        ljSystem.addForce(new NonbondedForce(*lj));
        VerletIntegrator ljIntegrator(0.001);
        Context ljContext(ljSystem, ljIntegrator, platform);
        ljContext.setPositions(s.positions);
        State ljState = getEnergyAndForces(ljContext);
        ASSERT_EQUAL_TOL(expected.energy, multipole.getPotentialEnergy(), energyTol);
        assertForcesEqual(expected.forces, multipole.getForces(), 10*energyTol);
        ASSERT_EQUAL_TOL(expected.energy+ljState.getPotentialEnergy(), both.getPotentialEnergy(), energyTol);
    }
}

/**
 * Test updateParametersInContext() (as OpenMM's TestHippoNonbondedForce.h does).
 */
void testChangingParameters() {
    MultipoleSystem s = createWaterDimer();
    System system;
    InteraxOptions options;
    options.coreDamping = true;
    InteraxMultipoleForce* force = createInteraxForce(s, options);
    createSystem(system, s, force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(s.positions);
    State state1 = getEnergyAndForces(context);

    // Change some of the parameters and make sure the results change.

    force->setParticleParameters(0, -0.2, {0.0, 0.0, 0.005}, {0.001, 0.0, 0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 0.0}, 6.0, 20.0, 0.0009,
            InteraxMultipoleForce::Bisector, 1, 2, -1);
    force->setExceptionParameters(1, 0, 2, 0.1, 0.1, 0.3);
    force->updateParametersInContext(context);
    State state2 = getEnergyAndForces(context);
    ASSERT(fabs(state1.getPotentialEnergy()-state2.getPotentialEnergy()) > 1.0)

    // Create a new context with the altered parameters and see if it agrees.

    VerletIntegrator integrator2(0.001);
    Context context2(system, integrator2, platform);
    context2.setPositions(s.positions);
    State state3 = getEnergyAndForces(context2);
    ASSERT_EQUAL_TOL(state2.getPotentialEnergy(), state3.getPotentialEnergy(), energyTol);
    assertForcesEqual(state3.getForces(), state2.getForces(), 10*energyTol);
}

/**
 * A molecule with a chiral centre (particle 0, a ZThenX frame with a y atom and four distinct neighbours) whose
 * dipole and quadrupole have y components, near a water molecule, and the mirror image of the pair (x -> -x).
 */
MultipoleSystem createChiralPair(bool mirror) {
    MultipoleSystem s;
    double qs = 1e-4;
    s.particles.push_back({0.1, {0.003, 0.004, 0.005}, {qs, 2*qs, -qs, 2*qs, 0.5*qs, 1.5*qs, -qs, 1.5*qs, -1.5*qs}, 4.0, 30.0, 0.0012,
                           InteraxMultipoleForce::ZThenX, 1, 2, 3});
    double charges[4] = {-0.25, 0.05, 0.08, 0.02};
    for (int k = 0; k < 4; k++)
        s.particles.push_back({charges[k], {0.0, 0.0, 0.002*(k+1)}, vector<double>(9, 0.0), 1.0+k, 35.0+3*k, 0.0005+0.0002*k,
                               InteraxMultipoleForce::ZOnly, 0, -1, -1});
    for (int i = 0; i < 5; i++)
        for (int j = i+1; j < 5; j++)
            s.exceptions.push_back({i, j, 0.0, 0.0, 0.0});
    s.positions = {Vec3(0, 0, 0), Vec3(0.14, 0.01, -0.02), Vec3(-0.05, 0.10, 0.03), Vec3(-0.04, -0.06, 0.09), Vec3(-0.03, -0.05, -0.10)};
    addWater(s, Vec3(0.30, 0.12, 0.05), Vec3(0.37, 0.16, 0.10), Vec3(0.33, 0.04, 0.02));
    if (mirror)
        for (Vec3& r : s.positions)
            r[0] = -r[0];
    return s;
}

/**
 * Test the chirality modes. With AmoebaChirality the multipoles of the chiral centre change their y components when
 * the frame is left-handed, so the energy of the mirror image is the same and its forces are the mirror images, on
 * every platform (HIPPO only applies the check on the GPU platforms); with NoChirality the y atom is ignored.
 */
void testChirality() {
    for (auto mode : {InteraxMultipoleForce::AmoebaChirality, InteraxMultipoleForce::NoChirality}) {
        InteraxOptions options;
        options.chiralityMode = mode;
        MultipoleResults results[2];
        for (int mirror = 0; mirror < 2; mirror++) {
            MultipoleSystem s = createChiralPair(mirror);
            System system;
            createSystem(system, s, createInteraxForce(s, options));
            results[mirror] = evaluate(system, s.positions, platform);
            if (platform.getName() != "Reference")
                compareResults(evaluate(system, s.positions, Platform::getPlatformByName("Reference")), results[mirror], energyTol);
            VerletIntegrator integrator(0.001);
            Context context(system, integrator, platform);
            context.setPositions(s.positions);
            if (getPlatformPrecision(platform, true) == "double")
                checkFiniteDifferenceForces(context, 1e-6);
            checkForceEnergyConsistency(context, consistencyTol, consistencyStep);
            if (mode == InteraxMultipoleForce::NoChirality) {
                // The same as without a y atom
                MultipoleSystem noY = s;
                noY.particles[0].atomY = -1;
                System system2;
                createSystem(system2, noY, createInteraxForce(noY, options));
                compareResults(evaluate(system2, s.positions, platform), results[mirror], energyTol);
            }
        }
        if (mode == InteraxMultipoleForce::AmoebaChirality) {
            ASSERT_EQUAL_TOL(results[0].energy, results[1].energy, energyTol);
            for (int i = 0; i < results[0].forces.size(); i++) {
                Vec3 f = results[0].forces[i];
                ASSERT_EQUAL_VEC(Vec3(-f[0], f[1], f[2]), results[1].forces[i], 10*energyTol);
            }
        }
        else
            ASSERT(fabs(results[0].energy-results[1].energy) > 10*energyTol*fabs(results[0].energy));  // 2% here
    }
}

/**
 * Test a particle at the same position as another one, fully excluded from it (all scale factors zero), as
 * InteraxFF's bond multipole sites sit on their host atom (non-periodic only): moving the host's multipoles onto the
 * coincident particle must change nothing, and the force on the pair must equal the host's force.
 */
void testCoincidentParticles() {
    MultipoleSystem withHost = createWaterDimer(), withSite = createWaterDimer();
    // particle 1 (an H) carries its multipoles itself, or gives them to a new particle 6 at its position
    MultipoleParticle site = withSite.particles[1];
    site.charge = 0.0;
    site.coreCharge = 0.0;
    site.polarizability = 0.0;
    withSite.particles[1].dipole = {0.0, 0.0, 0.0};
    withSite.particles[1].quadrupole = vector<double>(9, 0.0);
    site.atomY = -1;
    withSite.particles.push_back(site);
    withSite.positions.push_back(withSite.positions[1]);
    for (int i = 0; i < 3; i++)
        withSite.exceptions.push_back({i, 6, 0.0, 0.0, 0.0});
    withHost.particles[1].atomY = -1;
    System hostSystem, siteSystem;
    createSystem(hostSystem, withHost, createInteraxForce(withHost));
    createSystem(siteSystem, withSite, createInteraxForce(withSite));
    MultipoleResults expected = evaluate(hostSystem, withHost.positions, platform);
    MultipoleResults found = evaluate(siteSystem, withSite.positions, platform);
    ASSERT(isfinite(found.energy));
    ASSERT_EQUAL_TOL(expected.energy, found.energy, energyTol);
    for (int i = 0; i < 6; i++) {
        Vec3 f = found.forces[i];
        if (i == 1)
            f += found.forces[6];
        ASSERT_EQUAL_VEC(expected.forces[i], f, 10*energyTol);
    }
}

/**
 * Test, on a platform other than Reference, the InteraxFF options in a periodic box against the Reference platform.
 */
void testAgainstReference() {
    if (platform.getName() == "Reference")
        return;
    MultipoleSystem s = createWaterBox(4, 1.3, 4);
    for (bool useOwnKernel : {false, true}) {
        InteraxOptions options;
        options.coreDamping = true;
        options.coreDampingScale = 0.7;
        options.mutualDampingScale = 0.8;
        options.useOwnKernel = useOwnKernel;
        options.isAlchemical.resize(s.particles.size(), false);
        for (int i = 0; i < 6; i++)
            options.isAlchemical[i] = true;
        System system;
        createSystem(system, s, createInteraxForce(s, options));
        map<string, double> parameters = {{"lambda_electrostatics", 0.4}};
        MultipoleResults expected = evaluate(system, s.positions, Platform::getPlatformByName("Reference"), parameters);
        compareResults(expected, evaluate(system, s.positions, platform, parameters), energyTol);
    }
}

/**
 * Test an alchemical molecule at lambda = 0 overlapping another molecule, as a decoupled solute lets water through it:
 * its pairs must stay exactly zero, although the inverse powers of r overflow in single precision below ~0.001 nm.
 */
void testAlchemicalOverlap() {
    for (bool periodic : {false, true}) {
        MultipoleSystem s = (periodic ? createWaterBox(3, 1.25, 2) : createWaterTrimer());
        InteraxOptions options;
        options.coreDamping = true;
        options.coreDampingScale = 0.7;
        options.isAlchemical.resize(s.particles.size(), false);
        for (int i = 0; i < 3; i++)
            options.isAlchemical[i] = true;
        System system;
        createSystem(system, s, createInteraxForce(s, options));

        // Move the first water so that its oxygen is 0.0005 nm from the oxygen of the second.

        vector<Vec3> positions = s.positions;
        Vec3 shift = positions[3]+Vec3(5e-4, 0, 0)-positions[0];
        for (int i = 0; i < 3; i++)
            positions[i] += shift;
        map<string, double> parameters = {{"lambda_electrostatics", 0.0}};
        MultipoleResults found = evaluate(system, positions, platform, parameters);
        for (int i = 0; i < found.forces.size(); i++)
            for (int j = 0; j < 3; j++)
                ASSERT(isfinite(found.forces[i][j]));
        for (int i = 0; i < 3; i++)
            ASSERT_EQUAL_VEC(Vec3(), found.forces[i], 0.0);

        // The decoupled molecule does not interact: moving it must change nothing (compared on the same platform,
        // as testAlchemical() does, since single precision differs from Reference by ~3e-4 for this system).

        MultipoleResults apart = evaluate(system, s.positions, platform, parameters);
        ASSERT_EQUAL_TOL(apart.energy, found.energy, energyTol);
        for (int i = 3; i < found.forces.size(); i++)
            ASSERT_EQUAL_VEC(apart.forces[i], found.forces[i], 10*energyTol);
        if (platform.getName() != "Reference" && getPlatformPrecision(platform, true) == "double")
            compareResults(evaluate(system, positions, Platform::getPlatformByName("Reference"), parameters), found, energyTol);
    }
}

/**
 * Test that invalid parameters are rejected.
 */
void testIllegalSettings() {
    auto check = [&] (MultipoleSystem s, const string& description) {
        System system;
        createSystem(system, s, createInteraxForce(s));
        assertContextCreationFails(system, platform, description);
    };
    MultipoleSystem s = createWaterDimer();
    MultipoleSystem bad = s;
    bad.particles[1].quadrupole[0] += 0.001;
    check(bad, "a quadrupole that is not traceless");
    bad = s;
    bad.particles[1].quadrupole[1] = 0.001;
    check(bad, "a quadrupole that is not symmetric");
    bad = s;
    bad.particles[2].alpha = 0.0;
    check(bad, "a zero exponent");
    bad = s;
    bad.particles[2].polarizability = -0.001;
    check(bad, "a negative polarizability");
    bad = s;
    bad.particles[2].atomX = 6;
    check(bad, "an invalid frame particle");
    bad = createWaterBox(3, 1.25, 0);
    bad.cutoff = 0.7;
    check(bad, "a cutoff larger than half the box");
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        setForceResolution(platform);
        if (getPlatformPrecision(platform, true) == "double") {
            energyTol = 1e-7;
            dipoleTol = 1e-9;
            forceTol = 1e-6;
            consistencyTol = 1e-5;
            consistencyStep = 1e-4;
        }
        else {
            // As in OpenMM's TestHippoNonbondedForce.h, the energy in single precision is only consistent with the
            // forces to a few parts per thousand over a small step.
            energyTol = 3e-4;
            dipoleTol = 1e-6;
            forceTol = 1e-3;
            consistencyTol = 5e-2;
            consistencyStep = 1e-3;
        }
        testWaterDimer();
        testExceptions();
        testWaterBox();
        testMutualDampingScale();
        testCoreDamping();
        testCoreDampingPme();
        testInteraxOptionsForces();
        testZOnlyFrame();
        testNearlyEqualExponents();
        testAlchemical();
        testOwnKernel();
        testChangingParameters();
        testAgainstReference();
        testChirality();
        testCoincidentParticles();
        testAlchemicalOverlap();
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
