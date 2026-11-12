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
 * Utilities shared by the tests of the InteraxFF forces: tolerances that depend on the precision of the platform,
 * comparisons of energies and forces, finite difference checks, and independent implementations (in long double)
 * of the functions the forces are built from, written from the formulas in docs/SPEC.md.
 *
 * A platform test first includes its platform header (e.g. ReferenceInteraxTests.h, which defines the global
 * "platform" and setupKernels()) and then the test header of the force.
 */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "OpenMMInterax.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include "sfmt/SFMT.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

/**
 * Tests specific to the platform, defined by each platform test after it includes the test header.
 */
void runPlatformTests();

/**
 * The type in which the tests compute their expected values.
 */
typedef long double Real;

const Real PI_L = 3.141592653589793238462643383279502884L;

/**
 * Assert that two values agree to a tolerance relative to the magnitude of the expected value (unlike
 * ASSERT_EQUAL_TOL, which compares values smaller than 1 absolutely).
 */
#define ASSERT_EQUAL_REL(expected, found, tol) {double _expected_ = (expected), _found_ = (found); \
    if (!(std::abs(_expected_-_found_) <= (tol)*std::abs(_expected_))) {std::stringstream details; details.precision(17); \
    details << "Expected "<<_expected_<<", found "<<_found_; throwException(__FILE__, __LINE__, details.str());}};

/**
 * Get the precision in which a platform computes a force: "double" for the Reference platform, the "Precision"
 * property for the GPU platforms, and for the CPU platform "single" or "double" depending on the force (its pair
 * forces are computed in single precision, the three-body and multipole forces in double precision).
 */
static string getPlatformPrecision(Platform& platform, bool cpuUsesDoublePrecision) {
    if (platform.getName() == "Reference")
        return "double";
    if (platform.getName() == "CPU")
        return (cpuUsesDoublePrecision ? "double" : "single");
    try {
        return platform.getPropertyDefaultValue("Precision");
    }
    catch (...) {
        // This platform doesn't have a "Precision" property.
    }
    return "single";
}

/**
 * The resolution of the forces of the platform being tested (kJ/mol/nm), below which forces are not compared. The
 * GPU platforms accumulate forces in 64 bit fixed point with a resolution of 2^-32, so a sum of a few contributions
 * is exact only to ~1e-9 whatever the magnitude of the forces. It is set by setForceResolution().
 */
double forceResolution = 0.0;

/**
 * Set the force resolution of a platform: zero for the Reference and CPU platforms, 1e-8 for the others.
 */
static void setForceResolution(Platform& platform) {
    forceResolution = (platform.getName() == "Reference" || platform.getName() == "CPU" ? 0.0 : 1e-8);
}

/**
 * Assert that two sets of forces agree, component by component, to a tolerance relative to the largest expected
 * force (relative errors in single precision are of the order of the largest force, not of each force), plus the
 * force resolution of the platform.
 */
static void assertForcesEqual(const vector<Vec3>& expected, const vector<Vec3>& found, double tol) {
    ASSERT_EQUAL(expected.size(), found.size());
    double scale = 0;
    for (const Vec3& f : expected)
        scale = max(scale, sqrt(f.dot(f)));
    for (int i = 0; i < expected.size(); i++)
        for (int k = 0; k < 3; k++)
            if (!(fabs(expected[i][k]-found[i][k]) <= tol*scale+forceResolution)) {
                stringstream details;
                details.precision(12);
                details << "Force on particle " << i << ": expected " << expected[i] << ", found " << found[i] << " (largest force " << scale << ")";
                throwException(__FILE__, __LINE__, details.str());
            }
}

/**
 * Check that the forces are consistent with the energy: take a small step along the forces and check that the
 * energy changes by the expected amount (as in OpenMM's tests).
 */
static void checkForceEnergyConsistency(Context& context, double tol, double delta=1e-3) {
    State state = context.getState(State::Positions | State::Forces | State::Energy);
    int numParticles = context.getSystem().getNumParticles();
    double norm = 0.0;
    for (int i = 0; i < numParticles; ++i) {
        Vec3 f = state.getForces()[i];
        norm += f.dot(f);
    }
    norm = sqrt(norm);
    double step = 0.5*delta/norm;
    vector<Vec3> positions2(numParticles), positions3(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        Vec3 p = state.getPositions()[i];
        Vec3 f = state.getForces()[i];
        positions2[i] = p-f*step;
        positions3[i] = p+f*step;
    }
    context.setPositions(positions2);
    State state2 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state3 = context.getState(State::Energy);
    context.setPositions(state.getPositions());
    ASSERT_EQUAL_TOL(state2.getPotentialEnergy()-state3.getPotentialEnergy(), norm*delta, tol)
}

/**
 * Check every force component against a central finite difference of the energy, to a tolerance relative to the
 * largest force. This is meant for small systems on double precision platforms.
 */
static void checkFiniteDifferenceForces(Context& context, double tol, double delta=1e-5) {
    State state = context.getState(State::Positions | State::Forces);
    vector<Vec3> positions = state.getPositions();
    vector<Vec3> finiteDifference(positions.size());
    for (int i = 0; i < positions.size(); i++)
        for (int k = 0; k < 3; k++) {
            vector<Vec3> displaced = positions;
            displaced[i][k] = positions[i][k]+delta;
            context.setPositions(displaced);
            double energyPlus = context.getState(State::Energy).getPotentialEnergy();
            displaced[i][k] = positions[i][k]-delta;
            context.setPositions(displaced);
            double energyMinus = context.getState(State::Energy).getPotentialEnergy();
            finiteDifference[i][k] = (energyMinus-energyPlus)/(2*delta);
        }
    context.setPositions(positions);
    assertForcesEqual(finiteDifference, state.getForces(), tol);
}

/**
 * Get the energy and forces of a Context.
 */
static State getEnergyAndForces(Context& context, int groups=0xFFFFFFFF) {
    return context.getState(State::Energy | State::Forces, false, groups);
}

/**
 * Compute the derivative of a function of one variable with the five-point central difference.
 */
template <class Function>
Real fivePointDerivative(const Function& function, Real x, Real h) {
    return (function(x-2*h) - 8*function(x-h) + 8*function(x+h) - function(x+2*h))/(12*h);
}

/**
 * Get the minimum image of a displacement in a periodic box (triclinic boxes in OpenMM's reduced form), by
 * searching the neighboring images: this does not rely on the box being large compared to the displacement.
 */
static Vec3 getMinimumImage(Vec3 delta, const Vec3* box) {
    delta -= box[2]*round(delta[2]/box[2][2]);
    delta -= box[1]*round(delta[1]/box[1][1]);
    delta -= box[0]*round(delta[0]/box[0][0]);
    Vec3 best = delta;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++) {
                Vec3 image = delta + box[0]*i + box[1]*j + box[2]*k;
                if (image.dot(image) < best.dot(best))
                    best = image;
            }
    return best;
}

/**
 * The switching function S(r) = 1 - 10x^3 + 15x^4 - 6x^5, x = (r - r_switch)/(r_cut - r_switch), of the pair forces.
 */
static Real switchingFunction(Real r, Real switchingDistance, Real cutoff) {
    if (r <= switchingDistance)
        return 1;
    if (r >= cutoff)
        return 0;
    Real x = (r-switchingDistance)/(cutoff-switchingDistance);
    return 1 - 10*x*x*x + 15*x*x*x*x - 6*x*x*x*x*x;
}

/**
 * The closed form of the overlap integral S(r; a, b) of the normalized Slater densities exp(-r/s)/(8 pi s^3) of widths
 * a and b (docs/SPEC.md, MEDFF SI Eq. 32), for complex widths.
 */
static complex<Real> slaterOverlapClosedForm(Real r, complex<Real> a, complex<Real> b) {
    complex<Real> d = b*b-a*a;
    complex<Real> expA = exp(-r/a), expB = exp(-r/b);
    complex<Real> f = (Real) 4*a*a*b*b/(d*d*d)*(expA-expB) + r/(d*d)*(a*expA+b*expB);
    return f/(8*PI_L*r);
}

/**
 * The overlap integral S(r; a, b) of the normalized Slater densities exp(-r/s)/(8 pi s^3) of widths a and b. For
 * equal widths it is the limit exp(-x) (1 + x + x^2/3)/(64 pi a^3), x = r/a. Otherwise it is the closed form,
 * whose terms cancel to relative order t^3, t = (b - a)/a. For |t| < 0.05 it is instead computed from the closed
 * form on a circle |t| = 1/4 in the complex plane (where it does not cancel) with Cauchy's integral formula,
 * S(t) = 1/(2 pi) int S(z) z/(z - t) dtheta, z = exp(i theta)/4, which converges geometrically with the number of
 * points since S is analytic in t for |t| < 1.
 */
static Real slaterOverlap(Real r, Real a, Real b) {
    if (a == b) {
        Real x = r/a;
        return expl(-x)*(1 + x + x*x/3)/(64*PI_L*a*a*a);
    }
    Real t = (b-a)/a;
    if (fabsl(t) >= 0.05)
        return slaterOverlapClosedForm(r, a, b).real();
    const int numPoints = 64;
    const Real radius = 0.25;
    complex<Real> sum = 0;
    for (int k = 0; k < numPoints; k++) {
        complex<Real> z = polar(radius, 2*PI_L*(k+0.5L)/numPoints);
        sum += slaterOverlapClosedForm(r, a, a*((Real) 1+z))*z/(z-t);
    }
    return sum.real()/numPoints;
}

/**
 * The same overlap integral, computed numerically from the Fourier transforms of the densities, 1/(1 + k^2 s^2)^2:
 * S(r) = 1/(2 pi^2 r) int_0^inf k sin(kr) / ((1 + k^2 a^2)^2 (1 + k^2 b^2)^2) dk. This is independent of the
 * closed form, and accurate while S is not too small compared to the integrand (r less than ~10 widths).
 */
static Real slaterOverlapByQuadrature(Real r, Real a, Real b) {
    const int numIntervals = 400000;
    Real kMax = 400/min(a, b);
    Real h = kMax/numIntervals;
    Real sum = 0;
    for (int i = 0; i <= numIntervals; i++) {
        Real k = i*h;
        Real ka = 1+k*k*a*a, kb = 1+k*k*b*b;
        Real value = k*sinl(k*r)/(ka*ka*kb*kb);
        Real weight = (i == 0 || i == numIntervals ? 1 : (i%2 == 1 ? 4 : 2));
        sum += weight*value;
    }
    return sum*h/3/(2*PI_L*PI_L*r);
}

/**
 * The Tang-Toennies damping function f_n(x) = 1 - exp(-x) sum_{k=0}^{n} x^k/k!. Below x = n+1, where the closed form
 * cancels, it is evaluated as exp(-x) sum_{k>n} x^k/k!, whose terms then decrease.
 */
static Real tangToennies(int n, Real x) {
    Real term = 1, partialSum = 1;
    for (int k = 1; k <= n; k++) {
        term *= x/k;
        partialSum += term;
    }
    if (x >= n+1)
        return 1-expl(-x)*partialSum;
    Real sum = 0;
    for (int k = n+1; k < 1000; k++) {
        term *= x/k;
        sum += term;
        if (term < 1e-30L*sum)
            break;
    }
    return expl(-x)*sum;
}

/**
 * Generate a random number in [0, 1).
 */
static double random01(OpenMM_SFMT::SFMT& sfmt) {
    return OpenMM_SFMT::genrand_real2(sfmt);
}

/**
 * Create a System of particles of unit mass.
 */
static void addParticles(System& system, int numParticles) {
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
}

/**
 * Check that creating a Context for a System throws an exception.
 */
static void assertContextCreationFails(const System& system, Platform& platform, const string& description) {
    bool failed = false;
    try {
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
    }
    catch (const OpenMMException& e) {
        failed = true;
    }
    if (!failed)
        throwException(__FILE__, __LINE__, "No exception was thrown for "+description);
}
