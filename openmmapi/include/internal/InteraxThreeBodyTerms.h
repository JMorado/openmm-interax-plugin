#ifndef OPENMM_INTERAXTHREEBODYTERMS_H_
#define OPENMM_INTERAXTHREEBODYTERMS_H_

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
 * The energy of one Axilrod-Teller-Muto triple of InteraxThreeBodyForce and its gradients, shared by the Reference
 * and CPU platforms.  See docs/SPEC.md.
 *
 * @private
 */

#include "openmm/Vec3.h"
#include <cmath>

namespace InteraxPlugin {

namespace InteraxThreeBodyTerms {

/**
 * Compute the Tang-Toennies damping function f_3(x) = 1 - exp(-x) sum_{k<=3} x^k/k! of a dipole propagator and its
 * derivative f_3'(x) = exp(-x) x^3/3!. For x < 2 the closed form cancels, and f_3 = exp(-x) sum_{k>3} x^k/k! instead.
 */
inline void tangToenniesDamping3(double x, double& damping, double& dDampingdX) {
    double expX = std::exp(-x);
    double term3 = x*x*x/6;
    dDampingdX = expX*term3;
    if (x < 2) {
        double term = term3, tail = 0;
        for (int k = 4; k <= 30; k++) {
            term *= x/k;
            tail += term;
        }
        damping = expX*tail;
    }
    else
        damping = 1-expX*(1+x+x*x/2+term3);
}

/**
 * Compute the centroid damping function g(r) = 1 - exp(-(r/R0)^6) of a pair of sites and its derivative
 * g'(r) = 6 (r/R0)^6 exp(-(r/R0)^6) / r (InteraxThreeBodyForce::setCentroidDampingRadius()).
 */
inline void centroidDamping(double r, double invRadius, double& damping, double& dDampingdR) {
    double x = r*invRadius;
    double x6 = x*x*x;
    x6 *= x6;
    damping = -std::expm1(-x6);
    dDampingdR = 6*x6*std::exp(-x6)/r;
}

/**
 * Compute the energy of a triple of sites A, B, C from the minimum image vectors a = AB, b = AC and c = BC, and add
 * the forces on the three sites. The angle at each vertex is the angle between the vectors to its two neighbors,
 * so with u, v, w the squared lengths, P = uvw and p = a.b, q = a.c, s = b.c, the product of the three cosines is
 * -pqs/P and the undamped energy is E0 = C9 [P^(-3/2) - 3 pqs P^(-5/2)]. With damping, E = E0 f(|a|) f(|b|) f(|c|)
 * with the Tang-Toennies f_3 of argument x = rate*r for each pair.
 *
 * With a centroid damping radius R0 > 0, E is also multiplied by g(|a|) g(|b|) g(|c|), g(r) = 1 - exp(-(r/R0)^6).
 *
 * @param a, b, c          the minimum image vectors AB, AC and BC
 * @param c9               the C9 coefficient of the triple, including the scale factor and the alchemical scaling
 * @param dampingRates     the rates (dx/dr) of the pairs AB, AC and BC, or NULL for no Tang-Toennies damping
 * @param[in,out] forceA   the force on site A, to which the force of this triple is added
 * @param[in,out] forceB   the force on site B, to which the force of this triple is added
 * @param[in,out] forceC   the force on site C, to which the force of this triple is added
 * @param dampingRadius    the centroid damping radius R0, or 0 for no centroid damping
 * @return the energy of the triple
 */
inline double computeTriple(const OpenMM::Vec3& a, const OpenMM::Vec3& b, const OpenMM::Vec3& c, double c9, const double* dampingRates,
        OpenMM::Vec3& forceA, OpenMM::Vec3& forceB, OpenMM::Vec3& forceC, double dampingRadius=0.0) {
    double u = a.dot(a), v = b.dot(b), w = c.dot(c);
    double p = a.dot(b), q = a.dot(c), s = b.dot(c);
    double invP = 1/(u*v*w);
    double invP32 = invP*std::sqrt(invP);
    double invP52 = invP32*invP;
    double pqs = p*q*s;
    double energy = c9*(invP32 - 3*pqs*invP52);

    // Gradients with respect to a, b and c.

    double radial = c9*(-3*invP32 + 15*pqs*invP52);
    double cross = -3*c9*invP52;
    OpenMM::Vec3 dEda = a*(radial/u) + (b*(q*s) + c*(p*s))*cross;
    OpenMM::Vec3 dEdb = b*(radial/v) + (a*(q*s) + c*(p*q))*cross;
    OpenMM::Vec3 dEdc = c*(radial/w) + (a*(p*s) + b*(p*q))*cross;
    if (dampingRates != NULL || dampingRadius > 0) {
        // E = E0 fa fb fc, so dE/da = fa fb fc dE0/da + E0 fb fc dfa/da, and similarly for b and c. Each f is the
        // Tang-Toennies damping, the centroid damping, or their product.

        double lengths[3] = {std::sqrt(u), std::sqrt(v), std::sqrt(w)};
        double f[3], dfdr[3];
        for (int i = 0; i < 3; i++) {
            if (dampingRates != NULL) {
                tangToenniesDamping3(dampingRates[i]*lengths[i], f[i], dfdr[i]);
                dfdr[i] *= dampingRates[i];
            }
            else {
                f[i] = 1;
                dfdr[i] = 0;
            }
            if (dampingRadius > 0) {
                double g, dgdr;
                centroidDamping(lengths[i], 1/dampingRadius, g, dgdr);
                dfdr[i] = dfdr[i]*g + f[i]*dgdr;
                f[i] *= g;
            }
        }
        double damping = f[0]*f[1]*f[2];
        dEda = dEda*damping + a*(energy*dfdr[0]*f[1]*f[2]/lengths[0]);
        dEdb = dEdb*damping + b*(energy*f[0]*dfdr[1]*f[2]/lengths[1]);
        dEdc = dEdc*damping + c*(energy*f[0]*f[1]*dfdr[2]/lengths[2]);
        energy *= damping;
    }
    forceA += dEda + dEdb;
    forceB += dEdc - dEda;
    forceC -= dEdb + dEdc;
    return energy;
}

} // namespace InteraxThreeBodyTerms

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXTHREEBODYTERMS_H_*/
