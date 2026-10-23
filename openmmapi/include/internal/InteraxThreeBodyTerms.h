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
 * and CPU platforms. See docs/SPEC.md.
 *
 * @private
 */

#include "openmm/Vec3.h"
#include <cmath>

namespace InteraxPlugin {

namespace InteraxThreeBodyTerms {

/**
 * Compute the energy of a triple of sites A, B, C from the minimum image vectors a = AB, b = AC and c = BC, and add
 * the forces on the three sites. The angle at each vertex is the angle between the vectors to its two neighbors,
 * so with u, v, w the squared lengths, P = uvw and p = a.b, q = a.c, s = b.c, the product of the three cosines is
 * -pqs/P and E = C9 [P^(-3/2) - 3 pqs P^(-5/2)].
 *
 * @return the energy of the triple
 */
inline double computeTriple(const OpenMM::Vec3& a, const OpenMM::Vec3& b, const OpenMM::Vec3& c, double c9,
        OpenMM::Vec3& forceA, OpenMM::Vec3& forceB, OpenMM::Vec3& forceC) {
    double u = a.dot(a), v = b.dot(b), w = c.dot(c);
    double p = a.dot(b), q = a.dot(c), s = b.dot(c);
    double invP = 1/(u*v*w);
    double invP32 = invP*std::sqrt(invP);
    double invP52 = invP32*invP;
    double pqs = p*q*s;

    // Gradients with respect to a, b and c.

    double radial = c9*(-3*invP32 + 15*pqs*invP52);
    double cross = -3*c9*invP52;
    OpenMM::Vec3 dEda = a*(radial/u) + (b*(q*s) + c*(p*s))*cross;
    OpenMM::Vec3 dEdb = b*(radial/v) + (a*(q*s) + c*(p*q))*cross;
    OpenMM::Vec3 dEdc = c*(radial/w) + (a*(p*s) + b*(p*q))*cross;
    forceA += dEda + dEdb;
    forceB += dEdc - dEda;
    forceC -= dEdb + dEdc;
    return c9*(invP32 - 3*pqs*invP52);
}

} // namespace InteraxThreeBodyTerms

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXTHREEBODYTERMS_H_*/
