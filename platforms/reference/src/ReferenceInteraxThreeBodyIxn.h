#ifndef REFERENCE_INTERAX_THREEBODY_IXN_H_
#define REFERENCE_INTERAX_THREEBODY_IXN_H_

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

#include "InteraxThreeBodyForce.h"
#include "openmm/Vec3.h"
#include <vector>

namespace InteraxPlugin {

/**
 * This class evaluates InteraxThreeBodyForce on the Reference platform.
 */
class ReferenceInteraxThreeBodyIxn {
public:
    /**
     * Create a ReferenceInteraxThreeBodyIxn for the sites, cutoff, damping and periodicity of a force.
     */
    explicit ReferenceInteraxThreeBodyIxn(const InteraxThreeBodyForce& force);
    /**
     * Copy the polarizabilities, frequencies and widths of the sites, the scale factor and the damping scale from a
     * force.
     */
    void setParameters(const InteraxThreeBodyForce& force);
    /**
     * Compute the interactions.
     *
     * @param positions      the particle positions
     * @param boxVectors     the periodic box vectors
     * @param lambda         the value of the alchemical parameter
     * @param forces         the forces are added to this
     * @return the energy
     */
    double calculateIxn(const std::vector<OpenMM::Vec3>& positions, const OpenMM::Vec3* boxVectors, double lambda, std::vector<OpenMM::Vec3>& forces) const;
private:
    OpenMM::Vec3 getDelta(const OpenMM::Vec3& from, const OpenMM::Vec3& to, const OpenMM::Vec3* boxVectors) const;
    std::vector<std::vector<int> > siteParticles;
    std::vector<int> molecule;
    std::vector<double> alpha, frequency, width;
    std::vector<bool> isAlchemical;
    double cutoff, scaleFactor, dampingScale, dampingRadius;
    bool periodic, useDamping;
};

} // namespace InteraxPlugin

#endif /*REFERENCE_INTERAX_THREEBODY_IXN_H_*/
