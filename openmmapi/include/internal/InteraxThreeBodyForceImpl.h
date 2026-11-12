#ifndef OPENMM_INTERAXTHREEBODYFORCEIMPL_H_
#define OPENMM_INTERAXTHREEBODYFORCEIMPL_H_

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

#include "InteraxKernels.h"
#include "InteraxThreeBodyForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * This is the internal implementation of InteraxThreeBodyForce.
 */

class OPENMM_EXPORT_INTERAX InteraxThreeBodyForceImpl : public OpenMM::ForceImpl {
public:
    InteraxThreeBodyForceImpl(const InteraxThreeBodyForce& owner);
    void initialize(OpenMM::ContextImpl& context);
    const InteraxThreeBodyForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters();
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
    /**
     * Compute the C9 coefficient of a triple of sites from their polarizabilities and frequencies,
     * C9 = 3/2 a1 a2 a3 w1 w2 w3 (w1 + w2 + w3) / ((w1 + w2)(w2 + w3)(w3 + w1)).
     *
     * @param alpha1, alpha2, alpha3             the polarizabilities of the three sites, in nm^3
     * @param frequency1, frequency2, frequency3 the single-pole frequencies of the three sites, in kJ/mol
     * @return the C9 coefficient, in kJ/mol*nm^9
     */
    static double computeC9(double alpha1, double frequency1, double alpha2, double frequency2, double alpha3, double frequency3) {
        return 1.5*alpha1*alpha2*alpha3*frequency1*frequency2*frequency3*(frequency1+frequency2+frequency3)/
                ((frequency1+frequency2)*(frequency2+frequency3)*(frequency3+frequency1));
    }
private:
    const InteraxThreeBodyForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXTHREEBODYFORCEIMPL_H_*/
