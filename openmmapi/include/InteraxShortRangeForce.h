#ifndef OPENMM_INTERAXSHORTRANGEFORCE_H_
#define OPENMM_INTERAXSHORTRANGEFORCE_H_

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

#include "InteraxPairForce.h"
#include "openmm/Context.h"
#include <vector>

namespace InteraxPlugin {

/**
 * This class implements the InteraxFF short-range correction, a polynomial in the normalized overlap of the Slater
 * valence densities of the two particles:
 *
 *   E = sum_{pairs} [ (K_i + K_j)/2 St + (L_i + L_j)/2 St^2 ],   St = 8 pi (s_i + s_j)^3 S(r; s_i, s_j)
 *
 * where S is the overlap integral of the normalized densities exp(-r/s) / (8 pi s^3), so that St = 1 at r = 0.
 * K and L are per-particle amplitudes of either sign and s the per-particle Slater width. A particle with zero
 * width has no density and does not interact.
 */

class OPENMM_EXPORT_INTERAX InteraxShortRangeForce : public InteraxPairForce {
public:
    /**
     * This is an enumeration of the methods for the Slater overlap integral of nearly equal widths, where the
     * closed form cancels and a Taylor expansion in the width difference is used instead.
     */
    enum OverlapMethod {
        /**
         * An order 6 expansion with a switch chosen for the precision of the platform: the integral is exact to
         * ~1e-6 relative in single precision and ~1e-10 in double precision. This is the default.
         */
        Exact = 0,
        /**
         * The order 2 scheme of InteraxFF (MEDFF), used where the widths differ by less than 0.01 bohr, evaluated
         * exactly as the InteraxFF export does (about the width of the particle with the lower index). It reproduces
         * the export, including its truncation errors; in single precision the closed form near the switch has
         * relative errors up to ~3e-2.
         */
        InteraxOrder2 = 2,
        /**
         * The order 6 scheme of InteraxFF, used where the widths differ by less than 20%, evaluated exactly as the
         * InteraxFF export does.
         */
        InteraxOrder6 = 6
    };
    /**
     * Create an InteraxShortRangeForce.
     */
    InteraxShortRangeForce();
    /**
     * Get the number of particles for which force field parameters are defined.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Get whether a particle is alchemical.
     *
     * @param index    the index of the particle
     */
    bool getParticleIsAlchemical(int index) const;
    /**
     * Add the short-range correction parameters for a particle. This should be called once for each particle in
     * the System. When it is called for the i'th time, it specifies the parameters for the i'th particle.
     *
     * @param k             the amplitude K of the linear term, measured in kJ/mol
     * @param l             the amplitude L of the quadratic term, measured in kJ/mol
     * @param width         the Slater width of the density, measured in nm
     * @param isAlchemical  whether the particle is alchemical
     * @return the index of the particle that was added
     */
    int addParticle(double k, double l, double width, bool isAlchemical=false);
    /**
     * Get the short-range correction parameters for a particle.
     *
     * @param index              the index of the particle for which to get parameters
     * @param[out] k             the amplitude K of the linear term, measured in kJ/mol
     * @param[out] l             the amplitude L of the quadratic term, measured in kJ/mol
     * @param[out] width         the Slater width of the density, measured in nm
     * @param[out] isAlchemical  whether the particle is alchemical
     */
    void getParticleParameters(int index, double& k, double& l, double& width, bool& isAlchemical) const;
    /**
     * Set the short-range correction parameters for a particle.
     *
     * @param index         the index of the particle for which to set parameters
     * @param k             the amplitude K of the linear term, measured in kJ/mol
     * @param l             the amplitude L of the quadratic term, measured in kJ/mol
     * @param width         the Slater width of the density, measured in nm
     * @param isAlchemical  whether the particle is alchemical
     */
    void setParticleParameters(int index, double k, double l, double width, bool isAlchemical);
    /**
     * Get the method used to evaluate the Slater overlap integral for nearly equal widths, where its closed form loses
     * precision to cancellation.
     */
    OverlapMethod getOverlapMethod() const;
    /**
     * Set the method used to evaluate the Slater overlap integral for nearly equal widths.
     */
    void setOverlapMethod(OverlapMethod method);
    /**
     * Update the per-particle parameters in a Context to match those stored in this Force object. This method
     * provides an efficient method to update certain parameters in an existing Context without needing to
     * reinitialize it. Simply call setParticleParameters() to modify this object's parameters, then call
     * updateParametersInContext() to copy them over to the Context.
     *
     * The alchemical flags, the exclusions, the cutoff and the overlap method cannot be changed by this method.
     */
    void updateParametersInContext(OpenMM::Context& context);
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    std::vector<ParticleInfo> particles;
    OverlapMethod overlapMethod;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class InteraxShortRangeForce::ParticleInfo {
public:
    double k, l, width;
    bool isAlchemical;
    ParticleInfo() : k(0.0), l(0.0), width(0.0), isAlchemical(false) {
    }
    ParticleInfo(double k, double l, double width, bool isAlchemical) : k(k), l(l), width(width), isAlchemical(isAlchemical) {
    }
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXSHORTRANGEFORCE_H_*/
