#ifndef OPENMM_INTERAXDISPERSIONFORCE_H_
#define OPENMM_INTERAXDISPERSIONFORCE_H_

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
 * This class implements the InteraxFF pair dispersion: the exchange-hole dipole moment (XDM) C6, C8 and C10
 * coefficients with Tang-Toennies damping,
 *
 *   E = -sum_{pairs} sum_{n = 6, 8, 10} f_n(x) C_n / r^n,   x = beta * 2 r / (s_i + s_j),
 *   f_n(x) = 1 - exp(-x) sum_{k=0}^{n} x^k / k!,
 *
 * with the coefficients combined from the per-particle XDM moments m1, m2, m3 (<M_k^2>) and dispersion
 * polarizabilities alpha:
 *
 *   C6  = alpha_i alpha_j m1_i m1_j / D,                                     D = alpha_i m1_j + alpha_j m1_i
 *   C8  = 3/2 alpha_i alpha_j (m1_i m2_j + m2_i m1_j) / D
 *   C10 = [2 alpha_i alpha_j (m1_i m3_j + m3_i m1_j) + 21/5 alpha_i alpha_j m2_i m2_j] / D
 *
 * s is the Slater width of each particle and beta a global damping scale. A pair in which either particle has a
 * zero polarizability does not interact.
 *
 * Under periodic boundary conditions a long range correction may be added, computed with the same convention as
 * CustomNonbondedForce (see setUseLongRangeCorrection()).
 */

class OPENMM_EXPORT_INTERAX InteraxDispersionForce : public InteraxPairForce {
public:
    /**
     * Create an InteraxDispersionForce.
     */
    InteraxDispersionForce();
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
     * Add the dispersion parameters for a particle. This should be called once for each particle in the System.
     * When it is called for the i'th time, it specifies the parameters for the i'th particle.
     *
     * @param m1            the XDM moment <M_1^2>, measured in kJ/mol*nm^3
     * @param m2            the XDM moment <M_2^2>, measured in kJ/mol*nm^5
     * @param m3            the XDM moment <M_3^2>, measured in kJ/mol*nm^7
     * @param alpha         the dispersion polarizability, measured in nm^3
     * @param width         the Slater width that sets the damping length, measured in nm
     * @param isAlchemical  whether the particle is alchemical
     * @return the index of the particle that was added
     */
    int addParticle(double m1, double m2, double m3, double alpha, double width, bool isAlchemical=false);
    /**
     * Get the dispersion parameters for a particle.
     *
     * @param index              the index of the particle for which to get parameters
     * @param[out] m1            the XDM moment <M_1^2>, measured in kJ/mol*nm^3
     * @param[out] m2            the XDM moment <M_2^2>, measured in kJ/mol*nm^5
     * @param[out] m3            the XDM moment <M_3^2>, measured in kJ/mol*nm^7
     * @param[out] alpha         the dispersion polarizability, measured in nm^3
     * @param[out] width         the Slater width that sets the damping length, measured in nm
     * @param[out] isAlchemical  whether the particle is alchemical
     */
    void getParticleParameters(int index, double& m1, double& m2, double& m3, double& alpha, double& width, bool& isAlchemical) const;
    /**
     * Set the dispersion parameters for a particle.
     *
     * @param index         the index of the particle for which to set parameters
     * @param m1            the XDM moment <M_1^2>, measured in kJ/mol*nm^3
     * @param m2            the XDM moment <M_2^2>, measured in kJ/mol*nm^5
     * @param m3            the XDM moment <M_3^2>, measured in kJ/mol*nm^7
     * @param alpha         the dispersion polarizability, measured in nm^3
     * @param width         the Slater width that sets the damping length, measured in nm
     * @param isAlchemical  whether the particle is alchemical
     */
    void setParticleParameters(int index, double m1, double m2, double m3, double alpha, double width, bool isAlchemical);
    /**
     * Get the scale beta of the Tang-Toennies damping argument (dimensionless).
     */
    double getDampingScale() const;
    /**
     * Set the scale beta of the Tang-Toennies damping argument (dimensionless).
     */
    void setDampingScale(double scale);
    /**
     * Get whether to add a long range correction under periodic boundary conditions.
     */
    bool getUseLongRangeCorrection() const;
    /**
     * Set whether to add a long range correction under periodic boundary conditions. The correction is
     * E = 2 pi N^2 <I> / V, with <I> the average over all N(N+1)/2 particle pairs (self pairs included) of the
     * integral of r^2 E(r) beyond the cutoff (and of r^2 (1 - S(r)) E(r) over the switching interval), exactly as
     * for CustomNonbondedForce.
     */
    void setUseLongRangeCorrection(bool use);
    /**
     * Update the per-particle parameters and the damping scale in a Context to match those stored in this Force
     * object. This method provides an efficient method to update certain parameters in an existing Context without
     * needing to reinitialize it. Simply call setParticleParameters() to modify this object's parameters, then
     * call updateParametersInContext() to copy them over to the Context.
     *
     * The alchemical flags, the exclusions and the cutoff cannot be changed by this method.
     */
    void updateParametersInContext(OpenMM::Context& context);
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    double dampingScale;
    bool useLongRangeCorrection;
    std::vector<ParticleInfo> particles;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class InteraxDispersionForce::ParticleInfo {
public:
    double m1, m2, m3, alpha, width;
    bool isAlchemical;
    ParticleInfo() : m1(0.0), m2(0.0), m3(0.0), alpha(0.0), width(0.0), isAlchemical(false) {
    }
    ParticleInfo(double m1, double m2, double m3, double alpha, double width, bool isAlchemical) :
        m1(m1), m2(m2), m3(m3), alpha(alpha), width(width), isAlchemical(isAlchemical) {
    }
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXDISPERSIONFORCE_H_*/
