#ifndef OPENMM_INTERAXMULTIPOLEFORCE_H_
#define OPENMM_INTERAXMULTIPOLEFORCE_H_

/* -------------------------------------------------------------------------- *
 *                            OpenMM InteraxFF Plugin                         *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM InteraxFF plugin, built on the OpenMM molecular *
 * simulation toolkit (https://openmm.org). It is derived from OpenMM's       *
 * HippoNonbondedForce (AMOEBA plugin).                                       *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Peter Eastman (HippoNonbondedForce), Joao Morado                  *
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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportInterax.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace InteraxPlugin {

/**
 * This class implements the electrostatics and polarization of the InteraxFF force field: the permanent multipoles
 * (charge split into a point core charge and a Slater valence density, dipole and quadrupole in a local frame) and
 * induced dipoles, with the charge penetration of HIPPO (Rackers et al.). It is OpenMM's HippoNonbondedForce
 * without the repulsion, dispersion and charge transfer terms, with these additions:
 *
 * <ul>
 * <li>Core damping (setCoreDamping()). The permanent field that polarizes an atom is multiplied by
 * f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r), the fraction of the Slater density of the polarized atom (exponent a,
 * the particle's alpha) within r. HIPPO's bare core fields make the polarization energy diverge as
 * -polarizability*q_core^2/(2 r^4) when atoms overlap; with core damping it stays finite.</li>
 * <li>Mutual damping scale (setMutualDampingScale()): the induced dipole couplings use the exponents kappa*alpha
 * (HIPPO: kappa = 1).</li>
 * <li>Alchemical particles. The multipoles and polarizabilities of the particles flagged as alchemical are multiplied
 * by the context parameter named by getAlchemicalParameterName() ("lambda_electrostatics" by default), so that 0
 * removes their electrostatics and polarization from the System.</li>
 * <li>Exponents that are nearly but not exactly equal are handled without the cancellation of HIPPO's
 * unequal-exponent damping formulas, and zero polarizabilities are allowed.</li>
 * </ul>
 *
 * Induced dipoles are computed with HIPPO's extrapolated perturbation series (setExtrapolationCoefficients()); a single
 * coefficient of 1 gives direct polarization.
 */

class OPENMM_EXPORT_INTERAX InteraxMultipoleForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions. The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used. This is the default.
         */
        NoCutoff = 0,
        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the
         * interaction of each particle with all periodic copies of every other particle.
         */
        PME = 1
    };
    /**
     * The local frame types (same values as HippoNonbondedForce).
     */
    enum ParticleAxisTypes { ZThenX = 0, Bisector = 1, ZBisect = 2, ThreeFold = 3, ZOnly = 4, NoAxisType = 5 };
    /**
     * Create an InteraxMultipoleForce.
     */
    InteraxMultipoleForce();
    /**
     * Get the number of particles.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Get the number of exceptions.
     */
    int getNumExceptions() const {
        return exceptions.size();
    }
    /**
     * Get the method used for handling long-range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long-range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) for the real space part of PME.
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) for the real space part of PME.
     */
    void setCutoffDistance(double distance);
    /**
     * Get the coefficients of the mu_0, mu_1, mu_2, ..., mu_n terms of the extrapolated perturbation series for the
     * induced dipoles.
     */
    const std::vector<double>& getExtrapolationCoefficients() const;
    /**
     * Set the coefficients of the mu_0, mu_1, mu_2, ..., mu_n terms of the extrapolated perturbation series for the
     * induced dipoles. A single coefficient of 1 gives direct polarization.
     */
    void setExtrapolationCoefficients(const std::vector<double>& coefficients);
    /**
     * Get the parameters to use for PME calculations. If alpha is 0 (the default), these parameters are ignored and
     * instead their values are chosen based on the Ewald error tolerance.
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Set the parameters to use for PME calculations. If alpha is 0 (the default), these parameters are ignored and
     * instead their values are chosen based on the Ewald error tolerance.
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Get the parameters being used for PME in a particular Context.
     */
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Add the parameters for a particle. This should be called once for each particle in the System. When it is called
     * for the i'th time, it specifies the parameters for the i'th particle.
     *
     * @param charge          the total charge (e)
     * @param dipole          the permanent dipole in the local frame (e nm)
     * @param quadrupole      the permanent quadrupole in the local frame (e nm^2), 9 components of a traceless symmetric
     *                        tensor, in HIPPO's convention
     * @param coreCharge      the point core charge (e); the valence charge is charge - coreCharge
     * @param alpha           the exponent of the Slater valence density (1/nm)
     * @param polarizability  the atomic polarizability (nm^3), zero or positive
     * @param axisType        the local frame type (ParticleAxisTypes)
     * @param multipoleAtomZ  the particle defining the z axis of the local frame
     * @param multipoleAtomX  the particle defining the x axis of the local frame
     * @param multipoleAtomY  the particle defining the y axis of the local frame
     * @param isAlchemical    whether the multipoles and polarizability are scaled by the alchemical parameter
     * @return the index of the particle that was added
     */
    int addParticle(double charge, const std::vector<double>& dipole, const std::vector<double>& quadrupole, double coreCharge,
                    double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                    bool isAlchemical=false);
    /**
     * Get the parameters for a particle (see addParticle()).
     */
    void getParticleParameters(int index, double& charge, std::vector<double>& dipole, std::vector<double>& quadrupole, double& coreCharge,
                               double& alpha, double& polarizability, int& axisType, int& multipoleAtomZ, int& multipoleAtomX,
                               int& multipoleAtomY, bool& isAlchemical) const;
    /**
     * Set the parameters for a particle (see addParticle()).
     */
    void setParticleParameters(int index, double charge, const std::vector<double>& dipole, const std::vector<double>& quadrupole, double coreCharge,
                               double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                               bool isAlchemical=false);
    /**
     * Add scale factors for the interactions of a pair of particles.
     *
     * @param particle1                the first particle
     * @param particle2                the second particle
     * @param multipoleMultipoleScale  the scale of the permanent multipole-multipole interaction
     * @param dipoleMultipoleScale     the scale of the permanent field on the induced dipoles (and of the polarization
     *                                 energy and forces it gives)
     * @param dipoleDipoleScale        the scale of the coupling of the induced dipoles
     * @param replace                  whether to replace an exception that already exists for this pair
     * @return the index of the exception that was added
     */
    int addException(int particle1, int particle2, double multipoleMultipoleScale, double dipoleMultipoleScale, double dipoleDipoleScale,
                     bool replace=false);
    /**
     * Get the scale factors for an exception.
     */
    void getExceptionParameters(int index, int& particle1, int& particle2, double& multipoleMultipoleScale, double& dipoleMultipoleScale,
                                double& dipoleDipoleScale) const;
    /**
     * Set the scale factors for an exception.
     */
    void setExceptionParameters(int index, int particle1, int particle2, double multipoleMultipoleScale, double dipoleMultipoleScale,
                                double dipoleDipoleScale);
    /**
     * Get the error tolerance for Ewald summation.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get whether the permanent fields on the induced dipoles are damped by the density of the polarized atom.
     */
    bool getCoreDamping() const;
    /**
     * Set whether the permanent fields on the induced dipoles are damped by the density of the polarized atom.
     */
    void setCoreDamping(bool enabled);
    /**
     * Get the scale eta of the exponents in the core damping (default 1): the field on an atom is multiplied by f(r)
     * computed with the exponent eta*alpha, so eta < 1 damps over a larger distance.
     */
    double getCoreDampingScale() const;
    /**
     * Set the scale eta of the exponents in the core damping.
     */
    void setCoreDampingScale(double eta);
    /**
     * Get the scale kappa of the exponents in the damping of the induced dipole couplings.
     */
    double getMutualDampingScale() const;
    /**
     * Set the scale kappa of the exponents in the damping of the induced dipole couplings (default 1, as HIPPO).
     */
    void setMutualDampingScale(double kappa);
    /**
     * Get the name of the context parameter that scales the multipoles and polarizabilities of alchemical particles.
     */
    const std::string& getAlchemicalParameterName() const;
    /**
     * Set the name of the context parameter that scales the multipoles and polarizabilities of alchemical particles.
     */
    void setAlchemicalParameterName(const std::string& name);
    /**
     * Get the permanent dipoles of all particles in the lab frame (e nm).
     */
    void getLabFramePermanentDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Get the induced dipoles of all particles (e nm).
     */
    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object. The
     * exceptions, frames and set of alchemical particles cannot be changed.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary conditions.
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == InteraxMultipoleForce::PME;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    class ExceptionInfo;
    NonbondedMethod nonbondedMethod;
    double cutoffDistance;
    double ewaldErrorTol;
    double alpha;
    int nx, ny, nz;
    bool coreDamping;
    double coreDampingScale, mutualDampingScale;
    std::string alchemicalParameterName;
    std::vector<ParticleInfo> particles;
    std::vector<ExceptionInfo> exceptions;
    std::map<std::pair<int, int>, int> exceptionMap;
    std::vector<double> extrapolationCoefficients;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class InteraxMultipoleForce::ParticleInfo {
public:
    int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
    double charge, coreCharge, alpha, polarizability;
    bool isAlchemical;
    std::vector<double> dipole, quadrupole;
    ParticleInfo() {
        axisType = multipoleAtomZ = multipoleAtomX = multipoleAtomY = -1;
        charge = coreCharge = alpha = polarizability = 0.0;
        isAlchemical = false;
        dipole.resize(3);
        quadrupole.resize(9);
    }
    ParticleInfo(double charge, const std::vector<double>& dipole, const std::vector<double>& quadrupole, double coreCharge,
                 double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                 bool isAlchemical) :
        axisType(axisType), multipoleAtomZ(multipoleAtomZ), multipoleAtomX(multipoleAtomX), multipoleAtomY(multipoleAtomY),
        charge(charge), coreCharge(coreCharge), alpha(alpha), polarizability(polarizability), isAlchemical(isAlchemical),
        dipole(dipole), quadrupole(quadrupole) {
    }
};

/**
 * This is an internal class used to record information about an exception.
 * @private
 */
class InteraxMultipoleForce::ExceptionInfo {
public:
    int particle1, particle2;
    double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale;
    ExceptionInfo() {
        particle1 = particle2 = -1;
        multipoleMultipoleScale = dipoleMultipoleScale = dipoleDipoleScale = 0.0;
    }
    ExceptionInfo(int particle1, int particle2, double multipoleMultipoleScale, double dipoleMultipoleScale, double dipoleDipoleScale) :
        particle1(particle1), particle2(particle2), multipoleMultipoleScale(multipoleMultipoleScale),
        dipoleMultipoleScale(dipoleMultipoleScale), dipoleDipoleScale(dipoleDipoleScale) {
    }
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXMULTIPOLEFORCE_H_*/
