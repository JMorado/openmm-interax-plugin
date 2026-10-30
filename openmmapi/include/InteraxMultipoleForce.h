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
 * This class implements the electrostatics and polarization of the InteraxFF force field.  It is derived from
 * OpenMM's HippoNonbondedForce, and computes the same electrostatics and induction as HIPPO (Rackers et al.):
 * each particle carries a permanent charge, dipole and quadrupole defined in a local coordinate frame, with the
 * charge split into a point core charge and a Slater valence density of exponent alpha (the charge penetration of
 * HIPPO), and an atomic polarizability that gives it an induced dipole.  The repulsion, dispersion and charge
 * transfer terms of HIPPO are not included.  InteraxMultipoleForce adds the following features:
 *
 * <ul>
 * <li>Core damping (setCoreDamping()).  The permanent field that polarizes an atom is multiplied by
 * f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r), the fraction of the Slater density of the polarized atom (exponent
 * a = eta alpha, see setCoreDampingScale()) within r.  With HIPPO's bare core fields the polarization energy diverges
 * as -polarizability*q_core^2/(2 r^4) when atoms overlap; with core damping it stays finite.</li>
 * <li>A mutual damping scale (setMutualDampingScale()): the couplings between induced dipoles are damped with the
 * exponents kappa*alpha (kappa = 1 in HIPPO).</li>
 * <li>Alchemical particles.  The multipoles and polarizabilities of the particles flagged as alchemical are multiplied
 * by the context parameter named by getAlchemicalParameterName() ("lambda_electrostatics" by default), so that setting
 * it to 0 removes their electrostatics and polarization from the System.</li>
 * <li>Exponents that are nearly but not exactly equal are handled without the cancellation of HIPPO's
 * unequal-exponent damping formulas, and zero polarizabilities are allowed.</li>
 * </ul>
 *
 * Induced dipoles are computed with HIPPO's extrapolated perturbation series (see setExtrapolationCoefficients()).  A
 * single coefficient of 1 gives direct polarization.
 *
 * To use it, create an InteraxMultipoleForce object, then call addParticle() once for each particle.  After
 * an entry has been added, you can modify its force field parameters by calling setParticleParameters().
 * This will have no effect on Contexts that already exist unless you call updateParametersInContext().
 *
 * You also can specify "exceptions", particular pairs of particles whose interactions should be
 * reduced or completely omitted.  Call addException() to define exceptions.
 */

class OPENMM_EXPORT_INTERAX InteraxMultipoleForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 1
    };
    /**
     * This is an enumeration of the types of local coordinate frame, with the same values as in HippoNonbondedForce.
     */
    enum ParticleAxisTypes { ZThenX = 0, Bisector = 1, ZBisect = 2, ThreeFold = 3, ZOnly = 4, NoAxisType = 5 };
    /**
     * This is an enumeration of the ways the chirality of ZThenX frames is handled (see setChiralityMode()).
     */
    enum ChiralityMode {
        /**
         * The local frame is always x, y = z cross x, z: the y atom of a ZThenX frame is ignored.
         */
        NoChirality = 0,
        /**
         * AMOEBA's chirality check, at every evaluation on every platform: for a ZThenX particle with a y atom, the y
         * component of the dipole and the xy and yz components of the quadrupole change sign when the particle lies
         * on the negative side of the plane of its z, x and y atoms, (r_i - r_y) . ((r_z - r_y) x (r_x - r_y)) < 0.
         */
        AmoebaChirality = 1
    };
    /**
     * Create an InteraxMultipoleForce.
     */
    InteraxMultipoleForce();
    /**
     * Get the number of particles in the potential function.
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
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Get the coefficients for the mu_0, mu_1, mu_2, ..., mu_n terms in the extrapolation
     * algorithm for induced dipoles.
     */
    const std::vector<double>& getExtrapolationCoefficients() const;
    /**
     * Set the coefficients for the mu_0, mu_1, mu_2, ..., mu_n terms in the extrapolation
     * algorithm for induced dipoles.  A single coefficient of 1 gives direct polarization.
     *
     * @param coefficients      a vector whose mth entry specifies the coefficient for mu_m.  The length of this
     *                          vector determines how many iterations are performed.  It must not be empty.
     */
    void setExtrapolationCoefficients(const std::vector<double>& coefficients);
    /**
     * Get the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Add the nonbonded force parameters for a particle.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     *
     * @param charge            the particle's total charge, measured in units of the proton charge
     * @param dipole            the particle's molecular dipole (vector of size 3), measured in e*nm
     * @param quadrupole        the particle's molecular quadrupole (vector of size 9), measured in e*nm^2.  It must be
     *                          traceless and symmetric, in the same convention as HippoNonbondedForce.
     * @param coreCharge        the charge of the atomic core.  The valence charge is charge - coreCharge.
     * @param alpha             controls the width of the particle's electron density: the exponent of the Slater
     *                          valence density, measured in 1/nm.  It must be positive.
     * @param polarizability    atomic polarizability, measured in nm^3.  It must not be negative.
     * @param axisType          the particle's axis type
     * @param multipoleAtomZ    index of first atom used in defining the local coordinate system for multipoles
     * @param multipoleAtomX    index of second atom used in defining the local coordinate system for multipoles
     * @param multipoleAtomY    index of third atom used in defining the local coordinate system for multipoles
     * @param isAlchemical      whether the multipoles and polarizability are scaled by the alchemical parameter
     *
     * @return the index of the particle that was added
     */
    int addParticle(double charge, const std::vector<double>& dipole, const std::vector<double>& quadrupole, double coreCharge,
                    double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                    bool isAlchemical=false);
    /**
     * Get the nonbonded force parameters for a particle.
     *
     * @param index                 the index of the particle for which to get parameters
     * @param[out] charge           the particle's total charge, measured in units of the proton charge
     * @param[out] dipole           the particle's molecular dipole (vector of size 3), measured in e*nm
     * @param[out] quadrupole       the particle's molecular quadrupole (vector of size 9), measured in e*nm^2
     * @param[out] coreCharge       the charge of the atomic core
     * @param[out] alpha            the exponent of the Slater valence density, measured in 1/nm
     * @param[out] polarizability   atomic polarizability, measured in nm^3
     * @param[out] axisType         the particle's axis type
     * @param[out] multipoleAtomZ   index of first atom used in defining the local coordinate system for multipoles
     * @param[out] multipoleAtomX   index of second atom used in defining the local coordinate system for multipoles
     * @param[out] multipoleAtomY   index of third atom used in defining the local coordinate system for multipoles
     * @param[out] isAlchemical     whether the multipoles and polarizability are scaled by the alchemical parameter
     */
    void getParticleParameters(int index, double& charge, std::vector<double>& dipole, std::vector<double>& quadrupole, double& coreCharge,
                               double& alpha, double& polarizability, int& axisType, int& multipoleAtomZ, int& multipoleAtomX,
                               int& multipoleAtomY, bool& isAlchemical) const;
    /**
     * Set the nonbonded force parameters for a particle.
     *
     * @param index             the index of the particle for which to set parameters
     * @param charge            the particle's total charge, measured in units of the proton charge
     * @param dipole            the particle's molecular dipole (vector of size 3), measured in e*nm
     * @param quadrupole        the particle's molecular quadrupole (vector of size 9), measured in e*nm^2
     * @param coreCharge        the charge of the atomic core
     * @param alpha             the exponent of the Slater valence density, measured in 1/nm
     * @param polarizability    atomic polarizability, measured in nm^3
     * @param axisType          the particle's axis type
     * @param multipoleAtomZ    index of first atom used in defining the local coordinate system for multipoles
     * @param multipoleAtomX    index of second atom used in defining the local coordinate system for multipoles
     * @param multipoleAtomY    index of third atom used in defining the local coordinate system for multipoles
     * @param isAlchemical      whether the multipoles and polarizability are scaled by the alchemical parameter
     */
    void setParticleParameters(int index, double charge, const std::vector<double>& dipole, const std::vector<double>& quadrupole, double coreCharge,
                               double alpha, double polarizability, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY,
                               bool isAlchemical=false);
    /**
     * Add an interaction to the list of exceptions that should be calculated differently from other interactions.
     * If all scale factors are set to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     *
     * @param particle1                  the index of the first particle involved in the interaction
     * @param particle2                  the index of the second particle involved in the interaction
     * @param multipoleMultipoleScale    the factor by which to scale the Coulomb interaction between fixed multipoles
     * @param dipoleMultipoleScale       the factor by which to scale the Coulomb interaction between an induced dipole and a fixed multipole
     *                                   (the permanent field on the induced dipoles, and the polarization energy and forces it gives)
     * @param dipoleDipoleScale          the factor by which to scale the Coulomb interaction between induced dipoles
     * @param replace                    determines the behavior if there is already an exception for the same two particles.  If true, the existing one is replaced.
     *                                   If false, an exception is thrown.
     * @return the index of the exception that was added
     */
    int addException(int particle1, int particle2, double multipoleMultipoleScale, double dipoleMultipoleScale, double dipoleDipoleScale,
                     bool replace=false);
    /**
     * Get the scale factors for an interaction that should be calculated differently from others.
     *
     * @param index                           the index of the interaction for which to get parameters
     * @param[out] particle1                  the index of the first particle involved in the interaction
     * @param[out] particle2                  the index of the second particle involved in the interaction
     * @param[out] multipoleMultipoleScale    the factor by which to scale the Coulomb interaction between fixed multipoles
     * @param[out] dipoleMultipoleScale       the factor by which to scale the Coulomb interaction between an induced dipole and a fixed multipole
     * @param[out] dipoleDipoleScale          the factor by which to scale the Coulomb interaction between induced dipoles
     */
    void getExceptionParameters(int index, int& particle1, int& particle2, double& multipoleMultipoleScale, double& dipoleMultipoleScale,
                                double& dipoleDipoleScale) const;
    /**
     * Set the scale factors for an interaction that should be calculated differently from others.
     * If all scale factors are set to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     *
     * @param index                      the index of the interaction for which to set parameters
     * @param particle1                  the index of the first particle involved in the interaction
     * @param particle2                  the index of the second particle involved in the interaction
     * @param multipoleMultipoleScale    the factor by which to scale the Coulomb interaction between fixed multipoles
     * @param dipoleMultipoleScale       the factor by which to scale the Coulomb interaction between an induced dipole and a fixed multipole
     * @param dipoleDipoleScale          the factor by which to scale the Coulomb interaction between induced dipoles
     */
    void setExceptionParameters(int index, int particle1, int particle2, double multipoleMultipoleScale, double dipoleMultipoleScale,
                                double dipoleDipoleScale);
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get whether core damping is used: the permanent field that polarizes an atom is multiplied by the fraction
     * f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r) of the atom's own Slater density within r.
     */
    bool getCoreDamping() const;
    /**
     * Set whether core damping is used: the permanent field that polarizes an atom is multiplied by the fraction
     * f(r) = 1 - (1 + a r + (a r)^2/2) exp(-a r) of the atom's own Slater density within r.  This is off by default,
     * as in HIPPO.
     */
    void setCoreDamping(bool enabled);
    /**
     * Get the scale eta of the exponents used in the core damping: the field on an atom is damped with the exponent
     * a = eta*alpha, so eta < 1 damps over a larger distance.  The default value is 1.
     */
    double getCoreDampingScale() const;
    /**
     * Set the scale eta of the exponents used in the core damping: the field on an atom is damped with the exponent
     * a = eta*alpha, so eta < 1 damps over a larger distance.  The default value is 1.
     *
     * @param eta    the core damping scale.  It must be positive.
     */
    void setCoreDampingScale(double eta);
    /**
     * Get the scale kappa of the exponents used in the damping of the couplings between induced dipoles, which use
     * the exponents kappa*alpha.  The default value is 1, as in HIPPO.
     */
    double getMutualDampingScale() const;
    /**
     * Set the scale kappa of the exponents used in the damping of the couplings between induced dipoles, which use
     * the exponents kappa*alpha.  The default value is 1, as in HIPPO.
     *
     * @param kappa    the mutual damping scale.  It must be positive.
     */
    void setMutualDampingScale(double kappa);
    /**
     * Get whether the GPU platforms evaluate the real-space multipole interactions with a kernel of this force's own
     * (true), instead of replacing the source of the context's default nonbonded kernel as HippoNonbondedForce does
     * (false, the default).  With true the force can share a Context with other forces that use the default nonbonded
     * kernel (NonbondedForce, CustomNonbondedForce, the InteraxFF pair forces).  Their exclusions must then be the
     * same as the exceptions of this force, and forces in the same force group must use the same cutoff.
     */
    bool getUseOwnKernel() const;
    /**
     * Get how the chirality of ZThenX frames with a y atom is handled.  The default is AmoebaChirality, as in
     * HippoNonbondedForce on the GPU platforms.
     */
    ChiralityMode getChiralityMode() const;
    /**
     * Set how the chirality of ZThenX frames with a y atom is handled: NoChirality ignores the y atom; AmoebaChirality
     * (the default) flips the y components of the multipoles of a particle whose z, x, y atoms are arranged with
     * negative volume, at every evaluation.  HippoNonbondedForce applies this check at every step on its GPU
     * platforms but on its Reference platform only once, before the positions are known (so never); in
     * InteraxMultipoleForce all platforms apply it at every evaluation.
     */
    void setChiralityMode(ChiralityMode mode);
    /**
     * Set whether the GPU platforms evaluate the real-space multipole interactions with a kernel of this force's own.
     * See getUseOwnKernel() for details.
     */
    void setUseOwnKernel(bool enabled);
    /**
     * Get the name of the context parameter that scales the multipoles and polarizabilities of alchemical particles.
     */
    const std::string& getAlchemicalParameterName() const;
    /**
     * Set the name of the context parameter that scales the multipoles and polarizabilities of alchemical particles.
     * If any particle is alchemical, the force defines this parameter with a default value of 1.
     *
     * @param name    the name of the context parameter
     */
    void setAlchemicalParameterName(const std::string& name);
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     *
     * @param context         the Context for which to get the fixed dipoles
     * @param[out] dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Get the induced dipole moments of all particles.
     *
     * @param context         the Context for which to get the induced dipoles
     * @param[out] dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() to modify this object's parameters, then call updateParametersInContext() to
     * copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles and exceptions.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, only the scale factors for an exception  can be changed; the
     * pair of particles involved in the exception cannot change.  The local frames and the set of alchemical particles
     * cannot change either.  Finally, this method cannot be used to add new particles or exceptions, only to change the
     * parameters of existing ones.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if nonbondedMethod uses PBC and false otherwise
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
    bool coreDamping, useOwnKernel;
    ChiralityMode chiralityMode;
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
