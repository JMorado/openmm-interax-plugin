#ifndef OPENMM_INTERAXPAIRFORCE_H_
#define OPENMM_INTERAXPAIRFORCE_H_

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

#include "openmm/Force.h"
#include "internal/windowsExportInterax.h"
#include <string>
#include <utility>
#include <vector>

namespace InteraxPlugin {

/**
 * This is the abstract base class of the InteraxFF pair forces: InteraxDispersionForce, InteraxExchangeForce and
 * InteraxShortRangeForce.  It holds the settings they have in common: the method used for handling long range
 * interactions, the cutoff and switching function, the list of excluded particle pairs, and the treatment of
 * alchemical particles.  The per-particle parameters and the energy expression are defined by the subclasses.
 *
 * Every particle of the System must be added to the force, in the same order as in the System.  The number of
 * particles must be exactly equal to the number of particles in the System, or else an exception will be thrown
 * when you try to create a Context.  Every pair of particles interacts unless it has been excluded with
 * addExclusion(); the InteraxFF force field excludes every pair of particles within the same molecule.  When all
 * pair forces of a System have the same exclusions and cutoff, the CUDA and OpenCL platforms evaluate them together
 * with the default nonbonded kernel and its neighbor list.
 *
 * When using a cutoff, by default the interaction is sharply truncated at the cutoff distance.  Optionally you can
 * instead use a switching function to make the interaction smoothly go to zero over a finite distance range.  To
 * enable this, call setUseSwitchingFunction().  You must also call setSwitchingDistance() to specify the distance at
 * which the interaction should begin to decrease.  The switching distance must be less than the cutoff distance.
 * The pair energy is then multiplied by S(x) = 1 - 10x^3 + 15x^4 - 6x^5, x = (r - r_switch)/(r_cutoff - r_switch),
 * exactly as in CustomNonbondedForce.
 *
 * A particle may be flagged as alchemical.  The energy of a pair with exactly one alchemical particle is multiplied
 * by lambda, the value of the context parameter named by getAlchemicalParameterName() ("lambda_sterics" by
 * default), so that setting it to 0 decouples the alchemical particles from the rest of the System.  Pairs with
 * zero or two alchemical particles are not scaled.  Optionally, the scaled pairs are evaluated at a soft-core
 * distance (see setSoftcoreRadius()), so that an alchemical particle cannot sink into its partners while its
 * interactions are partly switched off.
 */

class OPENMM_EXPORT_INTERAX InteraxPairForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range interactions.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to the interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Interactions beyond the cutoff distance are ignored.
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy
         * of each other particle.  Interactions beyond the cutoff distance are ignored.
         */
        CutoffPeriodic = 2,
    };
    virtual ~InteraxPairForce() {
    }
    /**
     * Get the number of particles for which force field parameters are defined.
     */
    virtual int getNumParticles() const = 0;
    /**
     * Get whether a particle is alchemical.
     *
     * @param index    the index of the particle
     * @return true if the particle is alchemical
     */
    virtual bool getParticleIsAlchemical(int index) const = 0;
    /**
     * Get the method used for handling long range interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used.  If the NonbondedMethod in use is NoCutoff, this value will have
     * no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used.  If the NonbondedMethod in use is NoCutoff, this value will have
     * no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Get whether a switching function is applied to the interaction.  If the nonbonded method is set to NoCutoff,
     * this option is ignored.
     */
    bool getUseSwitchingFunction() const;
    /**
     * Set whether a switching function is applied to the interaction.  If the nonbonded method is set to NoCutoff,
     * this option is ignored.
     */
    void setUseSwitchingFunction(bool use);
    /**
     * Get the distance at which the switching function begins to reduce the interaction.  This must be less than
     * the cutoff distance.
     */
    double getSwitchingDistance() const;
    /**
     * Set the distance at which the switching function begins to reduce the interaction.  This must be less than
     * the cutoff distance.
     */
    void setSwitchingDistance(double distance);
    /**
     * Get the number of particle pairs whose interactions should be excluded.
     */
    int getNumExclusions() const {
        return exclusions.size();
    }
    /**
     * Add a particle pair to the list of interactions that should be excluded.
     *
     * @param particle1  the index of the first particle in the pair
     * @param particle2  the index of the second particle in the pair
     * @return the index of the exclusion that was added
     */
    int addExclusion(int particle1, int particle2);
    /**
     * Get the particles in a pair whose interaction should be excluded.
     *
     * @param index           the index of the exclusion for which to get particle indices
     * @param[out] particle1  the index of the first particle in the pair
     * @param[out] particle2  the index of the second particle in the pair
     */
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    /**
     * Set the particles in a pair whose interaction should be excluded.
     *
     * @param index      the index of the exclusion for which to set particle indices
     * @param particle1  the index of the first particle in the pair
     * @param particle2  the index of the second particle in the pair
     */
    void setExclusionParticles(int index, int particle1, int particle2);
    /**
     * Get the name of the context parameter that scales the interactions between an alchemical and a non-alchemical
     * particle.
     */
    const std::string& getAlchemicalParameterName() const;
    /**
     * Set the name of the context parameter that scales the interactions between an alchemical and a non-alchemical
     * particle.  If any particle is alchemical, the force defines this parameter with a default value of 1.
     *
     * @param name    the name of the context parameter
     */
    void setAlchemicalParameterName(const std::string& name);
    /**
     * Get the soft-core radius R (in nm).  A pair with exactly one alchemical particle at distance r is evaluated at
     * r_eff = (r^6 + (1 - lambda) R^6)^(1/6) and scaled by lambda, so the interaction is exact at lambda = 1 and the
     * particles cannot come closer than (1 - lambda)^(1/6) R in it.  The InteraxFF pair terms are finite at r = 0,
     * but their sum is not bounded below at deep overlap (the short-range correction can outweigh the exchange
     * repulsion there), and without electrostatics nothing else keeps decoupled particles apart.  A value of 0 (the
     * default) disables the soft core.
     *
     * @return the soft-core radius, measured in nm
     */
    double getSoftcoreRadius() const;
    /**
     * Set the soft-core radius R (in nm).  A value of 0 disables the soft core.  See getSoftcoreRadius() for details.
     *
     * @param radius    the soft-core radius, measured in nm.  It must not be negative.
     */
    void setSoftcoreRadius(double radius);
    /**
     * Returns whether or not this force makes use of periodic boundary conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == CutoffPeriodic;
    }
protected:
    InteraxPairForce();
private:
    NonbondedMethod nonbondedMethod;
    double cutoffDistance, switchingDistance, softcoreRadius;
    bool useSwitchingFunction;
    std::string alchemicalParameterName;
    std::vector<std::pair<int, int> > exclusions;
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXPAIRFORCE_H_*/
