#ifndef OPENMM_INTERAXTHREEBODYFORCE_H_
#define OPENMM_INTERAXTHREEBODYFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "internal/windowsExportInterax.h"
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * This class implements the three-body (Axilrod-Teller-Muto) dispersion of the InteraxFF force field.  The
 * interacting sites are the unweighted centroids of sets of particles: a whole molecule (its heavy atoms) for the
 * molecular ATM of InteraxFF, or a single atom for an atomic ATM.  Each site has a polarizability alpha and a
 * single-pole frequency w (so that its C6 coefficient is 3/4 alpha^2 w).  Every triple of sites A, B, C whose three
 * distances are all shorter than the cutoff contributes
 *
 *   E = scale C9 (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC r_BC)^3 f(r_AB) f(r_AC) f(r_BC)
 *   C9 = 3/2 a_A a_B a_C w_A w_B w_C (w_A + w_B + w_C) / ((w_A + w_B)(w_B + w_C)(w_C + w_A))
 *
 * where t_A, t_B, t_C are the internal angles of the triangle of sites, scale is a global factor (see
 * setScaleFactor()), and f is the damping function selected with setDampingMethod(): f = 1 without damping, or the
 * Tang-Toennies function f_3(x) = 1 - exp(-x) sum_{k=0}^{3} x^k/k!, x = beta 2 r_ij / (s_i + s_j), of each of the
 * three dipole propagators (r^-3) of the triple, with s the site widths and beta the damping scale.  Short range
 * damping is needed for atomic sites, which come much closer than molecular centroids.
 *
 * Independently of the damping method, a centroid damping radius R0 > 0 (see setCentroidDampingRadius()) multiplies
 * each triple by g(r_AB) g(r_AC) g(r_BC), g(r) = 1 - exp(-(r/R0)^6).  It keeps the molecular ATM finite when two
 * centroids come close (flexible molecules), where the undamped energy diverges as r^-3 per pair.  R0 = 0 (the
 * default) means no centroid damping.
 *
 * Each site belongs to a molecule: a triple containing two sites of the same molecule does not interact, so that
 * only intermolecular triples are counted.  The force on a site is shared equally among its particles.  Triples are
 * found from a neighbor list of the sites, so the cost grows with the number of triples inside the cutoff rather
 * than with the cube of the number of sites.
 *
 * To use this class, create an InteraxThreeBodyForce object, then call addSite() once for each site.  After a site
 * has been added, you can modify its parameters by calling setSiteParameters().  This will have no effect on
 * Contexts that already exist unless you call updateParametersInContext().
 *
 * A site may be flagged as alchemical.  The energy of a triple that contains at least one alchemical site is
 * multiplied by the context parameter named by getAlchemicalParameterName() ("lambda_sterics" by default).
 */

class OPENMM_EXPORT_INTERAX InteraxThreeBodyForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the damping functions that may be applied to the triples.
     */
    enum DampingMethod {
        /**
         * No damping (f = 1), for molecular sites (the InteraxFF molecular ATM).  This is the default.
         */
        NoDamping = 0,
        /**
         * A Tang-Toennies damping f_3 of each of the three pairs of a triple.
         */
        TangToennies = 1
    };
    /**
     * Create an InteraxThreeBodyForce.
     */
    InteraxThreeBodyForce();
    /**
     * Get the number of sites in the force.
     */
    int getNumSites() const {
        return sites.size();
    }
    /**
     * Add a site to the force.
     *
     * @param particles     the particles whose unweighted centroid is the site
     * @param alpha         the polarizability, measured in nm^3
     * @param frequency     the single-pole frequency w, measured in kJ/mol
     * @param width         the width s of the site used by the damping, measured in nm (unused without damping)
     * @param molecule      the molecule the site belongs to: sites of the same molecule do not form triples. A
     *                      negative value puts the site in a molecule of its own.
     * @param isAlchemical  whether the site is alchemical
     * @return the index of the site that was added
     */
    int addSite(const std::vector<int>& particles, double alpha, double frequency, double width=0.0, int molecule=-1, bool isAlchemical=false);
    /**
     * Get the parameters of a site.
     *
     * @param index              the index of the site for which to get parameters
     * @param[out] particles     the particles whose unweighted centroid is the site
     * @param[out] alpha         the polarizability, measured in nm^3
     * @param[out] frequency     the single-pole frequency w, measured in kJ/mol
     * @param[out] width         the width s of the site used by the damping, measured in nm
     * @param[out] molecule      the molecule the site belongs to, or a negative value for a molecule of its own
     * @param[out] isAlchemical  whether the site is alchemical
     */
    void getSiteParameters(int index, std::vector<int>& particles, double& alpha, double& frequency, double& width, int& molecule,
            bool& isAlchemical) const;
    /**
     * Set the parameters of a site.
     *
     * @param index         the index of the site for which to set parameters
     * @param particles     the particles whose unweighted centroid is the site
     * @param alpha         the polarizability, measured in nm^3
     * @param frequency     the single-pole frequency w, measured in kJ/mol
     * @param width         the width s of the site used by the damping, measured in nm
     * @param molecule      the molecule the site belongs to, or a negative value for a molecule of its own
     * @param isAlchemical  whether the site is alchemical
     */
    void setSiteParameters(int index, const std::vector<int>& particles, double alpha, double frequency, double width, int molecule,
            bool isAlchemical);
    /**
     * Get the cutoff distance (in nm) being used.  A triple interacts only if all three of its site distances are
     * shorter than the cutoff.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used.  A triple interacts only if all three of its site distances are
     * shorter than the cutoff.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Get the factor that multiplies every C9 coefficient.
     *
     * @return the scale factor (dimensionless)
     */
    double getScaleFactor() const;
    /**
     * Set the factor that multiplies every C9 coefficient.
     *
     * @param scale    the scale factor (dimensionless)
     */
    void setScaleFactor(double scale);
    /**
     * Get the damping function applied to the triples.
     */
    DampingMethod getDampingMethod() const;
    /**
     * Set the damping function applied to the triples.
     */
    void setDampingMethod(DampingMethod method);
    /**
     * Get the scale beta of the Tang-Toennies damping argument x = beta 2 r / (s_i + s_j).
     *
     * @return the damping scale (dimensionless)
     */
    double getDampingScale() const;
    /**
     * Set the scale beta of the Tang-Toennies damping argument x = beta 2 r / (s_i + s_j).
     *
     * @param scale    the damping scale (dimensionless)
     */
    void setDampingScale(double scale);
    /**
     * Get the centroid damping radius R0 (in nm).  Each triple is multiplied by g(r_AB) g(r_AC) g(r_BC) with
     * g(r) = 1 - exp(-(r/R0)^6).  A value of 0 means no centroid damping.
     *
     * @return the centroid damping radius, measured in nm
     */
    double getCentroidDampingRadius() const;
    /**
     * Set the centroid damping radius R0 (in nm).  Each triple is multiplied by g(r_AB) g(r_AC) g(r_BC) with
     * g(r) = 1 - exp(-(r/R0)^6).  A value of 0 (the default) means no centroid damping.  It is applied in addition
     * to the damping method.
     *
     * @param radius    the centroid damping radius, measured in nm.  It must not be negative.
     */
    void setCentroidDampingRadius(double radius);
    /**
     * Set whether to use periodic boundary conditions.  Each site distance is that of the nearest periodic image,
     * as in CustomCentroidBondForce.  The cutoff may exceed half the box size, in which case a triple still
     * interacts only through the nearest image of each of its pairs.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Get the name of the context parameter that scales the triples containing an alchemical site.
     */
    const std::string& getAlchemicalParameterName() const;
    /**
     * Set the name of the context parameter that scales the triples containing an alchemical site.  If any site is
     * alchemical, the force defines this parameter with a default value of 1.
     *
     * @param name    the name of the context parameter
     */
    void setAlchemicalParameterName(const std::string& name);
    /**
     * Update the per-site parameters, the scale factor, the damping scale and the centroid damping radius in a
     * Context to match those stored in this Force object.  This method provides an efficient method to update certain
     * parameters in an existing Context without needing to reinitialize it.  Simply call setSiteParameters() to
     * modify this object's parameters, then call updateParametersInContext() to copy them over to the Context.
     *
     * This method has several limitations.  Only the polarizabilities, frequencies and widths of the sites can be
     * changed: the particles and molecules of the sites, their alchemical flags, the damping method and the cutoff
     * cannot, nor can the centroid damping be switched on or off (a nonzero radius can be changed).  It cannot be used
     * to add new sites, only to change the parameters of existing ones.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return usePeriodic;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class SiteInfo;
    double cutoffDistance, scaleFactor, dampingScale, centroidDampingRadius;
    DampingMethod dampingMethod;
    bool usePeriodic;
    std::string alchemicalParameterName;
    std::vector<SiteInfo> sites;
};

/**
 * This is an internal class used to record information about a site.
 * @private
 */
class InteraxThreeBodyForce::SiteInfo {
public:
    std::vector<int> particles;
    double alpha, frequency, width;
    int molecule;
    bool isAlchemical;
    SiteInfo() : alpha(0.0), frequency(0.0), width(0.0), molecule(-1), isAlchemical(false) {
    }
    SiteInfo(const std::vector<int>& particles, double alpha, double frequency, double width, int molecule, bool isAlchemical) :
        particles(particles), alpha(alpha), frequency(frequency), width(width), molecule(molecule), isAlchemical(isAlchemical) {
    }
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXTHREEBODYFORCE_H_*/
