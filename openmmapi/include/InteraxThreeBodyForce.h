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
 * This class implements the InteraxFF three-body (Axilrod-Teller-Muto) dispersion between molecules. Each molecule
 * is a single site at the centroid of a set of its particles (the heavy atoms, in InteraxFF), with a molecular
 * polarizability alpha and a single-pole frequency w (so that its C6 is 3/4 alpha^2 w). Every triple of molecules
 * whose three distances are all shorter than the cutoff contributes
 *
 *   E = scale * C9 (1 + 3 cos t_A cos t_B cos t_C) / (r_AB r_AC r_BC)^3,
 *   C9 = 3/2 a_A a_B a_C w_A w_B w_C (w_A + w_B + w_C) / ((w_A + w_B)(w_B + w_C)(w_C + w_A)),
 *
 * where t are the internal angles of the triangle of centroids. The force on a centroid is shared equally among
 * its particles. Triples are found from a neighbor list of the centroids, so the cost grows with the number of
 * triples inside the cutoff rather than with the cube of the number of molecules.
 *
 * A molecule may be flagged as alchemical. A triple that contains at least one alchemical molecule is multiplied by
 * the context parameter named by getAlchemicalParameterName() ("lambda_sterics" by default).
 */

class OPENMM_EXPORT_INTERAX InteraxThreeBodyForce : public OpenMM::Force {
public:
    /**
     * Create an InteraxThreeBodyForce.
     */
    InteraxThreeBodyForce();
    /**
     * Get the number of molecules in the force.
     */
    int getNumMolecules() const {
        return molecules.size();
    }
    /**
     * Add a molecule to the force.
     *
     * @param particles     the particles whose unweighted centroid is the molecular site
     * @param alpha         the molecular polarizability, measured in nm^3
     * @param frequency     the single-pole frequency w, measured in kJ/mol
     * @param isAlchemical  whether the molecule is alchemical
     * @return the index of the molecule that was added
     */
    int addMolecule(const std::vector<int>& particles, double alpha, double frequency, bool isAlchemical=false);
    /**
     * Get the parameters of a molecule.
     *
     * @param index              the index of the molecule for which to get parameters
     * @param[out] particles     the particles whose unweighted centroid is the molecular site
     * @param[out] alpha         the molecular polarizability, measured in nm^3
     * @param[out] frequency     the single-pole frequency w, measured in kJ/mol
     * @param[out] isAlchemical  whether the molecule is alchemical
     */
    void getMoleculeParameters(int index, std::vector<int>& particles, double& alpha, double& frequency, bool& isAlchemical) const;
    /**
     * Set the parameters of a molecule.
     *
     * @param index         the index of the molecule for which to set parameters
     * @param particles     the particles whose unweighted centroid is the molecular site
     * @param alpha         the molecular polarizability, measured in nm^3
     * @param frequency     the single-pole frequency w, measured in kJ/mol
     * @param isAlchemical  whether the molecule is alchemical
     */
    void setMoleculeParameters(int index, const std::vector<int>& particles, double alpha, double frequency, bool isAlchemical);
    /**
     * Get the cutoff distance (in nm): a triple interacts only if all three of its centroid distances are shorter.
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm): a triple interacts only if all three of its centroid distances are shorter.
     */
    void setCutoffDistance(double distance);
    /**
     * Get the factor that multiplies every C9 coefficient.
     */
    double getScaleFactor() const;
    /**
     * Set the factor that multiplies every C9 coefficient.
     */
    void setScaleFactor(double scale);
    /**
     * Set whether to use periodic boundary conditions. Each centroid distance is that of the nearest periodic image,
     * as in CustomCentroidBondForce; the cutoff may exceed half the box size, in which case a triple still
     * interacts only through the nearest image of each of its pairs.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Get the name of the context parameter that scales the triples containing an alchemical molecule.
     */
    const std::string& getAlchemicalParameterName() const;
    /**
     * Set the name of the context parameter that scales the triples containing an alchemical molecule. Its default
     * value is 1.
     */
    void setAlchemicalParameterName(const std::string& name);
    /**
     * Update the molecular polarizabilities and frequencies and the scale factor in a Context to match those stored
     * in this Force object. This method provides an efficient method to update certain parameters in an existing
     * Context without needing to reinitialize it. Simply call setMoleculeParameters() to modify this object's
     * parameters, then call updateParametersInContext() to copy them over to the Context.
     *
     * The particles of each molecule, the alchemical flags and the cutoff cannot be changed by this method.
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
    class MoleculeInfo;
    double cutoffDistance, scaleFactor;
    bool usePeriodic;
    std::string alchemicalParameterName;
    std::vector<MoleculeInfo> molecules;
};

/**
 * This is an internal class used to record information about a molecule.
 * @private
 */
class InteraxThreeBodyForce::MoleculeInfo {
public:
    std::vector<int> particles;
    double alpha, frequency;
    bool isAlchemical;
    MoleculeInfo() : alpha(0.0), frequency(0.0), isAlchemical(false) {
    }
    MoleculeInfo(const std::vector<int>& particles, double alpha, double frequency, bool isAlchemical) :
        particles(particles), alpha(alpha), frequency(frequency), isAlchemical(isAlchemical) {
    }
};

} // namespace InteraxPlugin

#endif /*OPENMM_INTERAXTHREEBODYFORCE_H_*/
