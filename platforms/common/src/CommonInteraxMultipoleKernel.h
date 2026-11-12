#ifndef OPENMM_COMMONINTERAXMULTIPOLEKERNEL_H_
#define OPENMM_COMMONINTERAXMULTIPOLEKERNEL_H_

/* -------------------------------------------------------------------------- *
 *                            OpenMM InteraxFF Plugin                         *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM InteraxFF plugin, built on the OpenMM molecular *
 * simulation toolkit (https://openmm.org). It is derived from OpenMM's       *
 * common implementation of HippoNonbondedForce (AMOEBA plugin).              *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Peter Eastman (HippoNonbondedForce), Joao Morado                  *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "InteraxKernels.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include <map>
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * This kernel is invoked by InteraxMultipoleForce to calculate the forces acting on the system and the energy of the system.
 *
 * It is derived from the common implementation of HippoNonbondedForce, without repulsion, dispersion and charge transfer.
 * InteraxFF adds core damping of the permanent fields on the induced dipoles, a scale of the exponents in the damping of
 * the induced dipole couplings, alchemical scaling of the multipoles and polarizabilities, and a series for the damping
 * of Slater densities with nearly equal exponents. Like HippoNonbondedForce, by default it replaces the source of the
 * default nonbonded kernel, so it cannot share a Context with other forces that use it; with
 * InteraxMultipoleForce::setUseOwnKernel() it evaluates the real-space interactions with a kernel of its own instead.
 */
class CommonCalcInteraxMultipoleForceKernel : public CalcInteraxMultipoleForceKernel {
public:
    CommonCalcInteraxMultipoleForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system);
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the InteraxMultipoleForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const InteraxMultipoleForce& force);
    /**
     * Compute the FFT.
     */
    virtual void computeFFT(bool forward) = 0;
    /**
     * Get whether charge spreading should be done in fixed point.
     */
    virtual bool useFixedPointChargeSpreading() const = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Get the induced dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     * 
     * @param context    the Context for which to get the fixed dipoles
     * @param dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the InteraxMultipoleForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxMultipoleForce& force);
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
protected:
    class ForceInfo;
    class TorquePostComputation;
    /**
     * A per-atom array passed to the kernel of the real-space interactions: as a parameter (loaded for both atoms of
     * each pair) or as a plain argument, as the nonbonded utilities pass them to the default kernel.
     */
    struct KernelParameter {
        OpenMM::ComputeArray* array;
        std::string name, componentType;
        int numComponents;
        bool constant;
        std::string getType() const {
            return (numComponents == 1 ? componentType : componentType+std::to_string(numComponents));
        }
    };
    void computeInducedField(int optOrder);
    void computeExtrapolatedDipoles();
    void ensureMultipolesValid(OpenMM::ContextImpl& context);
    void addTorquesToForces();
    void createFieldKernel(const std::string& interactionSrc, std::vector<OpenMM::ComputeArray*> params, OpenMM::ComputeArray& fieldBuffer,
        OpenMM::ComputeKernel& kernel, OpenMM::ComputeKernel& exceptionKernel, OpenMM::ComputeArray& exceptionScale);
    /**
     * Get the defines of the InteraxFF terms, which all kernels of the real-space interactions need.
     */
    std::map<std::string, std::string> getPhysicsDefines() const;
    /**
     * Set the maximum number of tiles in the field kernels to the current size of the neighbor list.
     */
    void updateFieldKernelMaxTiles();
    /**
     * Add the constants of fully excluded pairs at the same position (COINCIDENT_R2, COINCIDENT_B0 ... B5) to a set of
     * defines or replacements.
     */
    void addCoincidentPairConstants(std::map<std::string, std::string>& defines) const;
    /**
     * Get the kernel of the real-space interactions (useOwnKernel), creating it if necessary.  It always computes the
     * forces, and the energy if includeEnergy is true.
     */
    OpenMM::ComputeKernel& getElectrostaticsKernel(bool includeEnergy);
    /**
     * Compute the real-space interactions with the kernel of this force (useOwnKernel).
     */
    void computeElectrostatics(bool includeEnergy);
    int numParticles, maxExtrapolationOrder, maxTiles, fieldThreadBlockSize, forceThreadBlockSize;
    int gridSizeX, gridSizeY, gridSizeZ;
    double pmeAlpha, cutoff, coreDampingScale, mutualDampingScale, lastLambda;
    bool usePME, hasInitializedKernels, multipolesAreValid, coreDamping, hasAlchemicalParticles, useOwnKernel, useChirality;
    std::string alchemicalParameter, electrostaticsSource;
    std::vector<double> extrapolationCoefficients;
    std::vector<KernelParameter> electrostaticsParams, electrostaticsArgs;
    std::map<bool, OpenMM::ComputeKernel> electrostaticsKernels;
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    OpenMM::ComputeArray multipoleParticles, isAlchemical;
    OpenMM::ComputeArray baseCoreCharge, baseValenceCharge, basePolarizability, baseLocalDipoles, baseLocalQuadrupoles;
    OpenMM::ComputeArray coreCharge, valenceCharge, alpha, polarizability;
    OpenMM::ComputeArray localDipoles, labDipoles, fracDipoles;
    OpenMM::ComputeArray localQuadrupoles, labQuadrupoles[5], fracQuadrupoles;
    OpenMM::ComputeArray field;
    OpenMM::ComputeArray inducedField;
    OpenMM::ComputeArray torque;
    OpenMM::ComputeArray inducedDipole;
    OpenMM::ComputeArray extrapolatedDipole, extrapolatedPhi;
    OpenMM::ComputeArray pmeGrid1, pmeGrid2, pmeGridLong;
    OpenMM::ComputeArray pmeBsplineModuliX, pmeBsplineModuliY, pmeBsplineModuliZ;
    OpenMM::ComputeArray pmePhi, pmePhidp, pmeCphi;
    OpenMM::ComputeArray lastPositions;
    OpenMM::ComputeArray exceptionScales[3];
    OpenMM::ComputeArray exceptionAtoms;
    OpenMM::ComputeArray singlePairsPlaceholder;
    OpenMM::ComputeKernel computeMomentsKernel, recordInducedDipolesKernel, mapTorqueKernel, alchemicalKernel;
    OpenMM::ComputeKernel fixedFieldKernel, fixedFieldExceptionKernel, mutualFieldKernel, mutualFieldExceptionKernel, computeExceptionsKernel;
    OpenMM::ComputeKernel pmeSpreadFixedMultipolesKernel, pmeSpreadInducedDipolesKernel, pmeFinishSpreadChargeKernel, pmeConvolutionKernel;
    OpenMM::ComputeKernel pmeFixedPotentialKernel, pmeInducedPotentialKernel, pmeFixedForceKernel, pmeInducedForceKernel, pmeRecordInducedFieldDipolesKernel;
    OpenMM::ComputeKernel pmeSelfEnergyKernel, pmeTransformMultipolesKernel, pmeTransformPotentialKernel;
    OpenMM::ComputeKernel initExtrapolatedKernel, iterateExtrapolatedKernel, computeExtrapolatedKernel, polarizationEnergyKernel;
    static const int PmeOrder = 5;
};

} // namespace InteraxPlugin

#endif /*OPENMM_COMMONINTERAXMULTIPOLEKERNEL_H_*/
