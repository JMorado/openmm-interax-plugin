#ifndef COMMON_INTERAX_MULTIPOLE_KERNEL_H_
#define COMMON_INTERAX_MULTIPOLE_KERNEL_H_

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
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include <map>
#include <string>
#include <vector>

namespace InteraxPlugin {

/**
 * The common (CUDA/OpenCL) implementation of InteraxMultipoleForce: OpenMM's HippoNonbondedForce kernels without
 * repulsion, dispersion and charge transfer, with core damping, the mutual damping scale, alchemical scaling and the
 * series for nearly equal exponents. The FFTs are platform specific (computeFFT()).
 *
 * Like HippoNonbondedForce, it replaces the kernel of the default NonbondedUtilities, so it cannot share a Context
 * with other nonbonded forces (the InteraxFF export wraps it in a CustomCVForce).
 */
class CommonCalcInteraxMultipoleForceKernel : public CalcInteraxMultipoleForceKernel {
public:
    CommonCalcInteraxMultipoleForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc,
                                          const OpenMM::System& system);
    void initialize(const OpenMM::System& system, const InteraxMultipoleForce& force);
    virtual void computeFFT(bool forward) = 0;
    virtual bool useFixedPointChargeSpreading() const = 0;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void copyParametersToContext(OpenMM::ContextImpl& context, const InteraxMultipoleForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
protected:
    class ForceInfo;
    class TorquePostComputation;
    void setParticleParameters(const InteraxMultipoleForce& force);
    void computeInducedField(int optOrder);
    void updateFieldKernelMaxTiles();
    void computeExtrapolatedDipoles();
    void ensureMultipolesValid(OpenMM::ContextImpl& context);
    void addTorquesToForces();
    void createFieldKernel(const std::string& interactionSrc, std::vector<OpenMM::ComputeArray*> params, OpenMM::ComputeArray& fieldBuffer,
                           OpenMM::ComputeKernel& kernel, OpenMM::ComputeKernel& exceptionKernel, OpenMM::ComputeArray& exceptionScale);
    std::map<std::string, std::string> getPhysicsDefines() const;
    int numParticles, maxExtrapolationOrder, maxTiles, fieldThreadBlockSize;
    int gridSizeX, gridSizeY, gridSizeZ;
    double pmeAlpha, cutoff, coreDampingScale, mutualDampingScale, lastLambda;
    bool usePME, coreDamping, hasAlchemicalParticles, hasInitializedKernels, multipolesAreValid;
    std::string alchemicalParameter;
    std::vector<double> extrapolationCoefficients;
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
    OpenMM::ComputeKernel alchemicalKernel, computeMomentsKernel, recordInducedDipolesKernel, mapTorqueKernel;
    OpenMM::ComputeKernel fixedFieldKernel, fixedFieldExceptionKernel, mutualFieldKernel, mutualFieldExceptionKernel, computeExceptionsKernel;
    OpenMM::ComputeKernel pmeSpreadFixedMultipolesKernel, pmeSpreadInducedDipolesKernel, pmeFinishSpreadChargeKernel, pmeConvolutionKernel;
    OpenMM::ComputeKernel pmeFixedPotentialKernel, pmeInducedPotentialKernel, pmeFixedForceKernel, pmeInducedForceKernel, pmeRecordInducedFieldDipolesKernel;
    OpenMM::ComputeKernel pmeSelfEnergyKernel, pmeTransformMultipolesKernel, pmeTransformPotentialKernel;
    OpenMM::ComputeKernel initExtrapolatedKernel, iterateExtrapolatedKernel, computeExtrapolatedKernel, polarizationEnergyKernel;
    static const int PmeOrder = 5;
};

} // namespace InteraxPlugin

#endif /*COMMON_INTERAX_MULTIPOLE_KERNEL_H_*/
