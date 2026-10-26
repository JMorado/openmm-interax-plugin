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

#include "CommonInteraxMultipoleKernel.h"
#include "CommonInteraxKernelSources.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <cmath>
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

/**
 * Write a set of defines as preprocessor directives, for kernel sources that the nonbonded utilities compile.
 */
static string definesToSource(const map<string, string>& defines) {
    stringstream src;
    for (auto& d : defines)
        src << "#define " << d.first << " " << d.second << "\n";
    return src.str();
}

/* -------------------------------------------------------------------------- *
 *                          InteraxMultipoleForce                             *
 * -------------------------------------------------------------------------- */

class CommonCalcInteraxMultipoleForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const InteraxMultipoleForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, coreCharge1, alpha1, polarizability1;
        double charge2, coreCharge2, alpha2, polarizability2;
        int axisType1, multipoleZ1, multipoleX1, multipoleY1;
        int axisType2, multipoleZ2, multipoleX2, multipoleY2;
        bool alchemical1, alchemical2;
        vector<double> dipole1, dipole2, quadrupole1, quadrupole2;
        force.getParticleParameters(particle1, charge1, dipole1, quadrupole1, coreCharge1, alpha1, polarizability1,
                                    axisType1, multipoleZ1, multipoleX1, multipoleY1, alchemical1);
        force.getParticleParameters(particle2, charge2, dipole2, quadrupole2, coreCharge2, alpha2, polarizability2,
                                    axisType2, multipoleZ2, multipoleX2, multipoleY2, alchemical2);
        if (charge1 != charge2 || coreCharge1 != coreCharge2 || alpha1 != alpha2 || polarizability1 != polarizability2 ||
                axisType1 != axisType2 || alchemical1 != alchemical2) {
            return false;
        }
        for (int i = 0; i < dipole1.size(); ++i)
            if (dipole1[i] != dipole2[i])
                return false;
        for (int i = 0; i < quadrupole1.size(); ++i)
            if (quadrupole1[i] != quadrupole2[i])
                return false;
        return true;
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale;
        force.getExceptionParameters(index, particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double multipoleMultipoleScale1, dipoleMultipoleScale1, dipoleDipoleScale1;
        double multipoleMultipoleScale2, dipoleMultipoleScale2, dipoleDipoleScale2;
        force.getExceptionParameters(group1, particle1, particle2, multipoleMultipoleScale1, dipoleMultipoleScale1, dipoleDipoleScale1);
        force.getExceptionParameters(group2, particle1, particle2, multipoleMultipoleScale2, dipoleMultipoleScale2, dipoleDipoleScale2);
        return (multipoleMultipoleScale1 == multipoleMultipoleScale2 && dipoleMultipoleScale1 == dipoleMultipoleScale2 &&
                dipoleDipoleScale1 == dipoleDipoleScale2);
    }
private:
    const InteraxMultipoleForce& force;
};

class CommonCalcInteraxMultipoleForceKernel::TorquePostComputation : public ComputeContext::ForcePostComputation {
public:
    TorquePostComputation(CommonCalcInteraxMultipoleForceKernel& owner) : owner(owner) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        owner.addTorquesToForces();
        return 0.0;
    }
private:
    CommonCalcInteraxMultipoleForceKernel& owner;
};

CommonCalcInteraxMultipoleForceKernel::CommonCalcInteraxMultipoleForceKernel(const std::string& name, const Platform& platform, ComputeContext& cc, const System& system) :
        CalcInteraxMultipoleForceKernel(name, platform), cc(cc), system(system), hasInitializedKernels(false), multipolesAreValid(false), lastLambda(-1) {
}

void CommonCalcInteraxMultipoleForceKernel::initialize(const System& system, const InteraxMultipoleForce& force) {
    ContextSelector selector(cc);
    extrapolationCoefficients = force.getExtrapolationCoefficients();
    usePME = (force.getNonbondedMethod() == InteraxMultipoleForce::PME);
    coreDamping = force.getCoreDamping();
    coreDampingScale = force.getCoreDampingScale();
    mutualDampingScale = force.getMutualDampingScale();
    alchemicalParameter = force.getAlchemicalParameterName();

    // Initialize particle parameters. The parameters of alchemical particles are scaled on the device (execute()), from
    // the unscaled values stored in the base arrays.

    numParticles = force.getNumParticles();
    vector<double> coreChargeVec, valenceChargeVec, alphaVec, polarizabilityVec;
    vector<double> localDipolesVec, localQuadrupolesVec;
    vector<mm_int4> multipoleParticlesVec;
    vector<int> isAlchemicalVec;
    vector<vector<int> > exclusions(numParticles);
    hasAlchemicalParticles = false;
    for (int i = 0; i < numParticles; i++) {
        double charge, coreCharge, alpha, polarizability;
        int axisType, atomX, atomY, atomZ;
        bool alchemical;
        vector<double> dipole, quadrupole;
        force.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, atomZ, atomX, atomY, alchemical);
        coreChargeVec.push_back(coreCharge);
        valenceChargeVec.push_back(charge-coreCharge);
        alphaVec.push_back(alpha);
        polarizabilityVec.push_back(polarizability);
        multipoleParticlesVec.push_back(mm_int4(atomX, atomY, atomZ, axisType));
        isAlchemicalVec.push_back(alchemical ? 1 : 0);
        hasAlchemicalParticles |= alchemical;
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(dipole[j]);
        localQuadrupolesVec.push_back(quadrupole[0]);
        localQuadrupolesVec.push_back(quadrupole[1]);
        localQuadrupolesVec.push_back(quadrupole[2]);
        localQuadrupolesVec.push_back(quadrupole[4]);
        localQuadrupolesVec.push_back(quadrupole[5]);
        exclusions[i].push_back(i);
    }
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    for (int i = numParticles; i < paddedNumAtoms; i++) {
        coreChargeVec.push_back(0);
        valenceChargeVec.push_back(0);
        alphaVec.push_back(0);
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(mm_int4(0, 0, 0, 0));
        isAlchemicalVec.push_back(0);
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
    }
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    baseCoreCharge.initialize(cc, paddedNumAtoms, elementSize, "baseCoreCharge");
    baseValenceCharge.initialize(cc, paddedNumAtoms, elementSize, "baseValenceCharge");
    basePolarizability.initialize(cc, paddedNumAtoms, elementSize, "basePolarizability");
    baseLocalDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "baseLocalDipoles");
    baseLocalQuadrupoles.initialize(cc, 5*paddedNumAtoms, elementSize, "baseLocalQuadrupoles");
    coreCharge.initialize(cc, paddedNumAtoms, elementSize, "coreCharge");
    valenceCharge.initialize(cc, paddedNumAtoms, elementSize, "valenceCharge");
    alpha.initialize(cc, paddedNumAtoms, elementSize, "alpha");
    polarizability.initialize(cc, paddedNumAtoms, elementSize, "polarizability");
    multipoleParticles.initialize<mm_int4>(cc, paddedNumAtoms, "multipoleParticles");
    isAlchemical.initialize<int>(cc, paddedNumAtoms, "isAlchemical");
    localDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "localDipoles");
    localQuadrupoles.initialize(cc, 5*paddedNumAtoms, elementSize, "localQuadrupoles");
    lastPositions.initialize(cc, cc.getPosq().getSize(), cc.getPosq().getElementSize(), "lastPositions");
    baseCoreCharge.upload(coreChargeVec, true);
    baseValenceCharge.upload(valenceChargeVec, true);
    alpha.upload(alphaVec, true);
    basePolarizability.upload(polarizabilityVec, true);
    multipoleParticles.upload(multipoleParticlesVec);
    isAlchemical.upload(isAlchemicalVec);
    baseLocalDipoles.upload(localDipolesVec, true);
    baseLocalQuadrupoles.upload(localQuadrupolesVec, true);
    
    // Create workspace arrays.
    
    labDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "dipole");
    labQuadrupoles[0].initialize(cc, paddedNumAtoms, elementSize, "qXX");
    labQuadrupoles[1].initialize(cc, paddedNumAtoms, elementSize, "qXY");
    labQuadrupoles[2].initialize(cc, paddedNumAtoms, elementSize, "qXZ");
    labQuadrupoles[3].initialize(cc, paddedNumAtoms, elementSize, "qYY");
    labQuadrupoles[4].initialize(cc, paddedNumAtoms, elementSize, "qYZ");
    fracDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "fracDipoles");
    fracQuadrupoles.initialize(cc, 6*paddedNumAtoms, elementSize, "fracQuadrupoles");
    field.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "field");
    inducedField.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "inducedField");
    torque.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "torque");
    inducedDipole.initialize(cc, 3*paddedNumAtoms, elementSize, "inducedDipole");
    int numOrders = extrapolationCoefficients.size();
    extrapolatedDipole.initialize(cc, 3*numParticles*numOrders, elementSize, "extrapolatedDipole");
    extrapolatedPhi.initialize(cc, 10*numParticles*numOrders, elementSize, "extrapolatedPhi");
    cc.addAutoclearBuffer(field);
    cc.addAutoclearBuffer(torque);
    
    // Record exceptions and exclusions.
    
    vector<double> exceptionScaleVec[3];
    vector<mm_int2> exceptionAtomsVec;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale;
        force.getExceptionParameters(i, particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale);
        exclusions[particle1].push_back(particle2);
        exclusions[particle2].push_back(particle1);
        if (usePME || multipoleMultipoleScale != 0 || dipoleMultipoleScale != 0 || dipoleDipoleScale != 0) {
            exceptionAtomsVec.push_back(mm_int2(particle1, particle2));
            exceptionScaleVec[0].push_back(multipoleMultipoleScale);
            exceptionScaleVec[1].push_back(dipoleMultipoleScale);
            exceptionScaleVec[2].push_back(dipoleDipoleScale);
        }
    }
    if (exceptionAtomsVec.size() > 0) {
        exceptionAtoms.initialize<mm_int2>(cc, exceptionAtomsVec.size(), "exceptionAtoms");
        exceptionAtoms.upload(exceptionAtomsVec);
        for (int i = 0; i < 3; i++) {
            exceptionScales[i].initialize(cc, exceptionAtomsVec.size(), elementSize, "exceptionScales");
            exceptionScales[i].upload(exceptionScaleVec[i], true);
        }
    }
    
    // Create the kernels.

    map<string, string> defines = getPhysicsDefines();
    defines["HIPPO"] = "1";
    defines["NUM_ATOMS"] = cc.intToString(numParticles);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0);
    maxExtrapolationOrder = extrapolationCoefficients.size();
    defines["MAX_EXTRAPOLATION_ORDER"] = cc.intToString(maxExtrapolationOrder);
    stringstream coefficients;
    for (int i = 0; i < maxExtrapolationOrder; i++) {
        if (i > 0)
            coefficients << ",";
        double sum = 0;
        for (int j = i; j < maxExtrapolationOrder; j++)
            sum += extrapolationCoefficients[j];
        coefficients << cc.doubleToString(sum);
    }
    defines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    cutoff = force.getCutoffDistance();
    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(pmeAlpha, nx, ny, nz);
        if (nx == 0 || pmeAlpha == 0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, pmeAlpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = cc.findLegalFFTDimension(gridSizeX);
            gridSizeY = cc.findLegalFFTDimension(gridSizeY);
            gridSizeZ = cc.findLegalFFTDimension(gridSizeZ);
        } else {
            gridSizeX = cc.findLegalFFTDimension(nx);
            gridSizeY = cc.findLegalFFTDimension(ny);
            gridSizeZ = cc.findLegalFFTDimension(nz);
        }
        defines["EWALD_ALPHA"] = cc.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    ComputeProgram program = cc.compileProgram(CommonInteraxKernelSources::interaxMultipoleMultipoles, defines);
    alchemicalKernel = program->createKernel("applyAlchemicalScale");
    alchemicalKernel->addArg();
    alchemicalKernel->addArg(isAlchemical);
    alchemicalKernel->addArg(baseCoreCharge);
    alchemicalKernel->addArg(baseValenceCharge);
    alchemicalKernel->addArg(basePolarizability);
    alchemicalKernel->addArg(baseLocalDipoles);
    alchemicalKernel->addArg(baseLocalQuadrupoles);
    alchemicalKernel->addArg(coreCharge);
    alchemicalKernel->addArg(valenceCharge);
    alchemicalKernel->addArg(polarizability);
    alchemicalKernel->addArg(localDipoles);
    alchemicalKernel->addArg(localQuadrupoles);
    computeMomentsKernel = program->createKernel("computeLabFrameMoments");
    computeMomentsKernel->addArg(cc.getPosq());
    computeMomentsKernel->addArg(multipoleParticles);
    computeMomentsKernel->addArg(localDipoles);
    computeMomentsKernel->addArg(localQuadrupoles);
    computeMomentsKernel->addArg(labDipoles);
    computeMomentsKernel->addArg(labQuadrupoles[0]);
    computeMomentsKernel->addArg(labQuadrupoles[1]);
    computeMomentsKernel->addArg(labQuadrupoles[2]);
    computeMomentsKernel->addArg(labQuadrupoles[3]);
    computeMomentsKernel->addArg(labQuadrupoles[4]);
    recordInducedDipolesKernel = program->createKernel("recordInducedDipoles");
    recordInducedDipolesKernel->addArg(field);
    recordInducedDipolesKernel->addArg(inducedDipole);
    recordInducedDipolesKernel->addArg(polarizability);
    mapTorqueKernel = program->createKernel("mapTorqueToForce");
    mapTorqueKernel->addArg(cc.getLongForceBuffer());
    mapTorqueKernel->addArg(torque);
    mapTorqueKernel->addArg(cc.getPosq());
    mapTorqueKernel->addArg(multipoleParticles);
    program = cc.compileProgram(CommonInteraxKernelSources::interaxMultipoleInducedField, defines);
    initExtrapolatedKernel = program->createKernel("initExtrapolatedDipoles");
    initExtrapolatedKernel->addArg(inducedDipole);
    initExtrapolatedKernel->addArg(extrapolatedDipole);
    iterateExtrapolatedKernel = program->createKernel("iterateExtrapolatedDipoles");
    iterateExtrapolatedKernel->addArg();
    iterateExtrapolatedKernel->addArg(inducedDipole);
    iterateExtrapolatedKernel->addArg(extrapolatedDipole);
    iterateExtrapolatedKernel->addArg(inducedField);
    iterateExtrapolatedKernel->addArg(polarizability);
    computeExtrapolatedKernel = program->createKernel("computeExtrapolatedDipoles");
    computeExtrapolatedKernel->addArg(inducedDipole);
    computeExtrapolatedKernel->addArg(extrapolatedDipole);
    polarizationEnergyKernel = program->createKernel("computePolarizationEnergy");
    polarizationEnergyKernel->addArg(cc.getEnergyBuffer());
    polarizationEnergyKernel->addArg(inducedDipole);
    polarizationEnergyKernel->addArg(field);

    // Set up PME.
    
    if (usePME) {

        // Create required data structures.

        int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);
        int gridElements = gridSizeX*gridSizeY*roundedZSize;
        pmeGrid1.initialize(cc, gridElements, elementSize, "pmeGrid1");
        pmeGrid2.initialize(cc, gridElements, 2*elementSize, "pmeGrid2");
        if (useFixedPointChargeSpreading()) {
            pmeGridLong.initialize(cc, 2*gridElements, sizeof(long long), "pmeGridLong");
            cc.addAutoclearBuffer(pmeGridLong);
        }
        else
            cc.addAutoclearBuffer(pmeGrid1);
        pmeBsplineModuliX.initialize(cc, gridSizeX, elementSize, "pmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cc, gridSizeY, elementSize, "pmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cc, gridSizeZ, elementSize, "pmeBsplineModuliZ");
        pmePhi.initialize(cc, 20*numParticles, elementSize, "pmePhi");
        pmePhidp.initialize(cc, 20*numParticles, elementSize, "pmePhidp");
        pmeCphi.initialize(cc, 10*numParticles, elementSize, "pmeCphi");

        // Create the PME kernels.

        map<string, string> pmeDefines;
        pmeDefines["HIPPO"] = "1";
        pmeDefines["EWALD_ALPHA"] = cc.doubleToString(pmeAlpha);
        pmeDefines["PME_ORDER"] = cc.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cc.intToString(numParticles);
        pmeDefines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        pmeDefines["EPSILON_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0);
        pmeDefines["GRID_SIZE_X"] = cc.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cc.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cc.intToString(gridSizeZ);
        pmeDefines["M_PI"] = cc.doubleToString(M_PI);
        pmeDefines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        pmeDefines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
        pmeDefines["MAX_EXTRAPOLATION_ORDER"] = cc.intToString(maxExtrapolationOrder);
        if (useFixedPointChargeSpreading())
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "";
        program = cc.compileProgram(CommonInteraxKernelSources::interaxMultipolePme, pmeDefines);
        pmeTransformMultipolesKernel = program->createKernel("transformMultipolesToFractionalCoordinates");
        pmeTransformMultipolesKernel->addArg(labDipoles);
        pmeTransformMultipolesKernel->addArg(labQuadrupoles[0]);
        pmeTransformMultipolesKernel->addArg(labQuadrupoles[1]);
        pmeTransformMultipolesKernel->addArg(labQuadrupoles[2]);
        pmeTransformMultipolesKernel->addArg(labQuadrupoles[3]);
        pmeTransformMultipolesKernel->addArg(labQuadrupoles[4]);
        pmeTransformMultipolesKernel->addArg(fracDipoles);
        pmeTransformMultipolesKernel->addArg(fracQuadrupoles);
        for (int i = 0; i < 3; i++)
            pmeTransformMultipolesKernel->addArg();
        pmeTransformPotentialKernel = program->createKernel("transformPotentialToCartesianCoordinates");
        pmeTransformPotentialKernel->addArg();
        pmeTransformPotentialKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeTransformPotentialKernel->addArg();
        pmeSpreadFixedMultipolesKernel = program->createKernel("gridSpreadFixedMultipoles");
        pmeSpreadFixedMultipolesKernel->addArg(cc.getPosq());
        pmeSpreadFixedMultipolesKernel->addArg(fracDipoles);
        pmeSpreadFixedMultipolesKernel->addArg(fracQuadrupoles);
        if (useFixedPointChargeSpreading())
            pmeSpreadFixedMultipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadFixedMultipolesKernel->addArg(pmeGrid1);
        pmeSpreadFixedMultipolesKernel->addArg(coreCharge);
        pmeSpreadFixedMultipolesKernel->addArg(valenceCharge);
        for (int i = 0; i < 6; i++)
            pmeSpreadFixedMultipolesKernel->addArg();
        pmeSpreadInducedDipolesKernel = program->createKernel("gridSpreadInducedDipoles");
        pmeSpreadInducedDipolesKernel->addArg(cc.getPosq());
        pmeSpreadInducedDipolesKernel->addArg(inducedDipole);
        if (useFixedPointChargeSpreading())
            pmeSpreadInducedDipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadInducedDipolesKernel->addArg(pmeGrid1);
        for (int i = 0; i < 6; i++)
            pmeSpreadInducedDipolesKernel->addArg();
        if (useFixedPointChargeSpreading()) {
            pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
            pmeFinishSpreadChargeKernel->addArg(pmeGridLong);
            pmeFinishSpreadChargeKernel->addArg(pmeGrid1);
        }
        pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
        pmeConvolutionKernel->addArg(pmeGrid2);
        pmeConvolutionKernel->addArg(pmeBsplineModuliX);
        pmeConvolutionKernel->addArg(pmeBsplineModuliY);
        pmeConvolutionKernel->addArg(pmeBsplineModuliZ); // HippoNonbondedForce passes the X moduli here
        for (int i = 0; i < 4; i++)
            pmeConvolutionKernel->addArg();
        pmeFixedPotentialKernel = program->createKernel("computeFixedPotentialFromGrid");
        pmeFixedPotentialKernel->addArg(pmeGrid1);
        pmeFixedPotentialKernel->addArg(pmePhi);
        pmeFixedPotentialKernel->addArg(field);
        pmeFixedPotentialKernel->addArg(cc.getPosq());
        pmeFixedPotentialKernel->addArg(labDipoles);
        for (int i = 0; i < 6; i++)
            pmeFixedPotentialKernel->addArg();
        pmeInducedPotentialKernel = program->createKernel("computeInducedPotentialFromGrid");
        pmeInducedPotentialKernel->addArg(pmeGrid1);
        pmeInducedPotentialKernel->addArg(extrapolatedPhi);
        pmeInducedPotentialKernel->addArg();
        pmeInducedPotentialKernel->addArg(pmePhidp);
        pmeInducedPotentialKernel->addArg(cc.getPosq());
        for (int i = 0; i < 6; i++)
            pmeInducedPotentialKernel->addArg();
        pmeFixedForceKernel = program->createKernel("computeFixedMultipoleForceAndEnergy");
        pmeFixedForceKernel->addArg(cc.getPosq());
        pmeFixedForceKernel->addArg(cc.getLongForceBuffer());
        pmeFixedForceKernel->addArg(torque);
        pmeFixedForceKernel->addArg(cc.getEnergyBuffer());
        pmeFixedForceKernel->addArg(labDipoles);
        pmeFixedForceKernel->addArg(coreCharge);
        pmeFixedForceKernel->addArg(valenceCharge);
        pmeFixedForceKernel->addArg(labQuadrupoles[0]);
        pmeFixedForceKernel->addArg(labQuadrupoles[1]);
        pmeFixedForceKernel->addArg(labQuadrupoles[2]);
        pmeFixedForceKernel->addArg(labQuadrupoles[3]);
        pmeFixedForceKernel->addArg(labQuadrupoles[4]);
        pmeFixedForceKernel->addArg(fracDipoles);
        pmeFixedForceKernel->addArg(fracQuadrupoles);
        pmeFixedForceKernel->addArg(pmePhi);
        pmeFixedForceKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeFixedForceKernel->addArg();
        pmeInducedForceKernel = program->createKernel("computeInducedDipoleForceAndEnergy");
        pmeInducedForceKernel->addArg(cc.getPosq());
        pmeInducedForceKernel->addArg(cc.getLongForceBuffer());
        pmeInducedForceKernel->addArg(torque);
        pmeInducedForceKernel->addArg(cc.getEnergyBuffer());
        pmeInducedForceKernel->addArg(labDipoles);
        pmeInducedForceKernel->addArg(coreCharge);
        pmeInducedForceKernel->addArg(valenceCharge);
        pmeInducedForceKernel->addArg(extrapolatedDipole);
        pmeInducedForceKernel->addArg(extrapolatedPhi);
        pmeInducedForceKernel->addArg(labQuadrupoles[0]);
        pmeInducedForceKernel->addArg(labQuadrupoles[1]);
        pmeInducedForceKernel->addArg(labQuadrupoles[2]);
        pmeInducedForceKernel->addArg(labQuadrupoles[3]);
        pmeInducedForceKernel->addArg(labQuadrupoles[4]);
        pmeInducedForceKernel->addArg(fracDipoles);
        pmeInducedForceKernel->addArg(fracQuadrupoles);
        pmeInducedForceKernel->addArg(inducedDipole);
        pmeInducedForceKernel->addArg(pmePhi);
        pmeInducedForceKernel->addArg(pmePhidp);
        pmeInducedForceKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeInducedForceKernel->addArg();
        pmeRecordInducedFieldDipolesKernel = program->createKernel("recordInducedFieldDipoles");
        pmeRecordInducedFieldDipolesKernel->addArg(pmePhidp);
        pmeRecordInducedFieldDipolesKernel->addArg(inducedField);
        pmeRecordInducedFieldDipolesKernel->addArg(inducedDipole);
        for (int i = 0; i < 3; i++)
            pmeRecordInducedFieldDipolesKernel->addArg();
        pmeSelfEnergyKernel = program->createKernel("calculateSelfEnergyAndTorque");
        pmeSelfEnergyKernel->addArg(torque);
        pmeSelfEnergyKernel->addArg(cc.getEnergyBuffer());
        pmeSelfEnergyKernel->addArg(labDipoles);
        pmeSelfEnergyKernel->addArg(coreCharge);
        pmeSelfEnergyKernel->addArg(valenceCharge);
        pmeSelfEnergyKernel->addArg(inducedDipole);
        pmeSelfEnergyKernel->addArg(labQuadrupoles[0]);
        pmeSelfEnergyKernel->addArg(labQuadrupoles[1]);
        pmeSelfEnergyKernel->addArg(labQuadrupoles[2]);
        pmeSelfEnergyKernel->addArg(labQuadrupoles[3]);
        pmeSelfEnergyKernel->addArg(labQuadrupoles[4]);

        // Initialize the B-spline moduli.

        double data[PmeOrder];
        double x = 0.0;
        data[0] = 1.0 - x;
        data[1] = x;
        for (int i = 2; i < PmeOrder; i++) {
            double denom = 1.0/i;
            data[i] = x*data[i-1]*denom;
            for (int j = 1; j < i; j++)
                data[i-j] = ((x+j)*data[i-j-1] + ((i-j+1)-x)*data[i-j])*denom;
            data[0] = (1.0-x)*data[0]*denom;
        }
        int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
        vector<double> bsplines_data(maxSize+1, 0.0);
        for (int i = 2; i <= PmeOrder+1; i++)
            bsplines_data[i] = data[i-2];
        for (int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? gridSizeX : dim == 1 ? gridSizeY : gridSizeZ);
            vector<double> moduli(ndata);

            // get the modulus of the discrete Fourier transform

            double factor = 2.0*M_PI/ndata;
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0;
                double ss = 0.0;
                for (int j = 1; j <= ndata; j++) {
                    double arg = factor*i*(j-1);
                    sc += bsplines_data[j]*cos(arg);
                    ss += bsplines_data[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }

            // Fix for exponential Euler spline interpolation failure.

            double eps = 1.0e-7;
            if (moduli[0] < eps)
                moduli[0] = 0.9*moduli[1];
            for (int i = 1; i < ndata-1; i++)
                if (moduli[i] < eps)
                    moduli[i] = 0.9*(moduli[i-1]+moduli[i+1]);
            if (moduli[ndata-1] < eps)
                moduli[ndata-1] = 0.9*moduli[ndata-2];

            // Compute and apply the optimal zeta coefficient.

            int jcut = 50;
            for (int i = 1; i <= ndata; i++) {
                int k = i - 1;
                if (i > ndata/2)
                    k = k - ndata;
                double zeta;
                if (k == 0)
                    zeta = 1.0;
                else {
                    double sum1 = 1.0;
                    double sum2 = 1.0;
                    factor = M_PI*k/ndata;
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor+M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor-M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    zeta = sum2/sum1;
                }
                moduli[i-1] = moduli[i-1]*zeta*zeta;
            }
            if (cc.getUseDoublePrecision()) {
                if (dim == 0)
                    pmeBsplineModuliX.upload(moduli);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(moduli);
                else
                    pmeBsplineModuliZ.upload(moduli);
            }
            else {
                vector<float> modulif(ndata);
                for (int i = 0; i < ndata; i++)
                    modulif[i] = (float) moduli[i];
                if (dim == 0)
                    pmeBsplineModuliX.upload(modulif);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(modulif);
                else
                    pmeBsplineModuliZ.upload(modulif);
            }
        }
    }

    // Add the interaction to the default nonbonded kernel. With useOwnKernel, the force instead only registers its
    // cutoff and exclusions with the nonbonded utilities (so that they build the neighbor list it uses), and evaluates
    // the interaction with a kernel of its own, built from the same source (computeElectrostatics()).
    
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    useOwnKernel = force.getUseOwnKernel();
    electrostaticsArgs = {{&torque, "torqueBuffers", "mm_ulong", 1, false}, {&extrapolatedDipole, "extrapolatedDipole", "real", 1, true}};
    electrostaticsParams = {{&coreCharge, "coreCharge", "real", 1, true}, {&valenceCharge, "valenceCharge", "real", 1, true},
            {&alpha, "alpha", "real", 1, true}, {&labDipoles, "dipole", "real", 3, true}, {&inducedDipole, "inducedDipole", "real", 3, true},
            {&labQuadrupoles[0], "qXX", "real", 1, true}, {&labQuadrupoles[1], "qXY", "real", 1, true}, {&labQuadrupoles[2], "qXZ", "real", 1, true},
            {&labQuadrupoles[3], "qYY", "real", 1, true}, {&labQuadrupoles[4], "qYZ", "real", 1, true}};
    if (!useOwnKernel) {
        nb.setKernelSource(definesToSource(getPhysicsDefines())+CommonInteraxKernelSources::interaxMultipoleInteractionHeader+CommonInteraxKernelSources::interaxMultipoleNonbonded);
        for (auto& a : electrostaticsArgs)
            nb.addArgument(ComputeParameterInfo(*a.array, a.name, a.componentType, a.numComponents, a.constant));
        for (auto& p : electrostaticsParams)
            nb.addParameter(ComputeParameterInfo(*p.array, p.name, p.componentType, p.numComponents, p.constant));
    }
    map<string, string> replacements;
    replacements["ENERGY_SCALE_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0);
    replacements["MAX_EXTRAPOLATION_ORDER"] = cc.intToString(maxExtrapolationOrder);
    replacements["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    replacements["USE_EWALD"] = (usePME ? "1" : "0");
    replacements["PME_ALPHA"] = (usePME ? cc.doubleToString(pmeAlpha) : "0");
    replacements["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
    string interactionSource = cc.replaceStrings(CommonInteraxKernelSources::interaxMultipoleInteraction, replacements);
    if (useOwnKernel) {
        nb.addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
        electrostaticsSource = interactionSource;
        singlePairsPlaceholder.initialize<mm_int2>(cc, 1, "singlePairsPlaceholder");
    }
    else
        nb.addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, interactionSource, force.getForceGroup());
    nb.setUsePadding(false);
    
    // Create the kernel for computing exceptions.
    
    if (exceptionAtoms.isInitialized()) {
        replacements["COMPUTE_INTERACTION"] = interactionSource;
        string exceptionsSrc = CommonInteraxKernelSources::interaxMultipoleInteractionHeader+CommonInteraxKernelSources::interaxMultipoleNonbondedExceptions;
        exceptionsSrc = cc.replaceStrings(exceptionsSrc, replacements);
        defines["NUM_EXCEPTIONS"] = cc.intToString(exceptionAtoms.getSize());
        program = cc.compileProgram(exceptionsSrc, defines);
        computeExceptionsKernel = program->createKernel("computeNonbondedExceptions");
        computeExceptionsKernel->addArg(cc.getLongForceBuffer());
        computeExceptionsKernel->addArg(cc.getEnergyBuffer());
        computeExceptionsKernel->addArg(torque);
        computeExceptionsKernel->addArg(cc.getPosq());
        computeExceptionsKernel->addArg(exceptionAtoms);
        computeExceptionsKernel->addArg(exceptionScales[0]);
        computeExceptionsKernel->addArg(exceptionScales[1]);
        computeExceptionsKernel->addArg(exceptionScales[2]);
        computeExceptionsKernel->addArg(coreCharge);
        computeExceptionsKernel->addArg(valenceCharge);
        computeExceptionsKernel->addArg(alpha);
        computeExceptionsKernel->addArg(labDipoles);
        computeExceptionsKernel->addArg(inducedDipole);
        computeExceptionsKernel->addArg(labQuadrupoles[0]);
        computeExceptionsKernel->addArg(labQuadrupoles[1]);
        computeExceptionsKernel->addArg(labQuadrupoles[2]);
        computeExceptionsKernel->addArg(labQuadrupoles[3]);
        computeExceptionsKernel->addArg(labQuadrupoles[4]);
        computeExceptionsKernel->addArg(extrapolatedDipole);
        if (nb.getUseCutoff())
            for (int i = 0; i < 5; i++)
                computeExceptionsKernel->addArg();
    }
    cc.addForce(new ForceInfo(force));
    cc.addPostComputation(new TorquePostComputation(*this));
    fieldThreadBlockSize = max(32, cc.getNonbondedUtilities().getForceThreadBlockSize());
    forceThreadBlockSize = cc.getNonbondedUtilities().getForceThreadBlockSize();
}

void CommonCalcInteraxMultipoleForceKernel::createFieldKernel(const string& interactionSrc, vector<ComputeArray*> params,
            ComputeArray& fieldBuffer, ComputeKernel& kernel, ComputeKernel& exceptionKernel, ComputeArray& exceptionScale) {
    // Create the kernel source.

    map<string, string> replacements;
    replacements["COMPUTE_FIELD"] = interactionSrc;
    stringstream extraArgs, atomParams, loadLocal1, loadLocal2, load1, load2, load3;
    for (auto param : params) {
        string name = param->getName();
        bool isReal3 = (param->getSize() == cc.getPaddedNumAtoms()*3);
        string type = (isReal3 ? "real3" : "real");
        extraArgs << ", GLOBAL const real* RESTRICT " << name;
        atomParams << type << " " << name << ";\n";
        loadLocal1 << "localData[localAtomIndex]." << name << " = " << name << "1;\n";
        if (isReal3) {
            loadLocal2 << "localData[localAtomIndex]." << name << " = make_real3(" << name << "[3*j], " << name << "[3*j+1], " << name << "[3*j+2]);\n";
            load1 << type << " " << name << "1 = make_real3(" << name << "[3*atom1], " << name << "[3*atom1+1], " << name << "[3*atom1+2]);\n";
            load3 << type << " " << name << "2 = make_real3(" << name << "[3*atom2], " << name << "[3*atom2+1], " << name << "[3*atom2+2]);\n";
        }
        else {
            loadLocal2 << "localData[localAtomIndex]." << name << " = " << name << "[j];\n";
            load1 << type << " " << name << "1 = " << name << "[atom1];\n";
            load3 << type << " " << name << "2 = " << name << "[atom2];\n";
        }
        load2 << type << " " << name << "2 = localData[atom2]." << name << ";\n";
    }
    replacements["PARAMETER_ARGUMENTS"] = extraArgs.str();
    replacements["ATOM_PARAMETER_DATA"] = atomParams.str();
    replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
    replacements["LOAD_ATOM2_PARAMETERS"] = load2.str();
    replacements["LOAD_ATOM2_PARAMETERS_FROM_GLOBAL"] = load3.str();
    string src = CommonInteraxKernelSources::interaxMultipoleInteractionHeader+
                 cc.replaceStrings(CommonInteraxKernelSources::interaxMultipoleComputeField, replacements);

    // Set defines and create the kernel.

    map<string, string> defines = getPhysicsDefines();
    if (usePME) {
        defines["USE_CUTOFF"] = "1";
        defines["USE_PERIODIC"] = "1";
        defines["USE_EWALD"] = "1";
        defines["PME_ALPHA"] = cc.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
    }
    defines["WARPS_PER_GROUP"] = cc.intToString(fieldThreadBlockSize/ComputeContext::TileSize);
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(fieldThreadBlockSize);
    defines["CUTOFF"] = cc.doubleToString(cutoff);
    defines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
    defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
    defines["TILE_SIZE"] = cc.intToString(ComputeContext::TileSize);
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(cc.getNonbondedUtilities().getExclusionTiles().getSize());
    defines["NUM_EXCEPTIONS"] = cc.intToString(exceptionAtoms.isInitialized() ? exceptionAtoms.getSize() : 0);
    ComputeProgram program = cc.compileProgram(src, defines);
    kernel = program->createKernel("computeField");

    // Build the list of arguments.

    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    kernel->addArg(cc.getPosq());
    kernel->addArg(cc.getNonbondedUtilities().getExclusions());
    kernel->addArg(cc.getNonbondedUtilities().getExclusionTiles());
    kernel->addArg(fieldBuffer);
    if (nb.getUseCutoff()) {
        kernel->addArg(nb.getInteractingTiles());
        kernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 5; i++)
            kernel->addArg();
        kernel->addArg(maxTiles);
        kernel->addArg(nb.getBlockCenters());
        kernel->addArg(nb.getBlockBoundingBoxes());
        kernel->addArg(nb.getInteractingAtoms());
    }
    else
        kernel->addArg(maxTiles);
    for (auto param : params)
        kernel->addArg(*param);
    
    // If there are any exceptions, build the kernel and arguments to compute them.
    
    if (exceptionAtoms.isInitialized()) {
        exceptionKernel = program->createKernel("computeFieldExceptions");
        exceptionKernel->addArg(cc.getPosq());
        exceptionKernel->addArg(fieldBuffer);
        exceptionKernel->addArg(exceptionAtoms);
        exceptionKernel->addArg(exceptionScale);
        if (nb.getUseCutoff())
            for (int i = 0; i < 5; i++)
                exceptionKernel->addArg();
        for (auto param : params)
            exceptionKernel->addArg(*param);
    }
}

double CommonCalcInteraxMultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        
        // These kernels can't be compiled in initialize(), because the nonbonded utilities object
        // has not yet been initialized then.

        maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : cc.getNumAtomBlocks()*(cc.getNumAtomBlocks()+1)/2);
        createFieldKernel(CommonInteraxKernelSources::interaxMultipoleFixedField, {&coreCharge, &valenceCharge, &alpha, &labDipoles, &labQuadrupoles[0],
                &labQuadrupoles[1], &labQuadrupoles[2], &labQuadrupoles[3], &labQuadrupoles[4]}, field, fixedFieldKernel,
                fixedFieldExceptionKernel, exceptionScales[1]);
        createFieldKernel(CommonInteraxKernelSources::interaxMultipoleMutualField, {&alpha, &inducedDipole}, inducedField, mutualFieldKernel,
                mutualFieldExceptionKernel, exceptionScales[2]);
    }

    // Scale the parameters of the alchemical particles, if the alchemical parameter has changed.

    double lambda = (hasAlchemicalParticles ? context.getParameter(alchemicalParameter) : 1.0);
    if (lambda != lastLambda) {
        if (cc.getUseDoublePrecision())
            alchemicalKernel->setArg(0, lambda);
        else
            alchemicalKernel->setArg(0, (float) lambda);
        alchemicalKernel->execute(cc.getNumAtoms());
        lastLambda = lambda;
    }

    // Compute the lab frame moments.

    computeMomentsKernel->execute(cc.getNumAtoms());

    if (usePME) {
        // Compute reciprocal box vectors.
        
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        double determinant = a[0]*b[1]*c[2];
        double scale = 1.0/determinant;
        mm_double4 recipBoxVectors[3];
        recipBoxVectors[0] = mm_double4(b[1]*c[2]*scale, 0, 0, 0);
        recipBoxVectors[1] = mm_double4(-b[0]*c[2]*scale, a[0]*c[2]*scale, 0, 0);
        recipBoxVectors[2] = mm_double4((b[0]*c[1]-b[1]*c[0])*scale, -a[0]*c[1]*scale, a[0]*b[1]*scale, 0);
        if (cc.getUseDoublePrecision()) {
            mm_double4 boxVectors[] = {mm_double4(a[0], a[1], a[2], 0), mm_double4(b[0], b[1], b[2], 0), mm_double4(c[0], c[1], c[2], 0)};
            pmeConvolutionKernel->setArg(4, mm_double4(a[0], b[1], c[2], 0));
            for (int i = 0; i < 3; i++) {
                pmeTransformMultipolesKernel->setArg(8+i, recipBoxVectors[i]);
                pmeTransformPotentialKernel->setArg(2+i, recipBoxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(6+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(9+i, recipBoxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(3+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(6+i, recipBoxVectors[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectors[i]);
                pmeFixedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(8+i, recipBoxVectors[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectors[i]);
                pmeFixedForceKernel->setArg(16+i, recipBoxVectors[i]);
                pmeInducedForceKernel->setArg(20+i, recipBoxVectors[i]);
                pmeRecordInducedFieldDipolesKernel->setArg(3+i, recipBoxVectors[i]);
            }
        }
        else {
            mm_float4 boxVectors[] = {mm_float4(a[0], a[1], a[2], 0), mm_float4(b[0], b[1], b[2], 0), mm_float4(c[0], c[1], c[2], 0)};
            pmeConvolutionKernel->setArg(4, mm_float4(a[0], b[1], c[2], 0));
            mm_float4 recipBoxVectorsFloat[3];
            recipBoxVectorsFloat[0] = mm_float4((float) recipBoxVectors[0].x, 0, 0, 0);
            recipBoxVectorsFloat[1] = mm_float4((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0, 0);
            recipBoxVectorsFloat[2] = mm_float4((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z, 0);
            for (int i = 0; i < 3; i++) {
                pmeTransformMultipolesKernel->setArg(8+i, recipBoxVectorsFloat[i]);
                pmeTransformPotentialKernel->setArg(2+i, recipBoxVectorsFloat[i]);
                pmeSpreadFixedMultipolesKernel->setArg(6+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(9+i, recipBoxVectorsFloat[i]);
                pmeSpreadInducedDipolesKernel->setArg(3+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(6+i, recipBoxVectorsFloat[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectorsFloat[i]);
                pmeFixedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(8+i, recipBoxVectorsFloat[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectorsFloat[i]);
                pmeFixedForceKernel->setArg(16+i, recipBoxVectorsFloat[i]);
                pmeInducedForceKernel->setArg(20+i, recipBoxVectorsFloat[i]);
                pmeRecordInducedFieldDipolesKernel->setArg(3+i, recipBoxVectorsFloat[i]);
            }
        }

        // Reciprocal space calculation for electrostatics.
        
        pmeTransformMultipolesKernel->execute(cc.getNumAtoms());
        pmeSpreadFixedMultipolesKernel->execute(cc.getNumAtoms());
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        computeFFT(true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeFFT(false);
        pmeFixedPotentialKernel->execute(cc.getNumAtoms());
        pmeTransformPotentialKernel->setArg(0, pmePhi);
        pmeTransformPotentialKernel->execute(cc.getNumAtoms());
        pmeFixedForceKernel->execute(cc.getNumAtoms());
    }

    // Compute the field from fixed multipoles.

    updateFieldKernelMaxTiles();
    if (nb.getUseCutoff())
        setPeriodicBoxArgs(cc, fixedFieldKernel, 6);
    fixedFieldKernel->execute(nb.getNumForceThreadBlocks()*fieldThreadBlockSize, fieldThreadBlockSize);
    if (exceptionAtoms.isInitialized()) {
        if (nb.getUseCutoff())
            setPeriodicBoxArgs(cc, fixedFieldExceptionKernel, 4);
        fixedFieldExceptionKernel->execute(exceptionAtoms.getSize());
    }

    // Iterate the induced dipoles.

    computeExtrapolatedDipoles();

    // Add the polarization energy.

    if (includeEnergy)
        polarizationEnergyKernel->execute(cc.getNumAtoms());

    // Compute the forces due to the reciprocal space PME calculation for induced dipoles.

    if (usePME) {
        pmeTransformPotentialKernel->setArg(0, pmePhidp);
        pmeTransformPotentialKernel->execute(cc.getNumAtoms());
        pmeInducedForceKernel->execute(cc.getNumAtoms());
        pmeSelfEnergyKernel->execute(cc.getNumAtoms());
    }

    // Compute nonbonded exceptions.

    if (exceptionAtoms.isInitialized()) {
        if (nb.getUseCutoff())
            setPeriodicBoxArgs(cc, computeExceptionsKernel, 19);
        computeExceptionsKernel->execute(exceptionAtoms.getSize());
    }

    // Compute the real-space interactions, if this force has a kernel of its own for them (otherwise the default
    // nonbonded kernel computes them after this).

    if (useOwnKernel && (includeForces || includeEnergy))
        computeElectrostatics(includeForces, includeEnergy);

    // Record the current atom positions so we can tell later if they have changed.
    
    cc.getPosq().copyTo(lastPositions);
    multipolesAreValid = true;
    return 0.0;
}

void CommonCalcInteraxMultipoleForceKernel::computeInducedField(int optOrder) {
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    cc.clearBuffer(inducedField);
    updateFieldKernelMaxTiles();
    if (nb.getUseCutoff())
        setPeriodicBoxArgs(cc, mutualFieldKernel, 6);
    mutualFieldKernel->execute(nb.getNumForceThreadBlocks()*fieldThreadBlockSize, fieldThreadBlockSize);
    if (exceptionAtoms.isInitialized()) {
        if (nb.getUseCutoff())
            setPeriodicBoxArgs(cc, mutualFieldExceptionKernel, 4);
        mutualFieldExceptionKernel->execute(exceptionAtoms.getSize());
    }
    if (usePME) {
        if (useFixedPointChargeSpreading())
            cc.clearBuffer(pmeGridLong);
        else
            cc.clearBuffer(pmeGrid1);
        pmeSpreadInducedDipolesKernel->execute(cc.getNumAtoms());
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        computeFFT(true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeFFT(false);
        pmeInducedPotentialKernel->setArg(2, optOrder);
        pmeInducedPotentialKernel->execute(cc.getNumAtoms());
        pmeRecordInducedFieldDipolesKernel->execute(cc.getNumAtoms());
    }
}

void CommonCalcInteraxMultipoleForceKernel::computeExtrapolatedDipoles() {
    // Start by storing the direct dipoles as PT0

    recordInducedDipolesKernel->execute(cc.getNumAtoms());
    initExtrapolatedKernel->execute(extrapolatedDipole.getSize());

    // Recursively apply alpha.Tau to the µ_(n) components to generate µ_(n+1), and store the result

    for (int order = 1; order < maxExtrapolationOrder; ++order) {
        computeInducedField(order-1);
        iterateExtrapolatedKernel->setArg(0, order);
        iterateExtrapolatedKernel->execute(extrapolatedDipole.getSize());
    }
    
    // Take a linear combination of the µ_(n) components to form the total dipole

    computeExtrapolatedKernel->execute(extrapolatedDipole.getSize());
    computeInducedField(maxExtrapolationOrder-1);
}

void CommonCalcInteraxMultipoleForceKernel::addTorquesToForces() {
    mapTorqueKernel->execute(cc.getNumAtoms());
}

void CommonCalcInteraxMultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numParticles = cc.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
    else {
        vector<float> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
}

void CommonCalcInteraxMultipoleForceKernel::ensureMultipolesValid(ContextImpl& context) {
    if (multipolesAreValid) {
        int numParticles = cc.getNumAtoms();
        if (cc.getUseDoublePrecision()) {
            vector<mm_double4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
        else {
            vector<mm_float4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
    }
    double lambda = (hasAlchemicalParticles ? context.getParameter(alchemicalParameter) : 1.0);
    if (!multipolesAreValid || lambda != lastLambda)
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
}

void CommonCalcInteraxMultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numParticles = cc.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
    else {
        vector<float> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
}

void CommonCalcInteraxMultipoleForceKernel::copyParametersToContext(ContextImpl& context, const InteraxMultipoleForce& force) {
    // Make sure the new parameters are acceptable.
    
    ContextSelector selector(cc);
    if (force.getNumParticles() != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    vector<double> coreChargeVec, valenceChargeVec, alphaVec, polarizabilityVec;
    vector<double> localDipolesVec, localQuadrupolesVec;
    vector<mm_int4> multipoleParticlesVec;
    vector<int> isAlchemicalVec;
    for (int i = 0; i < numParticles; i++) {
        double charge, coreCharge, alpha, polarizability;
        int axisType, atomX, atomY, atomZ;
        bool alchemical;
        vector<double> dipole, quadrupole;
        force.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, alpha, polarizability, axisType, atomZ, atomX, atomY, alchemical);
        coreChargeVec.push_back(coreCharge);
        valenceChargeVec.push_back(charge-coreCharge);
        alphaVec.push_back(alpha);
        polarizabilityVec.push_back(polarizability);
        multipoleParticlesVec.push_back(mm_int4(atomX, atomY, atomZ, axisType));
        isAlchemicalVec.push_back(alchemical ? 1 : 0);
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(dipole[j]);
        localQuadrupolesVec.push_back(quadrupole[0]);
        localQuadrupolesVec.push_back(quadrupole[1]);
        localQuadrupolesVec.push_back(quadrupole[2]);
        localQuadrupolesVec.push_back(quadrupole[4]);
        localQuadrupolesVec.push_back(quadrupole[5]);
    }
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    for (int i = numParticles; i < paddedNumAtoms; i++) {
        coreChargeVec.push_back(0);
        valenceChargeVec.push_back(0);
        alphaVec.push_back(0);
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(mm_int4(0, 0, 0, 0));
        isAlchemicalVec.push_back(0);
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
    }
    baseCoreCharge.upload(coreChargeVec, true);
    baseValenceCharge.upload(valenceChargeVec, true);
    alpha.upload(alphaVec, true);
    basePolarizability.upload(polarizabilityVec, true);
    multipoleParticles.upload(multipoleParticlesVec);
    isAlchemical.upload(isAlchemicalVec);
    baseLocalDipoles.upload(localDipolesVec, true);
    baseLocalQuadrupoles.upload(localQuadrupolesVec, true);
    lastLambda = -1; // The scaled parameters must be recomputed.
    
    // Record the per-exception parameters. The particles of the exceptions cannot change, since they are the exclusions
    // of the neighbor list.

    if (exceptionAtoms.isInitialized()) {
        vector<double> exceptionScaleVec[3];
        for (int i = 0; i < force.getNumExceptions(); i++) {
            int particle1, particle2;
            double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale;
            force.getExceptionParameters(i, particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale);
            if (usePME || multipoleMultipoleScale != 0 || dipoleMultipoleScale != 0 || dipoleDipoleScale != 0) {
                exceptionScaleVec[0].push_back(multipoleMultipoleScale);
                exceptionScaleVec[1].push_back(dipoleMultipoleScale);
                exceptionScaleVec[2].push_back(dipoleDipoleScale);
            }
        }
        if (exceptionScaleVec[0].size() != exceptionAtoms.getSize())
            throw OpenMMException("updateParametersInContext: The number of exceptions has changed");
        for (int i = 0; i < 3; i++)
            exceptionScales[i].upload(exceptionScaleVec[i], true);
    }
    cc.invalidateMolecules();
    multipolesAreValid = false;
}

void CommonCalcInteraxMultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = pmeAlpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

map<string, string> CommonCalcInteraxMultipoleForceKernel::getPhysicsDefines() const {
    // Below these |t| (and |t| a r) the pair damping factors use their series (interaxMultipoleInteractionHeader.cc);
    // chosen where the series error equals the cancellation error of the closed form (~1e-6 in single, ~1e-12 in
    // double precision).
    map<string, string> defines;
    bool doublePrecision = cc.getUseDoublePrecision();
    defines["PAIR_DAMPING_MAX_T"] = cc.doubleToString(doublePrecision ? 0.03 : 0.1);
    defines["PAIR_DAMPING_MAX_T2"] = cc.doubleToString(doublePrecision ? 0.03 : 0.2);
    defines["PAIR_DAMPING_MAX_TX"] = cc.doubleToString(1.0);
    defines["MUTUAL_DAMPING_SCALE"] = cc.doubleToString(mutualDampingScale);
    if (mutualDampingScale != 1.0)
        defines["MUTUAL_DAMPING_SCALED"] = "1";
    if (coreDamping)
        defines["CORE_DAMPING"] = "1";
    defines["CORE_DAMPING_SCALE"] = cc.doubleToString(coreDampingScale);
    return defines;
}

void CommonCalcInteraxMultipoleForceKernel::updateFieldKernelMaxTiles() {
    // The neighbor list grows (and its arrays are reallocated) whenever the number of interacting tiles exceeds its
    // capacity. The field kernels skip every tile without exclusions when the tile count exceeds maxTiles, so maxTiles
    // must follow the current capacity, as in OpenMM's AMOEBA kernels. Argument 11 is maxTiles (see createFieldKernel).

    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    if (!nb.getUseCutoff())
        return;
    int size = nb.getInteractingTiles().getSize();
    if (size != maxTiles) {
        maxTiles = size;
        fixedFieldKernel->setArg(11, (unsigned int) maxTiles);
        mutualFieldKernel->setArg(11, (unsigned int) maxTiles);
    }
}

ComputeKernel& CommonCalcInteraxMultipoleForceKernel::getElectrostaticsKernel(bool includeForces, bool includeEnergy) {
    // The kernel of the real-space interactions for useOwnKernel: the same source, placeholders and defines that the
    // CUDA nonbonded utilities use for the default kernel when HippoNonbondedForce replaces its source
    // (CudaNonbondedUtilities::createInteractionKernel in OpenMM 8.1), for this force's parameters only. Its arguments
    // are those of the default kernel, so it walks the same neighbor list.

    int key = (includeForces ? 2 : 0) + (includeEnergy ? 1 : 0);
    auto existing = electrostaticsKernels.find(key);
    if (existing != electrostaticsKernels.end())
        return existing->second;
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream localData, args, load1, broadcast, declareLocal, loadLocal2, load2j, load2g, clearLocal, shuffle;
    int localDataSize = 0;
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    for (auto& p : electrostaticsParams) {
        string type = p.getType();
        if (p.numComponents == 1)
            localData << type << " " << p.name << ";\n";
        else
            for (int j = 0; j < p.numComponents; j++)
                localData << p.componentType << " " << p.name << "_" << suffixes[j] << ";\n";
        localDataSize += p.numComponents*elementSize;
        args << ", " << (p.constant ? "const " : "") << type << "* __restrict__ global_" << p.name;
        load1 << type << " " << p.name << "1 = global_" << p.name << "[atom1];\n";
        broadcast << type << " shfl" << p.name << ";\n";
        for (int j = 0; j < p.numComponents; j++) {
            if (p.numComponents == 1)
                broadcast << "shfl" << p.name << "=real_shfl(" << p.name << "1,j);\n";
            else
                broadcast << "shfl" << p.name << "." << suffixes[j] << "=real_shfl(" << p.name << "1." << suffixes[j] << ",j);\n";
        }
        declareLocal << type << " shfl" << p.name << ";\n";
        loadLocal2 << "shfl" << p.name << " = global_" << p.name << "[j];\n";
        load2j << type << " " << p.name << "2 = shfl" << p.name << ";\n";
        load2g << type << " " << p.name << "2 = global_" << p.name << "[atom2];\n";
        clearLocal << "shfl" << p.name << " = " << (p.numComponents == 1 ? "0" : "make_"+type+"(0)") << ";\n";
        if (p.numComponents == 1)
            shuffle << "shfl" << p.name << "=real_shfl(shfl" << p.name << ", tgx+1);\n";
        else
            for (int j = 0; j < p.numComponents; j++)
                shuffle << "shfl" << p.name << "." << suffixes[j] << "=real_shfl(shfl" << p.name << "." << suffixes[j] << ", tgx+1);\n";
    }
    for (auto& a : electrostaticsArgs)
        args << ", " << (a.constant ? "const " : "") << a.getType() << "* __restrict__ " << a.name;
    stringstream broadcastWarpData, shuffleWarpData;
    broadcastWarpData << "posq2.x = real_shfl(shflPosq.x, j);\nposq2.y = real_shfl(shflPosq.y, j);\n";
    broadcastWarpData << "posq2.z = real_shfl(shflPosq.z, j);\nposq2.w = real_shfl(shflPosq.w, j);\n" << broadcast.str();
    shuffleWarpData << "shflPosq.x = real_shfl(shflPosq.x, tgx+1);\nshflPosq.y = real_shfl(shflPosq.y, tgx+1);\n";
    shuffleWarpData << "shflPosq.z = real_shfl(shflPosq.z, tgx+1);\nshflPosq.w = real_shfl(shflPosq.w, tgx+1);\n";
    shuffleWarpData << "shflForce.x = real_shfl(shflForce.x, tgx+1);\nshflForce.y = real_shfl(shflForce.y, tgx+1);\n";
    shuffleWarpData << "shflForce.z = real_shfl(shflForce.z, tgx+1);\n" << shuffle.str();
    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = electrostaticsSource;
    replacements["ATOM_PARAMETER_DATA"] = localData.str();
    replacements["PARAMETER_ARGUMENTS"] = args.str();
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
    replacements["BROADCAST_WARP_DATA"] = broadcastWarpData.str();
    replacements["DECLARE_LOCAL_PARAMETERS"] = declareLocal.str();
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
    replacements["LOAD_ATOM2_PARAMETERS"] = load2j.str();
    replacements["LOAD_ATOM2_PARAMETERS_FROM_GLOBAL"] = load2g.str();
    replacements["CLEAR_LOCAL_PARAMETERS"] = clearLocal.str();
    replacements["INIT_DERIVATIVES"] = "";
    replacements["SAVE_DERIVATIVES"] = "";
    replacements["SHUFFLE_WARP_DATA"] = shuffleWarpData.str();
    map<string, string> defines;
    if (nb.getUseCutoff()) {
        defines["USE_CUTOFF"] = "1";
        defines["USE_NEIGHBOR_LIST"] = "1";
    }
    if (usePME)
        defines["USE_PERIODIC"] = "1";
    defines["USE_EXCLUSIONS"] = "1";
    defines["USE_SYMMETRIC"] = "1";
    defines["ENABLE_SHUFFLE"] = "1";
    if (includeForces)
        defines["INCLUDE_FORCES"] = "1";
    if (includeEnergy)
        defines["INCLUDE_ENERGY"] = "1";
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(forceThreadBlockSize);
    defines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
    defines["MAX_CUTOFF"] = cc.doubleToString(cutoff);
    defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
    defines["TILE_SIZE"] = cc.intToString(ComputeContext::TileSize);
    int numExclusionTiles = nb.getExclusionTiles().getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(numExclusionTiles);
    int numContexts = cc.getNumContexts();
    defines["FIRST_EXCLUSION_TILE"] = cc.intToString(cc.getContextIndex()*numExclusionTiles/numContexts);
    defines["LAST_EXCLUSION_TILE"] = cc.intToString((cc.getContextIndex()+1)*numExclusionTiles/numContexts);
    if ((localDataSize/4)%2 == 0 && !cc.getUseDoublePrecision())
        defines["PARAMETER_SIZE_IS_EVEN"] = "1";
    string src = definesToSource(getPhysicsDefines())+cc.replaceStrings(CommonInteraxKernelSources::interaxMultipoleInteractionHeader+
            CommonInteraxKernelSources::interaxMultipoleNonbonded, replacements);
    ComputeProgram program = cc.compileProgram(src, defines);
    ComputeKernel kernel = program->createKernel("computeNonbonded");
    kernel->addArg(cc.getLongForceBuffer());
    kernel->addArg(cc.getEnergyBuffer());
    kernel->addArg(cc.getPosq());
    kernel->addArg(nb.getExclusions());
    kernel->addArg(nb.getExclusionTiles());
    kernel->addArg((unsigned int) nb.getStartTileIndex());
    kernel->addArg((unsigned long long) nb.getNumTiles());
    if (nb.getUseCutoff()) {
        kernel->addArg(nb.getInteractingTiles());
        kernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 5; i++)
            kernel->addArg(); // Periodic box, set in computeElectrostatics()
        kernel->addArg((unsigned int) maxTiles);
        kernel->addArg(nb.getBlockCenters());
        kernel->addArg(nb.getBlockBoundingBoxes());
        kernel->addArg(nb.getInteractingAtoms());
        // The interaction does not support the separate pair list (as with HippoNonbondedForce, the nonbonded utilities
        // then never use it), so the kernel does not read the single pairs.
        kernel->addArg(0u);
        kernel->addArg(singlePairsPlaceholder);
    }
    for (auto& p : electrostaticsParams)
        kernel->addArg(*p.array);
    for (auto& a : electrostaticsArgs)
        kernel->addArg(*a.array);
    electrostaticsKernels[key] = kernel;
    return electrostaticsKernels[key];
}

void CommonCalcInteraxMultipoleForceKernel::computeElectrostatics(bool includeForces, bool includeEnergy) {
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    ComputeKernel& kernel = getElectrostaticsKernel(includeForces, includeEnergy);
    if (nb.getUseCutoff()) {
        setPeriodicBoxArgs(cc, kernel, 9);
        kernel->setArg(14, (unsigned int) maxTiles); // Arrays are bound by reference; maxTiles follows the capacity
    }
    kernel->execute(nb.getNumForceThreadBlocks()*forceThreadBlockSize, forceThreadBlockSize);
}
