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

#include "CudaInteraxMultipoleKernel.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

/* -------------------------------------------------------------------------- *
 *                          InteraxMultipoleForce                             *
 * -------------------------------------------------------------------------- */

CudaCalcInteraxMultipoleForceKernel::~CudaCalcInteraxMultipoleForceKernel() {
    ContextSelector selector(cc);
    if (hasInitializedFFT) {
        cufftDestroy(fftForward);
        cufftDestroy(fftBackward);
    }
}

void CudaCalcInteraxMultipoleForceKernel::initialize(const System& system, const InteraxMultipoleForce& force) {
    CommonCalcInteraxMultipoleForceKernel::initialize(system, force);
    if (usePME) {
        ContextSelector selector(cc);
        cufftResult result = cufftPlan3d(&fftForward, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? CUFFT_D2Z : CUFFT_R2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        result = cufftPlan3d(&fftBackward, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? CUFFT_Z2D : CUFFT_C2R);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
        // Run the FFTs on the stream of the context, like the other kernels.

        CudaContext& cu = dynamic_cast<CudaContext&>(cc);
        cufftSetStream(fftForward, cu.getCurrentStream());
        cufftSetStream(fftBackward, cu.getCurrentStream());
        hasInitializedFFT = true;
    }
}

void CudaCalcInteraxMultipoleForceKernel::computeFFT(bool forward) {
    CudaArray& grid1 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid1);
    CudaArray& grid2 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid2);
    if (forward) {
        if (cc.getUseDoublePrecision())
            cufftExecD2Z(fftForward, (double*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer());
        else
            cufftExecR2C(fftForward, (float*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer());
    }
    else {
        if (cc.getUseDoublePrecision())
            cufftExecZ2D(fftBackward, (double2*) grid2.getDevicePointer(), (double*) grid1.getDevicePointer());
        else
            cufftExecC2R(fftBackward, (float2*) grid2.getDevicePointer(), (float*) grid1.getDevicePointer());
    }
}
