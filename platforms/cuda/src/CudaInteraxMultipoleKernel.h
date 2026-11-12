#ifndef OPENMM_CUDAINTERAXMULTIPOLEKERNEL_H_
#define OPENMM_CUDAINTERAXMULTIPOLEKERNEL_H_

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

#include "CommonInteraxMultipoleKernel.h"
#include "openmm/cuda/CudaContext.h"
#include <cufft.h>

namespace InteraxPlugin {

/**
 * This kernel is invoked by InteraxMultipoleForce to calculate the forces acting on the system and the energy of the system.
 * It is the common implementation, with cuFFT for the FFTs of PME.
 */
class CudaCalcInteraxMultipoleForceKernel : public CommonCalcInteraxMultipoleForceKernel {
public:
    CudaCalcInteraxMultipoleForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system) :
            CommonCalcInteraxMultipoleForceKernel(name, platform, cu, system), hasInitializedFFT(false) {
    }
    ~CudaCalcInteraxMultipoleForceKernel();
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
    void computeFFT(bool forward);
    /**
     * Get whether charge spreading should be done in fixed point.
     */
    bool useFixedPointChargeSpreading() const {
        return cc.getUseDoublePrecision();
    }
private:
    bool hasInitializedFFT;
    cufftHandle fftForward, fftBackward;
};

} // namespace InteraxPlugin

#endif /*OPENMM_CUDAINTERAXMULTIPOLEKERNEL_H_*/
