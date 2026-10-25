# HippoNonbondedForce (CUDA/OpenCL): PME uses the x B-spline moduli for the z axis (NaN or wrong energies in non-cubic grids)

## Summary

In the Common (CUDA/OpenCL) implementation of `HippoNonbondedForce`, the PME convolution kernel receives the
B-spline moduli of the x axis in place of those of the z axis:
- **Grids with gridSizeZ = gridSizeX** (e.g. cubic boxes): the result is right.
- **Otherwise:** the convolution uses wrong moduli. When gridSizeZ/2 + 1 > gridSizeX, it also reads past the end of
  the x array, which gives NaN energies and forces.

## Versions

- Observed with OpenMM 8.1.2 (conda-forge) on an RTX 2080 Ti, Linux.
- The code is unchanged on `master` (checked against commit 74912095, 2025-11-11).
- `AmoebaMultipoleForce` passes the right array, so only HIPPO is affected.

## Location

`plugins/amoeba/platforms/common/src/AmoebaCommonKernels.cpp`, `CommonCalcHippoNonbondedForceKernel::initialize()`
(lines 2739-2741 on master):

```cpp
pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
pmeConvolutionKernel->addArg(pmeGrid2);
pmeConvolutionKernel->addArg(pmeBsplineModuliX);
pmeConvolutionKernel->addArg(pmeBsplineModuliY);
pmeConvolutionKernel->addArg(pmeBsplineModuliX);   // should be pmeBsplineModuliZ
```

`reciprocalConvolution` (`multipolePme.cc`) takes `(pmeGrid, pmeBsplineModuliX, pmeBsplineModuliY, pmeBsplineModuliZ,
...)` and reads `pmeBsplineModuliZ[kz]` for `kz` up to `GRID_SIZE_Z/2`. The AMOEBA kernel in the same file (line 772
on master) passes `pmeBsplineModuliZ`.

## Reproduction

`hippo_moduli_repro.py` (attached, OpenMM only) builds 300 point charges with HIPPO PME. It compares the Reference and
CUDA (double precision) energies for a cubic box and for one twice as long in z:

```
$ python hippo_moduli_repro.py
OpenMM 8.1.2
box [3.0, 3.0, 3.0]: Reference -7537.592168  CUDA -7537.592329  difference -1.614e-04 kJ/mol
box [3.0, 3.0, 6.0]: Reference -3376.691154  CUDA nan  difference +nan kJ/mol
```

## Suggested fix

Pass `pmeBsplineModuliZ` as the fourth argument. With this change, in a copy of the HIPPO common code (our plugin), the
reproduction's elongated box gives -3376.691425 kJ/mol on CUDA, against -3376.691154 on Reference.
