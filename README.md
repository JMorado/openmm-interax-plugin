OpenMM InteraxFF Plugin
=======================

This plugin implements the intermolecular terms of the InteraxFF force field as five [OpenMM](https://openmm.org)
forces with Reference, CPU and CUDA implementations, serialization support, test cases and a Python API. It started
from the [OpenMM example plugin](https://github.com/openmm/openmmexampleplugin).

| Force | Energy |
|---|---|
| `InteraxDispersionForce` | Tang-Toennies damped C6/C8/C10 dispersion (XDM moments), with an optional long range correction |
| `InteraxExchangeForce` | exchange repulsion, proportional to the overlap of the Slater valence densities |
| `InteraxShortRangeForce` | short-range correction, linear and quadratic in the normalized overlap |
| `InteraxThreeBodyForce` | Axilrod-Teller-Muto dispersion between molecular (or atomic) sites |
| `InteraxMultipoleForce` | electrostatics (core + Slater valence charges, dipoles, quadrupoles) and polarization |

`InteraxMultipoleForce` is derived from OpenMM's `HippoNonbondedForce` (Reference and common kernels), without the
repulsion, dispersion and charge transfer of HIPPO, and adds:

* core damping: the permanent field that polarizes an atom is damped by the atom's own Slater density, so the
  polarization energy stays finite when atoms overlap (with HIPPO's bare core fields it diverges as
  -alpha q_core^2 / (2 r^4), which crashed liquid alcohols at full polarization strength);
* a damping scale kappa of the induced-dipole couplings;
* alchemical particles, whose multipoles and polarizabilities are scaled by `lambda_electrostatics`, in one force
  (HIPPO needs two copies mixed linearly);
* exact exponents: nearly equal exponents use a series instead of HIPPO's closed form, which cancels catastrophically,
  so the export no longer has to merge them (the merging caused a -0.17 kJ/mol difference from the PyTorch model for
  propanol);
* zero polarizabilities, and fixes of two HIPPO bugs (docs/upstream): undefined forces on ZOnly particles in the
  reference code, and the x B-spline moduli used for z in the GPU PME.

With HIPPO's options it reproduces HIPPO (tests/python/compare_multipole.py). Like HIPPO it replaces OpenMM's nonbonded
kernel on the GPU, so the export wraps it in a `CustomCVForce` (`electrostatics="interax"`).

Electrostatics and polarization can also stay in OpenMM's `HippoNonbondedForce` (`electrostatics="hippo"` in
`interax.openmm.create_system_from_topology`). Every force can decouple alchemical particles through a context
parameter (`lambda_sterics` by default, `lambda_electrostatics` for `InteraxMultipoleForce`). The formulas,
conventions and accuracy of the pair and three-body terms are in [docs/SPEC.md](docs/SPEC.md).


Building The Plugin
===================

The plugin is built with [CMake](http://www.cmake.org) against an installed OpenMM (8.1 or later):

1. Create a build directory and run `ccmake` or `cmake` in it, pointing at the source directory.
2. Set `OPENMM_DIR` to where OpenMM is installed (for a conda environment, the environment's root), and
   `CMAKE_INSTALL_PREFIX` to where the plugin should be installed. If it is not set, the plugin is installed next to
   OpenMM.
3. The CUDA platform is built when CMake finds the CUDA toolkit (`INTERAX_BUILD_CUDA_LIB`; set `CUDAToolkit_ROOT` if
   needed). The Python wrapper is built when it finds Python and SWIG (`PYTHON_EXECUTABLE`, `SWIG_EXECUTABLE`). Use the
   SWIG major version that OpenMM itself was built with.
4. Build with `make install`, then `make PythonInstall` for the Python wrapper.

For example, against the `interax` conda environment:

    cmake .. -DOPENMM_DIR=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=/path/to/openmm-interax \
             -DSWIG_EXECUTABLE=/path/to/swig -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python
    make -j16 install
    make PythonInstall

When the plugin is installed in its own prefix, load it at runtime and make its Python modules importable:

```python
import sys
import openmm

sys.path.insert(0, "/path/to/openmm-interax/python")
openmm.Platform.loadPluginsFromDirectory("/path/to/openmm-interax/lib/plugins")

import interaxplugin
import interaxplugin_convert
```

OpenMM compiles its CUDA kernels at runtime with NVRTC. If the driver is older than the NVRTC of the environment (the
error is `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), preload an NVRTC that matches the driver, e.g. the
`nvidia-cuda-nvrtc-cu12==12.2.140` wheel for a 535 driver: `LD_PRELOAD=.../libnvrtc.so.12`.


Using The Plugin
================

`interaxplugin_convert.convert_system(system)` takes a System built by the InteraxFF export and replaces its
dispersion, exchange repulsion, short-range correction and three-body forces by the plugin forces, keeping every other
force. With `overlap="exact"` (the default) the Slater overlaps are evaluated exactly; with `overlap="export"` they use
the export's Taylor scheme and the converted forces reproduce the export to rounding. With `copy=False` the System is
converted in place, which avoids copying the millions of triples of the exported three-body force.

`interaxplugin_convert.atomic_three_body(dispersion)` builds an atomic Axilrod-Teller-Muto force (one site per atom,
Tang-Toennies damping) from the exported dispersion. It is not a term of the trained InteraxFF models.

For alchemical decoupling, flag the solute's particles (and its three-body sites) as alchemical: the plugin forces then
scale their interactions with the rest of the System by the alchemical parameter. absolv's InteraxFF support
(`absolv.interax.apply_fep_interax`) does this for converted systems.


Test Cases
==========

Build the `test` target (`make test`, or `ctest` in the build directory) to run the C++ tests. As in OpenMM, the tests
of each force are written once in a shared header (`tests/TestInteraxDispersionForce.h`, `TestInteraxExchangeForce.h`,
`TestInteraxShortRangeForce.h`, `TestInteraxThreeBodyForce.h`, `TestInteraxMultipoleForce.h`) and run on every platform
through thin wrappers (`platforms/<platform>/tests/Test<Platform>Interax<Force>.cpp`), the CUDA tests in single, mixed
and double precision. They compare energies and forces with values computed from the documented formulas, check forces
against finite differences, cutoffs, periodic boxes, exclusions, alchemical scaling and parameter updates, and, with
HIPPO's options, compare `InteraxMultipoleForce` with `HippoNonbondedForce`. `TestInteraxExportValues` checks every
platform against energies and forces of the InteraxFF export stored in `tests/data`, and
`serialization/tests/TestSerializeInterax<Force>.cpp` the serialization of every force. The CPU pair forces support
rectangular boxes only (triclinic boxes raise an exception). The Python tests in `tests/python` need the plugin to be
installed:

| Test | What it checks |
|---|---|
| `test_export_values.py` | the stored export values, through the Python API (`pytest`) |
| `test_three_body.py` | the three-body force against an independent evaluation and finite differences (`pytest`) |
| `test_alchemical.py` | the alchemical scaling against decoupled systems |
| `compare_export.py` | freshly built InteraxFF systems against the export (needs InteraxFF and a model) |
| `compare_multipole.py` | `InteraxMultipoleForce` against HIPPO (or its Reference platform with core damping and kappa), and against finite differences |
| `compare_multipole_torch.py` | `InteraxMultipoleForce` against the PyTorch model (direct, mutual, core damping, kappa) |
| `check_multipole_lambda.py` | the end states of `lambda_electrostatics`, also through a `CustomCVForce` |

`generate_export_values.py` regenerates `tests/data` from the export, and `benchmark.py` and `benchmark_three_body.py`
compare the speed of the plugin with that of the export; `benchmark_multipole.py` compares `InteraxMultipoleForce`
with HIPPO (speed, mixed-precision accuracy, molecular dynamics and the alchemical HFE setup).


Implementation
==============

Each force has a Reference, a CPU and a common compute (CUDA) implementation:

* The pair forces share their math (`openmmapi/include/internal/InteraxPairTerms.h`) between the Reference and the CPU
  platforms, which vectorizes it with SSE or AVX (`CpuInteraxPairForceVec4.cpp`, `CpuInteraxPairForceAvx.cpp`, chosen
  at run time in `CpuInteraxPairForceFvec.cpp`, as OpenMM's `CpuCustomNonbondedForce`). On the GPU they add their interactions to OpenMM's nonbonded kernel
  (`platforms/common/src/kernels/interax*.cc`), so they share its neighbor list.
* The three-body force finds its triples from a neighbor list of the sites. On the GPU each thread block takes a site
  and loops over the pairs of its neighbors, loaded in tiles into local memory.

* `InteraxMultipoleForce` keeps HIPPO's structure: the Reference platform (also used by the CPU platform) is HIPPO's
  reference code, the GPU kernels are HIPPO's common kernels (`platforms/common/src/kernels/interaxMultipole*.cc`) with
  cuFFT for PME. The files derived from OpenMM keep the upstream code, names and comments wherever the code is the
  same, so that `diff` against OpenMM 8.1.2 (`plugins/amoeba`) shows only the InteraxFF changes, each marked at the point
  of change. Core damping adds (f - 1) times the full pair field to the field that polarizes each atom and the
  corresponding forces and torques; the series of the pair damping factors to t^6 is used where its error is below
  the cancellation error of the closed form (thresholds per precision in `CommonInteraxMultipoleKernel.cpp`).

The common kernels are encoded into the class `CommonInteraxKernelSources` at build time, as in the example plugin.
`third_party/pocketfft` is the header-only FFT (BSD licence) that OpenMM's own reference PME uses; it is only compiled
into the Reference platform.
