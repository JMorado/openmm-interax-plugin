OpenMM InteraxFF Plugin
=======================

This plugin implements the intermolecular terms of the InteraxFF force field, other than electrostatics and
polarization, as four [OpenMM](https://openmm.org) forces with Reference, CPU and CUDA implementations, serialization
support, test cases and a Python API. It started from the
[OpenMM example plugin](https://github.com/openmm/openmmexampleplugin).

| Force | Energy |
|---|---|
| `InteraxDispersionForce` | Tang-Toennies damped C6/C8/C10 dispersion (XDM moments), with an optional long range correction |
| `InteraxExchangeForce` | exchange repulsion, proportional to the overlap of the Slater valence densities |
| `InteraxShortRangeForce` | short-range correction, linear and quadratic in the normalized overlap |
| `InteraxThreeBodyForce` | Axilrod-Teller-Muto dispersion between molecular (or atomic) sites |

Electrostatics and polarization stay in OpenMM's `HippoNonbondedForce`, as in the InteraxFF export
(`interax.openmm.create_system_from_topology`). Every force can decouple alchemical particles through a context
parameter (`lambda_sterics` by default). The formulas, conventions and accuracy of every term are in
[docs/SPEC.md](docs/SPEC.md).


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

Build the `test` target (`make test`, or `ctest` in the build directory) to run the C++ tests: every platform is
checked against energies and forces of the InteraxFF export stored in `tests/data`, the CUDA tests in single, mixed and
double precision. The Python tests in `tests/python` need the plugin to be installed:

| Test | What it checks |
|---|---|
| `test_export_values.py` | the stored export values, through the Python API (`pytest`) |
| `test_three_body.py` | the three-body force against an independent evaluation and finite differences (`pytest`) |
| `test_alchemical.py` | the alchemical scaling against decoupled systems |
| `compare_export.py` | freshly built InteraxFF systems against the export (needs InteraxFF and a model) |

`generate_export_values.py` regenerates `tests/data` from the export, and `benchmark.py` and `benchmark_three_body.py`
compare the speed of the plugin with that of the export.


Implementation
==============

Each force has a Reference, a CPU and a common compute (CUDA) implementation:

* The pair forces share their math (`openmmapi/include/internal/InteraxPairTerms.h`) between the Reference and the CPU
  platforms, which vectorizes it with SSE or AVX. On the GPU they add their interactions to OpenMM's nonbonded kernel
  (`platforms/common/src/kernels/interax*.cc`), so they share its neighbor list.
* The three-body force finds its triples from a neighbor list of the sites. On the GPU each thread block takes a site
  and loops over the pairs of its neighbors, loaded in tiles into local memory.

The common kernels are encoded into the class `CommonInteraxKernelSources` at build time, as in the example plugin.
