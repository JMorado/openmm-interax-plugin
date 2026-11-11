OpenMM InteraxFF Plugin
=======================

[![CI](https://github.com/JMorado/openmm-interax-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/JMorado/openmm-interax-plugin/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![OpenMM](https://img.shields.io/badge/OpenMM-%E2%89%A5%208.1-blue)
![Platforms](https://img.shields.io/badge/platforms-Reference%20%7C%20CPU%20%7C%20CUDA-informational)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

This plugin implements the intermolecular terms of the InteraxFF force field as five [OpenMM](https://openmm.org)
forces, with Reference, CPU and CUDA implementations, serialization support, test cases and a Python API. It is
structured like the [OpenMM example plugin](https://github.com/openmm/openmmexampleplugin), from which it started.

| Force | Energy |
|---|---|
| `InteraxDispersionForce` | Tang-Toennies damped C6/C8/C10 pair dispersion from XDM moments, with an optional long range correction |
| `InteraxExchangeForce` | exchange repulsion, proportional to the overlap of the Slater valence densities of each pair |
| `InteraxShortRangeForce` | short-range correction, linear and quadratic in the normalized Slater overlap |
| `InteraxThreeBodyForce` | Axilrod-Teller-Muto three-body dispersion between atomic or molecular sites |
| `InteraxMultipoleForce` | electrostatics of core charges, Slater valence charges, dipoles and quadrupoles, and polarization |

Lengths are in nm, energies in kJ/mol and charges in e, as everywhere in OpenMM. The energy expressions, parameters
and conventions of every force are given in [docs/SPEC.md](docs/SPEC.md) and in the documentation of each class in
`openmmapi/include`.

`InteraxMultipoleForce` is derived from OpenMM's `HippoNonbondedForce`, without HIPPO's repulsion, dispersion and charge
transfer. It adds core damping (the permanent field that polarizes an atom is damped by the atom's own Slater density,
which keeps the polarization energy finite when atoms overlap), a damping scale of the couplings between induced
dipoles, alchemical scaling of multipoles and polarizabilities, an exact evaluation of the damping factors for nearly
equal exponents, and zero polarizabilities. With HIPPO's options it reproduces `HippoNonbondedForce`.

Every force can decouple alchemical particles through a context parameter (`lambda_sterics` by default,
`lambda_electrostatics` for `InteraxMultipoleForce`).


Building The Plugin
===================

The plugin is built with [CMake](https://cmake.org) against an installed OpenMM. It needs:

* OpenMM 8.1 or later, with its headers and libraries (a conda environment with the `openmm` package will do);
* CMake 3.17 or later and a C++11 compiler;
* the CUDA toolkit, for the CUDA platform (optional);
* Python and SWIG, for the Python wrapper (optional). SWIG must generate the runtime version of OpenMM's own Python
  module: SWIG 4.0-4.2 (runtime 4) for the conda-forge builds of OpenMM 8.1, SWIG 4.3 or later (runtime 5) for OpenMM
  builds made with it. With a mismatch the plugin's forces are not OpenMM forces to Python (`TypeError` in
  `System.addForce()`), so CMake compares the two and stops on a mismatch.

The steps are:

1. Create a build directory and run `cmake` (or `ccmake`) in it, pointing at the source directory.
2. Set `OPENMM_DIR` to where OpenMM is installed (for a conda environment, the environment's root), and
   `CMAKE_INSTALL_PREFIX` to where the plugin should be installed. If it is not set, the plugin is installed next to
   OpenMM: its platform libraries go into OpenMM's plugin directory, which OpenMM loads when it is imported, and its
   Python modules into the Python environment, so that nothing has to be set at runtime.
3. The CUDA platform is built when CMake finds the CUDA toolkit (`INTERAX_BUILD_CUDA_LIB`; set `CUDAToolkit_ROOT` if
   needed). The Python wrapper is built when CMake finds Python and SWIG (`INTERAX_BUILD_PYTHON_WRAPPERS`, with
   `PYTHON_EXECUTABLE` and `SWIG_EXECUTABLE`).
4. Run `make install` to build and install the libraries, then `make PythonInstall` to build and install the Python
   wrapper.

For example, into the active conda environment:

    mkdir build && cd build
    cmake .. -DOPENMM_DIR=$CONDA_PREFIX -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python
    make -j8 install
    make PythonInstall
    python -c "import interaxplugin; print(interaxplugin.__version__)"

To keep several builds side by side, give each its own prefix (`-DCMAKE_INSTALL_PREFIX=/path/to/openmm-interax`). The
Python modules then go into `PREFIX/python`, and the plugin has to be loaded at runtime (see below).

OpenMM compiles its CUDA kernels at runtime with NVRTC. If the driver is older than the NVRTC of the environment (the
error is `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), preload an NVRTC that matches the driver, for example from the
`nvidia-cuda-nvrtc-cu12` wheel of the driver's CUDA version: `LD_PRELOAD=.../libnvrtc.so.12`.

Versions
--------

The version of the plugin is set once, in the `PROJECT()` command of `CMakeLists.txt`, and copied from there to the
library and the Python package. Releases are tagged `vX.Y.Z` in git. Every build also records the git commit it was
built from (`git describe --always --dirty`, refreshed at every build), so that two builds of the same version can be
told apart:

| | Version | Git commit |
|---|---|---|
| C++ | `InteraxPlugin::getPluginVersion()` | `InteraxPlugin::getPluginGitCommit()` |
| Python | `interaxplugin.__version__` | `interaxplugin.getPluginGitCommit()` |

`interaxplugin_convert.build_info()` returns both, for example `0.1.0 (v0.1.0)`.


Using The Plugin
================

Installed next to OpenMM (the default), the plugin is loaded with OpenMM and `import interaxplugin` is all it takes.
A build in its own prefix is loaded first:

```python
import sys
import openmm

PREFIX = "/path/to/openmm-interax"  # CMAKE_INSTALL_PREFIX
sys.path.insert(0, f"{PREFIX}/python")
openmm.Platform.loadPluginsFromDirectory(f"{PREFIX}/lib/plugins")

import interaxplugin             # the forces
import interaxplugin_forcefield  # their ForceField (ffxml) generators
import interaxplugin_convert     # conversion of Systems built by InteraxFF
```

The forces are used like any other OpenMM force. For example, the atomic three-body dispersion of three atoms:

```python
import openmm
import interaxplugin

system = openmm.System()
force = interaxplugin.InteraxThreeBodyForce()
force.setCutoffDistance(1.2)
force.setDampingMethod(interaxplugin.InteraxThreeBodyForce.TangToennies)
for i in range(3):
    system.addParticle(39.9)
    # particles, polarizability (nm^3), frequency (kJ/mol), width (nm), molecule
    force.addSite([i], 1.64e-3, 1.7e3, 0.03, i)
system.addForce(force)
context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
context.setPositions([[0, 0, 0], [0.38, 0, 0], [0.19, 0.33, 0]])
print(context.getState(getEnergy=True).getPotentialEnergy())
```

As for OpenMM's own plugins, `System.getForce()` returns a plain `Force` in Python: each force class has
`isinstance(force)` and `cast(force)` to test for it and recover it. The forces can be serialized with `XmlSerializer`
once `interaxplugin` has been imported.

Creating Systems
----------------

**From force field files.** Importing `interaxplugin_forcefield` registers OpenMM `ForceField` generators for the
ffxml tags `<InteraxMultipoleForce>`, `<InteraxDispersionForce>`, `<InteraxExchangeForce>`, `<InteraxShortRangeForce>`
and `<InteraxThreeBodyForce>`, so that `ForceField.createSystem()` builds the plugin forces as for any OpenMM force
field. Parameters are per atom type (the arguments of each force's `addParticle()`, in OpenMM units); multipole frames
are given by the atom types of the z, x and y atoms, as in OpenMM's AMOEBA and HIPPO files; every intramolecular pair
is excluded. The module docstring lists the tags and their settings. A water model, for example:

```python
import io
from openmm import app

WATER = """<ForceField>
 <AtomTypes>
  <Type name="w-O" class="w-O" element="O" mass="15.999"/>
  <Type name="w-H" class="w-H" element="H" mass="1.008"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="w-O"/><Atom name="H1" type="w-H"/><Atom name="H2" type="w-H"/>
   <Bond atomName1="O" atomName2="H1"/><Bond atomName1="O" atomName2="H2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond type1="w-O" type2="w-H" length="0.09572" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle type1="w-H" type2="w-O" type3="w-H" angle="1.82421813418" k="836.8"/>
 </HarmonicAngleForce>
 <InteraxMultipoleForce extrapolationCoefficients="1.0">
  <Atom type="w-O" charge="-0.8" coreCharge="6.0" alpha="40.0" polarizability="0.0009" axisType="1"
        d0="0" d1="0" d2="0.003" q11="1e-4" q12="0" q13="0" q21="0" q22="-6e-5" q23="0" q31="0" q32="0" q33="-4e-5"
        zAtomType="w-H" xAtomType="w-H"/>
  <Atom type="w-H" charge="0.4" coreCharge="1.0" alpha="45.0" polarizability="0.0003" axisType="0"
        d0="0" d1="0" d2="-0.002" q11="0" q12="0" q13="0" q21="0" q22="0" q23="0" q31="0" q32="0" q33="0"
        zAtomType="w-O" xAtomType="w-H"/>
 </InteraxMultipoleForce>
 <InteraxDispersionForce dampingScale="1.0">
  <Atom type="w-O" m1="0.12" m2="0.02" m3="0.005" alpha="0.0012" width="0.03"/>
  <Atom type="w-H" m1="0.03" m2="0.004" m3="0.001" alpha="0.0003" width="0.025"/>
 </InteraxDispersionForce>
 <InteraxExchangeForce>
  <Atom type="w-O" amplitude="3.0" width="0.03"/>
  <Atom type="w-H" amplitude="1.2" width="0.025"/>
 </InteraxExchangeForce>
</ForceField>"""

forcefield = app.ForceField(io.StringIO(WATER))
topology = app.Topology()
chain = topology.addChain()
positions = []
for k in range(2):
    residue = topology.addResidue("HOH", chain)
    o = topology.addAtom("O", app.element.oxygen, residue)
    h1 = topology.addAtom("H1", app.element.hydrogen, residue)
    h2 = topology.addAtom("H2", app.element.hydrogen, residue)
    topology.addBond(o, h1)
    topology.addBond(o, h2)
    positions += [[0.3 * k, 0, 0], [0.3 * k + 0.0957, 0, 0], [0.3 * k - 0.024, 0.0927, 0]]
system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)  # app.PME with a periodic box
```

The parameters of this water are illustrative only.

**From InteraxFF.** InteraxFF (`interax.openmm.create_system_from_topology`) builds Systems with
`InteraxMultipoleForce` and `InteraxThreeBodyForce`, and with its pair terms as `CustomNonbondedForce`s.
`interaxplugin_convert.convert_system(system)` replaces those custom forces by `InteraxDispersionForce`,
`InteraxExchangeForce` and `InteraxShortRangeForce`, which are faster and evaluate the Slater overlaps exactly, and
keeps every other force:

```python
from interax.openmm import create_system_from_topology

system = interaxplugin_convert.convert_system(create_system_from_topology(topology, force_field), copy=False)
```

With `overlap="exact"` (the default) the Slater overlaps are evaluated exactly; with `overlap="export"` they use the
Taylor expansions of the InteraxFF implementation, which the converted forces then reproduce to rounding. With
`copy=False` the System is converted in place.

For alchemical decoupling, flag the solute's particles (and its three-body sites) as alchemical: the plugin forces then
scale their interactions with the rest of the System by the alchemical parameter.


Test Cases
==========

Build the `test` target (`make test`, or `ctest` in the build directory) to run the C++ tests. As in OpenMM, the tests
of each force are written once in a shared header (`tests/TestInterax<Force>.h`) and run on every platform through thin
wrappers (`platforms/<platform>/tests/Test<Platform>Interax<Force>.cpp`), the CUDA tests in single, mixed and double
precision. They compare energies and forces with values computed from the energy expressions, check forces against
finite differences, cutoffs, periodic and triclinic boxes, exclusions, alchemical scaling and parameter updates, and
compare `InteraxMultipoleForce` with `HippoNonbondedForce` for HIPPO's options. `TestInteraxExportValues` checks every
platform against energies and forces of the InteraxFF implementation stored in `tests/data`, and
`serialization/tests/TestSerializeInterax<Force>.cpp` the serialization of every force.

The Python tests in `tests/python` need the plugin to be installed and run with `pytest`. They find the plugin through
`INTERAX_PLUGIN_PREFIX` (unset: installed next to OpenMM), and InteraxFF and a model through `INTERAX_REPO` (an InteraxFF
checkout) and `INTERAX_MODEL` (a model directory, with `nagl_model.pt`, `per_atomic_number_model.pt` and
`force_field_state.pt`), as `tests/python/plugin_path.py` describes. The comparisons with InteraxFF run only when both
are set:

    export INTERAX_PLUGIN_PREFIX=/path/to/openmm-interax INTERAX_REPO=/path/to/InteraxFF INTERAX_MODEL=/path/to/model
    cd tests/python && python -m pytest

| Test | What it checks |
|---|---|
| `test_export_values.py` | the stored InteraxFF values (`tests/data`), through the Python API |
| `test_three_body.py` | the three-body force against an independent evaluation and finite differences |
| `test_softcore.py` | the soft core of the pair forces |
| `test_forcefield.py` | the ffxml generators against forces built with the API |
| `test_alchemical.py` | the alchemical scaling against decoupled systems (needs InteraxFF) |
| `test_multipole_features.py` | the options of `InteraxMultipoleForce` on every platform (needs InteraxFF) |

The other scripts in `tests/python` compare the plugin with InteraxFF (`compare_export.py`, `check_multipole_lambda.py`),
regenerate `tests/data` (`generate_export_values.py`), and benchmark the forces (`benchmark.py`,
`benchmark_three_body.py`).

GitHub Actions (`.github/workflows/ci.yml`) builds the plugin against the current conda-forge OpenMM and runs the
Reference, CPU and serialization tests and the Python tests on every push and pull request.
`devtools/run_all_tests.sh BUILD_DIR PREFIX [GPU]` builds, installs and runs every suite on a GPU machine, CUDA
included.


Implementation
==============

Each force has a Reference, a CPU and a common compute implementation; the common compute kernels
(`platforms/common`) are compiled for the CUDA platform. The plugin has no OpenCL platform library.

* The pair forces share their math (`openmmapi/include/internal/InteraxPairTerms.h`) between the Reference and the CPU
  platforms, which vectorizes it with SSE or AVX (`CpuInteraxPairForceVec4.cpp`, `CpuInteraxPairForceAvx.cpp`, chosen
  at run time, as in OpenMM's `CpuCustomNonbondedForce`). On the GPU they add their interactions to OpenMM's nonbonded
  kernel (`platforms/common/src/kernels/interax*.cc`) and share its neighbor list.
* The three-body force finds its triples from neighbor lists of the sites. On the GPU each site keeps a half list (the
  sites above it within the cutoff plus a 0.1 nm skin), rebuilt when a site has moved by more than half the skin; at
  every evaluation the neighbors within the cutoff are compacted into tiles, and a thread block per site loops over
  the pairs of tiles, skipping those whose bounding boxes are more than the cutoff apart. Each triple is computed once.
* `InteraxMultipoleForce` keeps HIPPO's structure: its Reference platform (also used by the CPU platform) is derived
  from HIPPO's reference code, and its GPU kernels from HIPPO's common kernels
  (`platforms/common/src/kernels/interaxMultipole*.cc`), with cuFFT for PME. The files derived from OpenMM keep the
  upstream code, names and comments where the code is the same, so that `diff` against OpenMM 8.1.2
  (`plugins/amoeba`) shows the changes of this plugin, each marked at the point of change.

The common kernels are encoded into the class `CommonInteraxKernelSources` at build time, as in the example plugin.
`third_party/pocketfft` is the header-only FFT (BSD license) that OpenMM's reference PME uses; it is compiled into the
Reference platform only.


License
=======

The plugin is distributed under the MIT license, stated in the header of each source file; the files derived from
OpenMM keep OpenMM's copyright notices. `third_party/pocketfft` is distributed under the BSD license.
