OpenMM InteraxFF Plugin
=======================

[![CI](https://github.com/JMorado/openmm-interax-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/JMorado/openmm-interax-plugin/actions/workflows/ci.yml)
![OpenMM](https://img.shields.io/badge/OpenMM-%E2%89%A5%208.1-blue)
![Platforms](https://img.shields.io/badge/platforms-Reference%20%7C%20CPU%20%7C%20CUDA-informational)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Force field files](https://img.shields.io/badge/ffxml-ForceField.createSystem-success)

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
   OpenMM: its platform libraries in OpenMM's plugin directory, which OpenMM loads when it is imported, and its
   Python modules in the Python environment, so nothing has to be set at runtime.
3. The CUDA platform is built when CMake finds the CUDA toolkit (`INTERAX_BUILD_CUDA_LIB`; set `CUDAToolkit_ROOT` if
   needed). The Python wrapper is built when it finds Python and SWIG (`PYTHON_EXECUTABLE`, `SWIG_EXECUTABLE`). Use the
   SWIG with the runtime version of OpenMM's Python module: for OpenMM 8.1, SWIG 4.0-4.2 (runtime 4); SWIG 4.3 and
   later generate runtime 5, whose plugin forces OpenMM's Python API rejects (`TypeError` in `System.addForce()` or
   `Force.setName()`). CMake compares the two and stops on a mismatch.
4. Build with `make install`, then `make PythonInstall` for the Python wrapper.

For example, into the active conda environment:

    cmake .. -DOPENMM_DIR=$CONDA_PREFIX -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python
    make -j16 install
    make PythonInstall
    python -c "import interaxplugin_convert; print(interaxplugin_convert.build_info())"

To keep several builds side by side, give each its own prefix (`-DCMAKE_INSTALL_PREFIX=/path/to/openmm-interax`):
the Python modules then go into `PREFIX/python`, and the plugin is loaded at runtime:

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

Installed next to OpenMM (the default), `import interaxplugin` is all it takes. A build in its own prefix is loaded
first (`PREFIX` is its `CMAKE_INSTALL_PREFIX`):

```python
import sys
import openmm

PREFIX = "/path/to/openmm-interax"  # CMAKE_INSTALL_PREFIX
sys.path.insert(0, f"{PREFIX}/python")
openmm.Platform.loadPluginsFromDirectory(f"{PREFIX}/lib/plugins")

import interaxplugin             # the forces
import interaxplugin_forcefield  # their ForceField (ffxml) generators
import interaxplugin_convert     # conversion of exported InteraxFF Systems

print(interaxplugin_convert.build_info())  # version and git commit of the build
```

Creating Systems
----------------

**From force field files.** Importing `interaxplugin_forcefield` registers OpenMM `ForceField` generators for the
ffxml tags `<InteraxMultipoleForce>`, `<InteraxDispersionForce>`, `<InteraxExchangeForce>`, `<InteraxShortRangeForce>`
and `<InteraxThreeBodyForce>`, so that `ForceField.createSystem()` builds the plugin forces from scratch, as for any
OpenMM force field. Parameters are per atom type (the arguments of each force's `addParticle()`, in OpenMM units);
multipole frames are given by the atom types of the z, x and y atoms, as in OpenMM's AMOEBA and HIPPO files; every
intramolecular pair is excluded. The module docstring lists the tags and their settings. A water model, for example:

```python
import io
import numpy as np
from openmm import app, unit

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
# an H-bonded dimer: about -18 kJ/mol
```

The parameters of this water are illustrative. InteraxFF writes such files itself, one per molecule, from its
predicted parameters, with a residue template generator (`interax.openmm_forcefield`, in the way openmmforcefields
parameterizes small molecules):

```python
import torch
from openff.toolkit import Molecule, Topology
from openff.units import unit as off_unit
from interax.force_field import ForceField
from interax.models import NAGLModel, PerAtomicNumberModel
from interax.openmm_forcefield import create_system

MODEL = "/shared/joaomorado/repos/InteraxFF/examples/models_v3R4bfdE1MQsce07fFIA"
nagl = NAGLModel.load(f"{MODEL}/nagl_model.pt")
ff = ForceField(nagl, PerAtomicNumberModel.load(f"{MODEL}/per_atomic_number_model.pt", device="cpu"))
ff.load_state_dict(torch.load(f"{MODEL}/force_field_state.pt", map_location="cpu"), strict=False)

water, methanol = Molecule.from_smiles("O"), Molecule.from_smiles("CO")
for molecule in (water, methanol):
    molecule.generate_conformers(n_conformers=1)
topology = Topology.from_molecules([water, methanol])  # periodic: set topology.box_vectors (PME)
xyz = topology.get_positions().m_as("nanometer")
xyz[water.n_atoms:] += [0.35, 0.0, 0.0]               # both molecules start at the origin
topology.set_positions(xyz * off_unit.nanometer)

system, omm_topology, positions = create_system(topology, ff, nagl)
```

Every term of the model is included, the three-body dispersion too (molecular: one site per molecule, cheap in
InteraxThreeBodyForce); `components=[...]` selects terms, for instance for an energy decomposition. `create_system` returns the OpenMM Topology and positions with the virtual sites (lone pairs, bond multipoles) added.
For a Topology from elsewhere (a PDB file), use the template generator with OpenMM's own tools:

```python
from openmm import app
from interax.openmm_forcefield import interax_forcefield

forcefield = interax_forcefield(ff, nagl, molecules=[water, methanol])  # an openmm.app.ForceField
modeller = app.Modeller(omm_topology, positions)                        # e.g. app.PDBFile(...).topology, .positions
modeller.addExtraParticles(forcefield)                                  # adds the virtual sites the templates define
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)  # app.PME for a periodic box
```

The valence terms, constraints and intramolecular nonbonded interactions come from the base force field (OpenFF Sage
by default, `base_force_field=`), as in the InteraxFF export; pass `constraints=None` to `createSystem`. Models with
terms the plugin does not implement raise `NotImplementedError` (see `interax/openmm_forcefield.py`).

**From an exported System.** `interaxplugin_convert.convert_system(system)` takes a System built by the InteraxFF export and replaces its
dispersion, exchange repulsion, short-range correction and three-body forces by the plugin forces, keeping every other
force. With `overlap="exact"` (the default) the Slater overlaps are evaluated exactly; with `overlap="export"` they use
the export's Taylor scheme and the converted forces reproduce the export to rounding. With `copy=False` the System is
converted in place, which avoids copying the millions of triples of the exported three-body force. This is the route
`examples/simulate_liquid.py` of InteraxFF takes; it gives the same System as the force field route above, with the
virtual sites after all atoms:

```python
from interax.openmm import create_system_from_topology, extend_positions_for_sites

exported = create_system_from_topology(topology, ff, nagl, electrostatics="interax")
system = interaxplugin_convert.convert_system(exported, copy=False)
positions = extend_positions_for_sites(topology.get_positions().to_openmm(), system)
```

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
`serialization/tests/TestSerializeInterax<Force>.cpp` the serialization of every force. Every platform supports
triclinic boxes (OpenMM's reduced form). GitHub Actions (`.github/workflows/ci.yml`) builds the plugin against conda-forge OpenMM and runs the
Reference, CPU and serialization tests and the force field file test on every push and pull request;
`devtools/run_all_tests.sh BUILD_DIR PREFIX [GPU]` builds, installs and runs every suite on a GPU node (CUDA, and the
InteraxFF tests against the build). The Python tests in `tests/python` need the plugin to be installed:

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
