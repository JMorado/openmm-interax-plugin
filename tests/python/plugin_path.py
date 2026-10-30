"""
Where the Python tests and scripts find the plugin, InteraxFF and an InteraxFF model, from the environment only:

  INTERAX_PLUGIN_PREFIX  the plugin's install prefix (CMAKE_INSTALL_PREFIX: PREFIX/python, PREFIX/lib/plugins);
                         default OpenMM's own directory, the build's default install prefix
  INTERAX_REPO           an InteraxFF checkout, put on sys.path (not needed when InteraxFF is installed)
  INTERAX_MODEL          an InteraxFF model directory, for the comparisons on real systems; default
                         INTERAX_REPO/examples/<the script's model> when INTERAX_REPO is set

Importing this module puts the plugin's Python modules on sys.path and loads its platform libraries.
"""
import os
import sys

import openmm
import openmm.version

PREFIX = os.environ.get("INTERAX_PLUGIN_PREFIX") or os.path.dirname(openmm.version.openmm_library_path)
INTERAX_REPO = os.environ.get("INTERAX_REPO")

sys.path.insert(0, os.path.join(PREFIX, "python"))
_plugins = os.path.join(PREFIX, "lib", "plugins")
# OpenMM loads the libraries of its own plugin directory when it is imported
if os.path.realpath(_plugins) != os.path.realpath(openmm.Platform.getDefaultPluginsDirectory()):
    openmm.Platform.loadPluginsFromDirectory(_plugins)
if INTERAX_REPO:
    sys.path.insert(0, INTERAX_REPO)


def model(name):
    """The model directory: INTERAX_MODEL, else INTERAX_REPO/examples/name, else None."""
    if os.environ.get("INTERAX_MODEL"):
        return os.environ["INTERAX_MODEL"]
    return os.path.join(INTERAX_REPO, "examples", name) if INTERAX_REPO else None


def available_platforms():
    """The platforms on which the plugin's forces run: registered by OpenMM, with the plugin's kernels (the CUDA
    platform without libInteraxPluginCUDA, e.g. in a build without the CUDA toolkit, is left out) and a usable device."""
    import interaxplugin

    names = set()
    for i in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(i)
        system = openmm.System()
        system.addParticle(1.0)
        force = interaxplugin.InteraxDispersionForce()
        force.addParticle(0.1, 0.01, 0.001, 0.001, 0.03)
        system.addForce(force)
        try:
            openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
        except Exception:
            continue
        names.add(platform.getName())
    return names
