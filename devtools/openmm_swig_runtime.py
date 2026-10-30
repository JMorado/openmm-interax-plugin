"""Print the SWIG runtime version of OpenMM's Python module (the N of swig_runtime_data<N>), or nothing if unknown.

The plugin's Python module must be generated with a SWIG of the same runtime version: SWIG keeps its type tables per
runtime version, and with different versions the plugin's forces are not recognized as OpenMM Forces (TypeError in
System.addForce(), Force.setName(), ...). SWIG 4.0-4.2 use runtime 4, SWIG 4.3 and later runtime 5.
"""
import glob
import os
import re

try:
    import openmm

    for path in glob.glob(os.path.join(os.path.dirname(openmm.__file__), "_openmm*")):
        match = re.search(rb"swig_runtime_data(\d+)", open(path, "rb").read())
        if match:
            print(match.group(1).decode())
            break
except Exception:
    pass
