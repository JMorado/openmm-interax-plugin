set(WRAP_FILE InteraxPluginWrapper.cpp)
set(MODULE_NAME interaxplugin)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "-I${CMAKE_SOURCE_DIR}/openmmapi/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/interaxplugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/interaxplugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(INTERAXPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(INTERAXPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
set(INTERAXPLUGIN_INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/interaxplugin_convert.py ${CMAKE_CURRENT_BINARY_DIR}/interaxplugin_convert.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/interaxplugin_forcefield.py ${CMAKE_CURRENT_BINARY_DIR}/interaxplugin_forcefield.py COPYONLY)
# Installed into OpenMM's directory (the default prefix; for conda, the environment), the module goes into the Python
# environment, next to OpenMM's, and needs no path. Installed into its own prefix (side-by-side builds), it goes into
# PREFIX/python, which then has to be put on sys.path (and PREFIX/lib/plugins loaded).
get_filename_component(_INSTALL_PREFIX_REAL "${CMAKE_INSTALL_PREFIX}" REALPATH)
get_filename_component(_OPENMM_DIR_REAL "${OPENMM_DIR}" REALPATH)
if(_INSTALL_PREFIX_REAL STREQUAL _OPENMM_DIR_REAL)
    set(PIP_TARGET_ARGS "")
else()
    set(PIP_TARGET_ARGS --target "${CMAKE_INSTALL_PREFIX}/python")
endif()
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install --no-deps --no-build-isolation ${PIP_TARGET_ARGS} --upgrade .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
