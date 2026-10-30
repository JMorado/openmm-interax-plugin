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
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install --no-deps --no-build-isolation --target "${CMAKE_INSTALL_PREFIX}/python" --upgrade .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
