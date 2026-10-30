#---------------------------------------------------
# OpenMM InteraxFF Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(OpenMMInteraxPlugin)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Specify the C++ version we are building for.
SET(CMAKE_CXX_STANDARD 11)
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Set flags for linking on mac
IF(APPLE)
    SET(CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    SET(CMAKE_INSTALL_PREFIX ${OPENMM_DIR} CACHE PATH "Where to install the plugin" FORCE)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# Libraries find the core plugin library and OpenMM where they are installed.
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${OPENMM_DIR}/lib")
# The test executables link OpenMM's platform libraries, whose own dependencies (e.g. nvrtc) are next to OpenMM.
IF(NOT APPLE)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${OPENMM_DIR}/lib -Wl,-rpath,${OPENMM_DIR}/lib")
ENDIF(NOT APPLE)

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(INTERAX_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)

# Set the library name
SET(INTERAX_LIBRARY_NAME InteraxPlugin)
SET(SHARED_INTERAX_TARGET ${INTERAX_LIBRARY_NAME})

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${INTERAX_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Build stamp: the plugin version and the git commit of the source at CMake time ("-dirty" with uncommitted changes),
# returned by InteraxPlugin::getPluginVersion() and getPluginGitCommit(). Re-run CMake before a release build.
SET(INTERAX_PLUGIN_VERSION "0.2.0")
EXECUTE_PROCESS(COMMAND git describe --always --dirty --abbrev=12
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE INTERAX_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
IF(NOT INTERAX_GIT_COMMIT)
    SET(INTERAX_GIT_COMMIT "unknown")
ENDIF()
MESSAGE(STATUS "InteraxFF plugin ${INTERAX_PLUGIN_VERSION}, commit ${INTERAX_GIT_COMMIT}")
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/src/InteraxPluginVersion.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/InteraxPluginVersion.cpp)
SET(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}/InteraxPluginVersion.cpp)

# Create the library.

ADD_LIBRARY(${SHARED_INTERAX_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_INTERAX_TARGET}
    PROPERTIES COMPILE_FLAGS "-DINTERAX_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_INTERAX_TARGET} OpenMM)
INSTALL(TARGETS ${SHARED_INTERAX_TARGET} DESTINATION lib)

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing

ENABLE_TESTING()
ADD_SUBDIRECTORY(serialization/tests)
# The platform tests share tests/TestInteraxExportValues.h and the export values in tests/data.
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/tests)
ADD_DEFINITIONS(-DINTERAX_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

# Build the implementations for different platforms

ADD_SUBDIRECTORY(platforms/reference)
ADD_SUBDIRECTORY(platforms/cpu)
ADD_SUBDIRECTORY(platforms/common)

FIND_PACKAGE(CUDAToolkit QUIET)
SET(INTERAX_BUILD_CUDA_LIB ${CUDAToolkit_FOUND} CACHE BOOL "Build implementation for CUDA")
IF(INTERAX_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(INTERAX_BUILD_CUDA_LIB)

# Build the Python API

FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(INTERAX_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(INTERAX_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF(INTERAX_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(INTERAX_BUILD_PYTHON_WRAPPERS)
