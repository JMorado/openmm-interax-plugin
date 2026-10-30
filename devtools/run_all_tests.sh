#!/usr/bin/env bash
# Build the plugin in BUILD_DIR, install it to PREFIX, and run every test suite: the C++ tests (ctest: Reference, CPU,
# CUDA single/mixed/double), the plugin's Python tests, and InteraxFF's tests (which use the plugin through
# INTERAX_PLUGIN_PREFIX). Prints one summary line per suite and exits non-zero if any failed.
#
# Usage: devtools/run_all_tests.sh BUILD_DIR PREFIX [GPU_INDEX]
#   BUILD_DIR  a configured CMake build directory of this repository (see README, "Building The Plugin")
#   PREFIX     the install prefix it was configured with (CMAKE_INSTALL_PREFIX)
#   GPU_INDEX  the CUDA device for the GPU tests (default 0)
#
# Environment (defaults for metro05): INTERAX_ENV (the interax conda environment), PLUGIN_ENV (the environment with
# the compilers, CMake and the CUDA toolkit), SWIG_PYTHONPATH (the pip-installed SWIG), NVRTC_PRELOAD (an NVRTC that
# matches the driver), INTERAX_REPO. For a scheduled run, prefer a time when no GPU jobs run on the node.
set -u
BUILD_DIR=${1:?usage: run_all_tests.sh BUILD_DIR PREFIX [GPU_INDEX]}
PREFIX=${2:?usage: run_all_tests.sh BUILD_DIR PREFIX [GPU_INDEX]}
GPU=${3:-0}
INTERAX_ENV=${INTERAX_ENV:-/shared/joaomorado/micromamba/envs/interax}
PLUGIN_ENV=${PLUGIN_ENV:-/shared/joaomorado/micromamba/envs/ommplugin}
SWIG_PYTHONPATH=${SWIG_PYTHONPATH:-/shared/joaomorado/opt/swig-4.2.1}
NVRTC_PRELOAD=${NVRTC_PRELOAD:-/shared/joaomorado/opt/nvrtc-12.2/nvidia/cuda_nvrtc/lib/libnvrtc.so.12}
INTERAX_REPO=${INTERAX_REPO:-/shared/joaomorado/repos/InteraxFF}
SOURCE_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR=${LOG_DIR:-$BUILD_DIR/test-logs/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$LOG_DIR"
status=0

summary() {  # name, exit code, log
    if [ "$2" -eq 0 ]; then echo "PASS  $1  ($3)"; else echo "FAIL  $1  ($3)"; status=1; fi
}

echo "InteraxFF plugin tests: $(git -C "$SOURCE_DIR" describe --always --dirty --abbrev=12), logs in $LOG_DIR"

# Build and install (the Python wrapper too)
(
    export PATH=$PLUGIN_ENV/bin:$INTERAX_ENV/bin:$PATH PYTHONPATH=$SWIG_PYTHONPATH
    cd "$BUILD_DIR" && cmake . && make -j16 install && rm -rf python/build && make PythonInstall
) > "$LOG_DIR/build.log" 2>&1
rc=$?; summary "build + install" $rc "$LOG_DIR/build.log"
[ $status -ne 0 ] && exit 1

# C++ tests
(cd "$BUILD_DIR" && LD_PRELOAD=$NVRTC_PRELOAD CUDA_VISIBLE_DEVICES=$GPU "$PLUGIN_ENV/bin/ctest" -j6 --timeout 1500) > "$LOG_DIR/ctest.log" 2>&1
rc=$?; summary "C++ (ctest): $(grep -o '[0-9]*% tests passed.*' "$LOG_DIR/ctest.log")" $rc "$LOG_DIR/ctest.log"

# Plugin Python tests
(cd "$SOURCE_DIR/tests/python" && INTERAX_PLUGIN_PREFIX=$PREFIX LD_PRELOAD=$NVRTC_PRELOAD CUDA_VISIBLE_DEVICES=$GPU \
    "$INTERAX_ENV/bin/python" -m pytest -q -p no:cacheprovider -o addopts="" .) > "$LOG_DIR/pytest_plugin.log" 2>&1
rc=$?; summary "plugin Python: $(tail -1 "$LOG_DIR/pytest_plugin.log")" $rc "$LOG_DIR/pytest_plugin.log"

# InteraxFF tests against this build
(cd "$INTERAX_REPO" && INTERAX_PLUGIN_PREFIX=$PREFIX LD_PRELOAD=$NVRTC_PRELOAD CUDA_VISIBLE_DEVICES=$GPU \
    "$INTERAX_ENV/bin/python" -m pytest interax/tests -q -p no:cacheprovider -o addopts="") > "$LOG_DIR/pytest_interax.log" 2>&1
rc=$?; summary "InteraxFF: $(tail -1 "$LOG_DIR/pytest_interax.log")" $rc "$LOG_DIR/pytest_interax.log"

exit $status
