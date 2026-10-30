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
# Run it with the build tools on PATH (CMake, the compilers, the CUDA toolkit). Optional environment:
#   PYTHON          the Python with OpenMM and pytest for the tests (default: python on PATH); the InteraxFF suite needs
#                   InteraxFF's dependencies in it too
#   INTERAX_REPO    an InteraxFF checkout: its test suite runs against this build, and the plugin's comparisons with the
#                   InteraxFF export run (skipped without it)
#   INTERAX_MODEL   the model of those comparisons (default: INTERAX_REPO/examples, see tests/python/plugin_path.py)
#   SWIG_PYTHONPATH a pip-installed SWIG to put on PYTHONPATH for the build (one whose runtime matches OpenMM's module)
#   NVRTC_PRELOAD   an NVRTC library to preload for the CUDA tests (when the toolkit's is newer than the driver)
# For a scheduled run, prefer a time when no GPU jobs run on the node.
set -u
BUILD_DIR=${1:?usage: run_all_tests.sh BUILD_DIR PREFIX [GPU_INDEX]}
PREFIX=${2:?usage: run_all_tests.sh BUILD_DIR PREFIX [GPU_INDEX]}
GPU=${3:-0}
PYTHON=${PYTHON:-python}
INTERAX_REPO=${INTERAX_REPO:-}
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
    [ -n "${SWIG_PYTHONPATH:-}" ] && export PYTHONPATH=$SWIG_PYTHONPATH
    cd "$BUILD_DIR" && cmake . && make -j16 install && rm -rf python/build && make PythonInstall
) > "$LOG_DIR/build.log" 2>&1
rc=$?; summary "build + install" $rc "$LOG_DIR/build.log"
[ $status -ne 0 ] && exit 1

# C++ tests
(cd "$BUILD_DIR" && LD_PRELOAD=${NVRTC_PRELOAD:-} CUDA_VISIBLE_DEVICES=$GPU ctest -j6 --timeout 1500) > "$LOG_DIR/ctest.log" 2>&1
rc=$?; summary "C++ (ctest): $(grep -o '[0-9]*% tests passed.*' "$LOG_DIR/ctest.log")" $rc "$LOG_DIR/ctest.log"

# Plugin Python tests
(cd "$SOURCE_DIR/tests/python" && INTERAX_PLUGIN_PREFIX=$PREFIX INTERAX_REPO=$INTERAX_REPO LD_PRELOAD=${NVRTC_PRELOAD:-} \
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m pytest -q -p no:cacheprovider -o addopts="" .) > "$LOG_DIR/pytest_plugin.log" 2>&1
rc=$?; summary "plugin Python: $(tail -1 "$LOG_DIR/pytest_plugin.log")" $rc "$LOG_DIR/pytest_plugin.log"

# InteraxFF tests against this build
if [ -n "$INTERAX_REPO" ]; then
    (cd "$INTERAX_REPO" && INTERAX_PLUGIN_PREFIX=$PREFIX LD_PRELOAD=${NVRTC_PRELOAD:-} CUDA_VISIBLE_DEVICES=$GPU \
        "$PYTHON" -m pytest interax/tests -q -p no:cacheprovider -o addopts="") > "$LOG_DIR/pytest_interax.log" 2>&1
    rc=$?; summary "InteraxFF: $(tail -1 "$LOG_DIR/pytest_interax.log")" $rc "$LOG_DIR/pytest_interax.log"
else
    echo "SKIP  InteraxFF (no INTERAX_REPO)"
fi

exit $status
