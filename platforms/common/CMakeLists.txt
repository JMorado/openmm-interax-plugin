# Part of the OpenMM InteraxFF plugin. Copyright (c) 2026 the Authors. MIT license, see LICENSE.

#---------------------------------------------------
# OpenMM InteraxFF Plugin Common Compute Kernels
#
# The kernels in src/kernels are written for OpenMM's common compute framework and built for the CUDA platform. They
# are encoded into the C++ class CommonInteraxKernelSources, which the platform library compiles together with
# src/CommonInteraxKernels.cpp.
#----------------------------------------------------

SET(KERNEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(KERNEL_SOURCE_CLASS CommonInteraxKernelSources)
SET(KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.cpp)
SET(KERNELS_H ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.h)
FILE(GLOB COMMON_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cc)
ADD_CUSTOM_COMMAND(OUTPUT ${KERNELS_CPP} ${KERNELS_H}
    COMMAND ${CMAKE_COMMAND}
    ARGS -D KERNEL_SOURCE_DIR=${KERNEL_SOURCE_DIR} -D KERNELS_CPP=${KERNELS_CPP} -D KERNELS_H=${KERNELS_H} -D KERNEL_SOURCE_CLASS=${KERNEL_SOURCE_CLASS} -P ${CMAKE_CURRENT_SOURCE_DIR}/EncodeKernelFiles.cmake
    DEPENDS ${COMMON_KERNELS} ${KERNEL_SOURCE_DIR}/${KERNEL_SOURCE_CLASS}.cpp.in ${KERNEL_SOURCE_DIR}/${KERNEL_SOURCE_CLASS}.h.in
)
SET_SOURCE_FILES_PROPERTIES(${KERNELS_CPP} ${KERNELS_H} PROPERTIES GENERATED TRUE)
ADD_CUSTOM_TARGET(InteraxCommonKernels DEPENDS ${KERNELS_CPP} ${KERNELS_H})
