cmake_minimum_required(VERSION 3.24)

# try to find the CUDA compiler if not set, and if the default location exists
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# set before project() to avoid CMake warning
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real 100-real 120) # 120: cubin and PTX
endif()

project(gpcmap LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# the stream masks: libsmctrl from the submodule
add_library(smctrl SHARED third_party/libsmctrl/libsmctrl.c)
target_include_directories(smctrl PUBLIC third_party/libsmctrl)
target_link_libraries(smctrl PUBLIC CUDA::cuda_driver)

# the driver's table needs the open-gpu-kernel-modules tree of the installed driver
option(GPCMAP_RM "the rm method, when the driver's kernel-module tree is found" ON)
if(GPCMAP_RM)
    include(cmake/NVOpenGpuKernelModules.cmake)
endif()

add_library(gpcmap SHARED src/gpcmap.cu src/gpcmap_rm.c)
target_include_directories(gpcmap PUBLIC include)
target_link_libraries(gpcmap PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
if(GPCMAP_NV_FOUND)
    target_compile_definitions(gpcmap PRIVATE GPCMAP_RM)
    target_include_directories(gpcmap PRIVATE ${GPCMAP_NV_INCLUDE_DIRS})
endif()

# link the CLI tool to libgpcmap and libsmctrl
add_executable(gpcmap_cli src/gpcmap_cli.c)
set_target_properties(gpcmap_cli PROPERTIES OUTPUT_NAME gpcmap BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
target_link_options(gpcmap_cli PRIVATE LINKER:--no-as-needed)
target_link_libraries(gpcmap_cli PRIVATE gpcmap smctrl)
