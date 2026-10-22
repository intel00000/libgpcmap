# Finds the open-gpu-kernel-modules tree of the installed driver, for the rm
# method: GPCMAP_NV, the NV_OPEN_GPU_KERNEL_MODULES environment, or the
# usual places, taking a tree whose version matches the loaded driver.
# GPCMAP_FETCH_NV fetches the driver's tag from GitHub when none is found.
# Sets GPCMAP_NV_FOUND, GPCMAP_NV_DIR, GPCMAP_NV_VERSION, GPCMAP_NV_INCLUDE_DIRS.

set(GPCMAP_NV "" CACHE PATH "open-gpu-kernel-modules tree of the installed driver")
option(GPCMAP_FETCH_NV "fetch the open-gpu-kernel-modules tag of the installed driver when no tree is found" OFF)

set(GPCMAP_NV_FOUND FALSE)
set(_gpcmap_driver "")
foreach(f /sys/module/nvidia/version /proc/driver/nvidia/version)
    if(EXISTS ${f} AND NOT _gpcmap_driver)
        file(READ ${f} _text)
        string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" _gpcmap_driver "${_text}")
    endif()
endforeach()

function(_gpcmap_nv_version dir out)
    set(v "")
    if(EXISTS ${dir}/version.mk)
        file(STRINGS ${dir}/version.mk line REGEX "^NVIDIA_VERSION *=")
        string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" v "${line}")
    endif()
    set(${out} "${v}" PARENT_SCOPE)
endfunction()

function(_gpcmap_nv_complete dir out)
    set(ok TRUE)
    foreach(h src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h
              src/nvidia/arch/nvalloc/unix/include/nv_escape.h
              kernel-open/common/inc/nv-ioctl.h)
        if(NOT EXISTS ${dir}/${h})
            set(ok FALSE)
        endif()
    endforeach()
    set(${out} ${ok} PARENT_SCOPE)
endfunction()

set(_gpcmap_explicit "")
if(GPCMAP_NV)
    list(APPEND _gpcmap_explicit ${GPCMAP_NV})
endif()
if(DEFINED ENV{NV_OPEN_GPU_KERNEL_MODULES})
    list(APPEND _gpcmap_explicit $ENV{NV_OPEN_GPU_KERNEL_MODULES})
endif()
file(GLOB _gpcmap_found LIST_DIRECTORIES true
     /usr/src/open-gpu-kernel-modules* /opt/open-gpu-kernel-modules*
     $ENV{HOME}/open-gpu-kernel-modules* ${CMAKE_SOURCE_DIR}/../open-gpu-kernel-modules*
     /usr/src/nvidia-*)

foreach(dir ${_gpcmap_explicit} ${_gpcmap_found})
    _gpcmap_nv_complete(${dir} complete)
    if(NOT complete)
        message(STATUS "gpcmap: ${dir} lacks the SDK headers; skipped")
        continue()
    endif()
    _gpcmap_nv_version(${dir} v)
    if(_gpcmap_driver AND v AND NOT v STREQUAL _gpcmap_driver)
        if(dir IN_LIST _gpcmap_explicit)
            message(WARNING "gpcmap: ${dir} is open-gpu-kernel-modules ${v}, the loaded driver is ${_gpcmap_driver}; the RM ABI may differ")
        else()
            message(STATUS "gpcmap: ${dir} is ${v}, the driver is ${_gpcmap_driver}; skipped")
            continue()
        endif()
    endif()
    set(GPCMAP_NV_FOUND TRUE)
    set(GPCMAP_NV_DIR ${dir})
    set(GPCMAP_NV_VERSION "${v}")
    break()
endforeach()

if(NOT GPCMAP_NV_FOUND AND GPCMAP_FETCH_NV AND _gpcmap_driver)
    include(FetchContent)
    FetchContent_Declare(nv_open_gpu_kernel_modules
        GIT_REPOSITORY https://github.com/NVIDIA/open-gpu-kernel-modules.git
        GIT_TAG ${_gpcmap_driver}
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nv_open_gpu_kernel_modules)
    set(GPCMAP_NV_FOUND TRUE)
    set(GPCMAP_NV_DIR ${nv_open_gpu_kernel_modules_SOURCE_DIR})
    set(GPCMAP_NV_VERSION ${_gpcmap_driver})
endif()

if(GPCMAP_NV_FOUND)
    set(GPCMAP_NV_INCLUDE_DIRS
        ${GPCMAP_NV_DIR}/src/common/sdk/nvidia/inc
        ${GPCMAP_NV_DIR}/src/nvidia/arch/nvalloc/unix/include
        ${GPCMAP_NV_DIR}/kernel-open/common/inc)
    message(STATUS "gpcmap: rm method from ${GPCMAP_NV_DIR} (${GPCMAP_NV_VERSION}; loaded driver '${_gpcmap_driver}')")
else()
    message(STATUS "gpcmap: no open-gpu-kernel-modules tree for driver '${_gpcmap_driver}'; the rm method is left out (GPCMAP_NV=<tree>, or GPCMAP_FETCH_NV=ON)")
endif()
