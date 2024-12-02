# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.13)

###################################################
# Project SIGMA includes the following components: #
#   1. SIGMA C++ library                           #
#   2. SIGMA C export library                      #
#   3. SIGMA C++ examples                          #
#   4. SIGMA C++ tests                             #
###################################################

# [option] CMAKE_BUILD_TYPE (default: "Release")
# Build in one of the following modes: Release, Debug, MiniSizeRel, or RelWithDebInfo.
# Most generators recognize these and can set the compiler flags accordingly. We set
# the build type here before creating the project to prevent the CMake generator from
# overriding our default of "Release".


find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")

project(SIGMA VERSION 4.1.1 LANGUAGES CXX C CUDA)

########################
# Global configuration #
########################

# CMake modules
include(CMakeDependentOption)
include(CMakePushCheckState)
include(CheckIncludeFiles)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CheckTypeSize)
include(CheckSymbolExists)

# Extra modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(SIGMAMacros)

# Always build position-independent-code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Make the install target depend on the all target (required by vcpkg)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY OFF)

# In Debug mode, define SIGMA_DEBUG
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SIGMA_DEBUG ON)
    # In Debug mode, enable extra compiler flags.
    include(EnableCXXCompilerFlags)
else()
    set(SIGMA_DEBUG OFF)
endif()
message(STATUS "Microsoft SIGMA debug mode: ${SIGMA_DEBUG}")

# [option] SIGMA_USE_CXX17 (default: ON)
# Use C++17, use C++14 otherwise.
set(SIGMA_USE_CXX17_OPTION_STR "Use C++17")
option(SIGMA_USE_CXX17 ${SIGMA_USE_CXX17_OPTION_STR} ON)
message(STATUS "SIGMA_USE_CXX17: ${SIGMA_USE_CXX17}")

# Enable features from C++17 if available, disable features if set to OFF.
include(EnableCXX17)

# Enable security-related compile options (MSVC only)
set(SIGMA_SECURE_COMPILE_OPTIONS_OPTIONS_STR "Enable Control Flow Guard and Spectre mitigations (MSVC only)")
option(SIGMA_SECURE_COMPILE_OPTIONS ${SIGMA_SECURE_COMPILE_OPTIONS_OPTIONS_STR} OFF)
mark_as_advanced(SIGMA_SECURE_COMPILE_OPTIONS)

# Path for output
if(ANDROID_ABI)
    # Android compiles several targets at the same time. Need to specify
    # separate directories for separate ABIs.
    set(OUTLIB_PATH "lib/${ANDROID_ABI}")
else()
    set(OUTLIB_PATH "lib")
endif()

# Required files and directories
include(GNUInstallDirs)

# Runtime path setup
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Source Tree
set(SIGMA_INCLUDES_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(SIGMA_CONFIG_IN_FILENAME ${CMAKE_CURRENT_LIST_DIR}/cmake/SIGMAConfig.cmake.in)
set(SIGMA_CONFIG_H_IN_FILENAME ${SIGMA_INCLUDES_DIR}/util/config.h.in)

# Build tree
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${OUTLIB_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${OUTLIB_PATH})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SIGMA_TARGETS_FILENAME ${CMAKE_CURRENT_BINARY_DIR}/cmake/SIGMATargets.cmake)
set(SIGMA_CONFIG_FILENAME ${CMAKE_CURRENT_BINARY_DIR}/cmake/SIGMAConfig.cmake)
set(SIGMA_CONFIG_VERSION_FILENAME ${CMAKE_CURRENT_BINARY_DIR}/cmake/SIGMAConfigVersion.cmake)
set(SIGMA_CONFIG_H_FILENAME ${CMAKE_CURRENT_BINARY_DIR}/src/util/config.h)
set(SIGMA_THIRDPARTY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)

# Install
set(SIGMA_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/SIGMA-${SIGMA_VERSION_MAJOR}.${SIGMA_VERSION_MINOR})
set(SIGMA_INCLUDES_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/SIGMA-${SIGMA_VERSION_MAJOR}.${SIGMA_VERSION_MINOR})

# pkg-config
set(SIGMA_PKGCONFIG_IN_DIR ${CMAKE_CURRENT_LIST_DIR}/pkgconfig)
set(SIGMA_PKGCONFIG_IN_FILENAME ${SIGMA_PKGCONFIG_IN_DIR}/sigma.pc.in)
set(SIGMA_PKGCONFIG_SHARED_IN_FILENAME ${SIGMA_PKGCONFIG_IN_DIR}/sigma_shared.pc.in)
set(SIGMA_PKGCONFIG_MSGSL_IN_FILENAME ${SIGMA_PKGCONFIG_IN_DIR}/sigma_msgsl.pc.in)
set(SIGMA_PKGCONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig)
set(SIGMA_PKGCONFIG_FILENAME ${SIGMA_PKGCONFIG_DIR}/sigma.pc)
set(SIGMA_PKGCONFIG_SHARED_FILENAME ${SIGMA_PKGCONFIG_DIR}/sigma_shared.pc)
set(SIGMA_PKGCONFIG_MSGSL_FILENAME ${SIGMA_PKGCONFIG_DIR}/sigma_msgsl.pc)
set(SIGMA_PKGCONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Clean up artifacts from previous versions
include(CleanArtifacts)

#########################
# External dependencies #
#########################

# [option] SIGMA_BUILD_DEPS (default: ON)
# Download and build missing dependencies, throw error if disabled.
set(SIGMA_BUILD_DEPS_OPTION_STR "Automatically download and build unmet dependencies")
option(SIGMA_BUILD_DEPS ${SIGMA_BUILD_DEPS_OPTION_STR} ON)
message(STATUS "SIGMA_BUILD_DEPS: ${SIGMA_BUILD_DEPS}")

if(SIGMA_BUILD_DEPS)
    include(FetchContent)
    mark_as_advanced(FETCHCONTENT_BASE_DIR)
    mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_QUIET)
endif()

# [option] SIGMA_USE_MSGSL (default: ON)
set(SIGMA_USE_MSGSL_OPTION_STR "Use Microsoft GSL")
option(SIGMA_USE_MSGSL ${SIGMA_USE_MSGSL_OPTION_STR} ON)
set(SIGMA_USE_MSGSL OFF)
message(STATUS "SIGMA_USE_MSGSL: ${SIGMA_USE_MSGSL}")

if(SIGMA_USE_MSGSL)
    if(SIGMA_BUILD_DEPS)
        message(STATUS "Microsoft GSL: download ...")
        sigma_fetch_thirdparty_content(ExternalMSGSL)
    else()
        find_package(Microsoft.GSL 4 CONFIG)
        if(NOT Microsoft.GSL_FOUND)
            message(FATAL_ERROR "Microsoft GSL: not found")
        else()
            message(STATUS "Microsoft GSL: found")
        endif()
    endif()
endif()

# [option] SIGMA_USE_ZLIB (default: ON)
set(SIGMA_USE_ZLIB_OPTION_STR "Use ZLIB for compressed serialization")
option(SIGMA_USE_ZLIB ${SIGMA_USE_ZLIB_OPTION_STR} ON)
set(SIGMA_USE_ZLIB OFF)
message(STATUS "SIGMA_USE_ZLIB: ${SIGMA_USE_ZLIB}")

if(SIGMA_USE_ZLIB)
    if(SIGMA_BUILD_DEPS)
        message(STATUS "ZLIB: download ...")
        sigma_fetch_thirdparty_content(ExternalZLIB)
        set(zlib "zlibstatic")
    else()
        find_package(ZLIB 1)
        if(NOT ZLIB_FOUND)
            message(FATAL_ERROR "ZLIB: not found")
        else()
            message(STATUS "ZLIB: found")
        endif()
        set(zlib "ZLIB::ZLIB")
    endif()
endif()

# [option] SIGMA_USE_ZSTD (default: ON)
set(SIGMA_USE_ZSTD_OPTION_STR "Use Zstandard for compressed serialization")
option(SIGMA_USE_ZSTD ${SIGMA_USE_ZSTD_OPTION_STR} ON)
set(SIGMA_USE_ZSTD OFF)
message(STATUS "SIGMA_USE_ZSTD: ${SIGMA_USE_ZSTD}")

if(SIGMA_USE_ZSTD)
    if(SIGMA_BUILD_DEPS)
        message(STATUS "Zstandard: download ...")
        sigma_fetch_thirdparty_content(ExternalZSTD)
        set(zstd_static "libzstd_static")
    else()
        find_package(zstd CONFIG)
        if(NOT zstd_FOUND)
            message(FATAL_ERROR "Zstandard: not found")
        else()
            if(TARGET zstd::libzstd_static)
                set(zstd_static "zstd::libzstd_static")
            elseif(TARGET libzstd)
                get_target_property(libzstd_type libzstd TYPE)
                if(libzstd_type STREQUAL "STATIC_LIBRARY")
                    set(zstd_static "libzstd")
                    message(STATUS "Zstandard: found")
                else()
                    message(FATAL_ERROR "Zstandard: must be static")
                endif()
            elseif(TARGET zstd::libzstd_shared)
                message(FATAL_ERROR "Zstandard: must be static")
            else()
                message(FATAL_ERROR "Zstandard: not found")
            endif()
        endif()
    endif()
endif()

# [option] SIGMA_USE_INTEL_HEXL (default: OFF)
set(SIGMA_USE_INTEL_HEXL_OPTION_STR "Use Intel HEXL library")
option(SIGMA_USE_INTEL_HEXL ${SIGMA_USE_INTEL_HEXL_OPTION_STR} OFF)
message(STATUS "SIGMA_USE_INTEL_HEXL: ${SIGMA_USE_INTEL_HEXL}")

if(SIGMA_USE_INTEL_HEXL)
    if(SIGMA_BUILD_DEPS)
        message(STATUS "Intel HEXL: download ...")
        sigma_fetch_thirdparty_content(ExternalIntelHEXL)
    else()
        find_package(HEXL 1.2.4)
        if (NOT TARGET HEXL::hexl)
            message(FATAL_ERROR "Intel HEXL: not found")
        endif()
    endif()
endif()

####################
# SIGMA C++ library #
####################

# [option] BUILD_SHARED_LIBS (default: OFF)
# Build a shared library if set to ON. Build a static library regardlessly.
set(BUILD_SHARED_LIBS_STR "Build shared library")
option(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_STR} OFF)
message(STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
if(WIN32 AND BUILD_SHARED_LIBS)
    message(FATAL_ERROR "On Windows only static build is supported; set `BUILD_SHARED_LIBS=OFF`")
endif()

# [option] SIGMA_THROW_ON_TRANSPARENT_CIPHERTEXT (default: ON)
set(SIGMA_THROW_ON_TRANSPARENT_CIPHERTEXT_STR "Throw an exception when Evaluator outputs a transparent ciphertext")
option(SIGMA_THROW_ON_TRANSPARENT_CIPHERTEXT ${SIGMA_THROW_ON_TRANSPARENT_CIPHERTEXT_STR} ON)
message(STATUS "SIGMA_THROW_ON_TRANSPARENT_CIPHERTEXT: ${SIGMA_THROW_ON_TRANSPARENT_CIPHERTEXT}")
mark_as_advanced(FORCE SIGMA_THROW_ON_TRANSPARENT_CIPHERTEXT)

# [option] SIGMA_USE_GAUSSIAN_NOISE (default: OFF)
# Use Gaussian distribution for noise sampling if set to ON, use centered binomial otherwise.
set(SIGMA_USE_GAUSSIAN_NOISE_STR "Use a rounded Gaussian distribution for noise sampling instead of a Centered Binomial Distribution")
option(SIGMA_USE_GAUSSIAN_NOISE ${SIGMA_USE_GAUSSIAN_NOISE_STR} OFF)
message(STATUS "SIGMA_USE_GAUSSIAN_NOISE: ${SIGMA_USE_GAUSSIAN_NOISE}")
mark_as_advanced(FORCE SIGMA_USE_GAUSSIAN_NOISE)

# [option] SIGMA_DEFAULT_PRNG (default: Blake2xb)
# Choose either Blake2xb or Shake256 to be the default PRNG.
set(SIGMA_DEFAULT_PRNG_STR "Choose the default PRNG")
set(SIGMA_DEFAULT_PRNG "Blake2xb" CACHE STRING ${SIGMA_DEFAULT_PRNG_STR} FORCE)
message(STATUS "SIGMA_DEFAULT_PRNG: ${SIGMA_DEFAULT_PRNG}")
set_property(CACHE SIGMA_DEFAULT_PRNG PROPERTY
    STRINGS "Blake2xb" "Shake256")
mark_as_advanced(FORCE SIGMA_DEFAULT_PRNG)

# [option] SIGMA_AVOID_BRANCHING (default: OFF)
# Avoid branching in critical conditional move operations if set to ON, use faster method if set to OFF.
set(SIGMA_AVOID_BRANCHING_STR "Use an always branching-free conditional move")
option(SIGMA_AVOID_BRANCHING ${SIGMA_AVOID_BRANCHING_STR} OFF)
message(STATUS "SIGMA_AVOID_BRANCHING: ${SIGMA_AVOID_BRANCHING}")
mark_as_advanced(FORCE SIGMA_AVOID_BRANCHING)

# [option] SIGMA_USE_INTRIN (default: ON)
set(SIGMA_USE_INTRIN_OPTION_STR "Use intrinsics")
option(SIGMA_USE_INTRIN ${SIGMA_USE_INTRIN_OPTION_STR} ON)
# Look for header files, use intrinsics if available, set to OFF otherwise.
include(CheckCXXIntrinsicsHeader)
if(NOT SIGMA_INTRIN_HEADER_FOUND)
    set(SIGMA_USE_INTRIN OFF CACHE BOOL ${SIGMA_USE_INTRIN_OPTION_STR} FORCE)
endif()
message(STATUS "SIGMA_USE_INTRIN: ${SIGMA_USE_INTRIN}")

# [option] SIGMA_USE_${A_SPECIFIC_INTRIN} (default: ON, advanced)
# Not available if SIGMA_USE_INTRIN is OFF.
# Use a specific intrinsic if available, set to OFF otherwise.
include(CheckCXXIntrinsicsSpecific)

if(MSVC)
    set(SIGMA_USE__UMUL128_OPTION_STR "Use _umul128")
    cmake_dependent_option(SIGMA_USE__UMUL128 ${SIGMA_USE__UMUL128_OPTION_STR} ON "SIGMA_USE_INTRIN" OFF)
    mark_as_advanced(FORCE SIGMA_USE__UMUL128)
    if(NOT SIGMA__UMUL128_FOUND)
        set(SIGMA_USE__UMUL128 OFF CACHE BOOL ${SIGMA_USE__UMUL128_OPTION_STR} FORCE)
    endif()

    set(SIGMA_USE__BITSCANREVERSE64_OPTION_STR "Use _BitScanReverse64")
    cmake_dependent_option(SIGMA_USE__BITSCANREVERSE64 ${SIGMA_USE__BITSCANREVERSE64_OPTION_STR} ON "SIGMA_USE_INTRIN" OFF)
    mark_as_advanced(FORCE SIGMA_USE__BITSCANREVERSE64)
    if(NOT SIGMA__BITSCANREVERSE64_FOUND)
        set(SIGMA_USE__BITSCANREVERSE64 OFF CACHE BOOL ${SIGMA_USE__BITSCANREVERSE64_OPTION_STR} FORCE)
    endif()
else()
    set(SIGMA_USE___INT128_OPTION_STR "Use __int128")
    cmake_dependent_option(SIGMA_USE___INT128 ${SIGMA_USE___INT128_OPTION_STR} ON "SIGMA_USE_INTRIN" OFF)
    mark_as_advanced(FORCE SIGMA_USE___INT128)
    if(NOT SIGMA___INT128_FOUND)
        set(SIGMA_USE___INT128 OFF CACHE BOOL ${SIGMA_USE___INT128_OPTION_STR} FORCE)
    endif()

    set(SIGMA_USE___BUILTIN_CLZLL_OPTION_STR "Use __builtin_clzll")
    cmake_dependent_option(SIGMA_USE___BUILTIN_CLZLL ${SIGMA_USE___BUILTIN_CLZLL_OPTION_STR} ON "SIGMA_USE_INTRIN" OFF)
    mark_as_advanced(FORCE SIGMA_USE___BUILTIN_CLZLL)
    if(NOT SIGMA___BUILTIN_CLZLL_FOUND)
        set(SIGMA_USE___BUILTIN_CLZLL OFF CACHE BOOL ${SIGMA_USE___BUILTIN_CLZLL_OPTION_STR} FORCE)
    endif()
endif()

set(SIGMA_USE__ADDCARRY_U64_OPTION_STR "Use _addcarry_u64")
cmake_dependent_option(SIGMA_USE__ADDCARRY_U64 ${SIGMA_USE__ADDCARRY_U64_OPTION_STR} ON "SIGMA_USE_INTRIN" OFF)
mark_as_advanced(FORCE SIGMA_USE__ADDCARRY_U64)
if(NOT SIGMA__ADDCARRY_U64_FOUND)
    set(SIGMA_USE__ADDCARRY_U64 OFF CACHE BOOL ${SIGMA_USE__ADDCARRY_U64_OPTION_STR} FORCE)
endif()

set(SIGMA_USE__SUBBORROW_U64_OPTION_STR "Use _subborrow_u64")
cmake_dependent_option(SIGMA_USE__SUBBORROW_U64 ${SIGMA_USE__SUBBORROW_U64_OPTION_STR} ON "SIGMA_USE_INTRIN" OFF)
mark_as_advanced(FORCE SIGMA_USE__SUBBORROW_U64)
if(NOT SIGMA__SUBBORROW_U64_FOUND)
    set(SIGMA_USE__SUBBORROW_U64 OFF CACHE BOOL ${SIGMA_USE__SUBBORROW_U64_OPTION_STR} FORCE)
endif()

# [option] SIGMA_USE_${A_SPECIFIC_MEMSET_METHOD} (default: ON, advanced)
# Use a specific memset method if available, set to OFF otherwise.
include(CheckMemset)

set(SIGMA_USE_MEMSET_S_OPTION_STR "Use memset_s")
option(SIGMA_USE_MEMSET_S ${SIGMA_USE_MEMSET_S_OPTION_STR} ON)
mark_as_advanced(FORCE SIGMA_USE_MEMSET_S)
if(NOT SIGMA_MEMSET_S_FOUND)
    set(SIGMA_USE_MEMSET_S OFF CACHE BOOL ${SIGMA_USE_MEMSET_S_OPTION_STR} FORCE)
endif()
message(STATUS "SIGMA_USE_MEMSET_S: ${SIGMA_USE_MEMSET_S}")

set(SIGMA_USE_EXPLICIT_BZERO_OPTION_STR "Use explicit_bzero")
option(SIGMA_USE_EXPLICIT_BZERO ${SIGMA_USE_EXPLICIT_BZERO_OPTION_STR} ON)
mark_as_advanced(FORCE SIGMA_USE_EXPLICIT_BZERO)
if(NOT SIGMA_EXPLICIT_BZERO_FOUND)
    set(SIGMA_USE_EXPLICIT_BZERO OFF CACHE BOOL ${SIGMA_USE_EXPLICIT_BZERO_OPTION_STR} FORCE)
endif()
message(STATUS "SIGMA_USE_EXPLICIT_BZERO: ${SIGMA_USE_EXPLICIT_BZERO}")

set(SIGMA_USE_EXPLICIT_MEMSET_OPTION_STR "Use explicit_memset")
option(SIGMA_USE_EXPLICIT_MEMSET ${SIGMA_USE_EXPLICIT_MEMSET_OPTION_STR} ON)
mark_as_advanced(FORCE SIGMA_USE_EXPLICIT_MEMSET)
if(NOT SIGMA_EXPLICIT_MEMSET_FOUND)
    set(SIGMA_USE_EXPLICIT_MEMSET OFF CACHE BOOL ${SIGMA_USE_EXPLICIT_MEMSET_OPTION_STR} FORCE)
endif()
message(STATUS "SIGMA_USE_EXPLICIT_MEMSET: ${SIGMA_USE_EXPLICIT_MEMSET}")

# [option] SIGMA_USE_ALIGNED_ALLOC (default: ON, advanced)
# Not available if SIGMA_USE_CXX17 is OFF or building for Android.
# Use 64-byte aligned malloc if available, set of OFF otherwise
if(MSVC)
    set(SIGMA_USE_ALIGNED_ALLOC_OPTION_STR "Use _aligned_malloc")
else()
    set(SIGMA_USE_ALIGNED_ALLOC_OPTION_STR "Use ::aligned_alloc")
endif()
cmake_dependent_option(SIGMA_USE_ALIGNED_ALLOC ${SIGMA_USE_ALIGNED_ALLOC_OPTION_STR} ON "SIGMA_USE_CXX17;NOT ANDROID_ABI" OFF)
mark_as_advanced(FORCE SIGMA_USE_ALIGNED_ALLOC)

# Add source files to library and header files to install
set(SIGMA_SOURCE_FILES "")
add_subdirectory(src)

# Create the config file
configure_file(${SIGMA_CONFIG_H_IN_FILENAME} ${SIGMA_CONFIG_H_FILENAME})
install(
    FILES ${SIGMA_CONFIG_H_FILENAME}
    DESTINATION ${SIGMA_INCLUDES_INSTALL_DIR}/sigma/util)

# Build only a static library
if(NOT BUILD_SHARED_LIBS)
    add_library(sigma STATIC ${SIGMA_SOURCE_FILES})
    sigma_set_version_filename(sigma)
    sigma_set_language(sigma)
    sigma_set_include_directories(sigma)
    sigma_set_version(sigma)
    sigma_link_threads(sigma)
    sigma_install_target(sigma SIGMATargets)

    if(SIGMA_USE_MSGSL)
        if(SIGMA_BUILD_DEPS)
            target_include_directories(sigma PUBLIC $<BUILD_INTERFACE:${msgsl_SOURCE_DIR}/include>)
        else()
            target_link_libraries(sigma PUBLIC Microsoft.GSL::GSL)
        endif()
    endif()

    if(SIGMA_USE_ZLIB)
        if(SIGMA_BUILD_DEPS)
            add_dependencies(sigma ${zlib})
            target_include_directories(sigma PRIVATE $<BUILD_INTERFACE:${zlib_SOURCE_DIR}>)
            target_include_directories(sigma PRIVATE $<BUILD_INTERFACE:$<TARGET_PROPERTY:${zlib},BINARY_DIR>>)
            sigma_combine_archives(sigma ${zlib})
        else()
            target_link_libraries(sigma PRIVATE ${zlib})
        endif()
    endif()

    if(SIGMA_USE_ZSTD)
        if(SIGMA_BUILD_DEPS)
            add_dependencies(sigma ${zstd_static})
            target_include_directories(sigma PRIVATE $<BUILD_INTERFACE:${zstd_SOURCE_DIR}/lib>)
            target_include_directories(sigma PRIVATE $<BUILD_INTERFACE:${zstd_SOURCE_DIR}/lib/common>)
            sigma_combine_archives(sigma ${zstd_static})
        else()
            target_link_libraries(sigma PRIVATE ${zstd_static})
        endif()
    endif()

    if(SIGMA_USE_INTEL_HEXL)
        if(SIGMA_BUILD_DEPS)
            add_dependencies(sigma HEXL::hexl)
            target_include_directories(sigma PRIVATE
                $<BUILD_INTERFACE:$<TARGET_PROPERTY:HEXL::hexl,INTERFACE_INCLUDE_DIRECTORIES>>)
            target_compile_options(sigma PRIVATE $<TARGET_PROPERTY:HEXL::hexl,INTERFACE_COMPILE_OPTIONS>)
            get_target_property(HEXL_INTERFACE_LINK_OPTIONS HEXL::hexl INTERFACE_LINK_OPTIONS)
            if (NOT "${HEXL_INTERFACE_LINK_OPTIONS}" STREQUAL "HEXL_INTERFACE_LINK_OPTIONS-NOTFOUND")
                target_link_libraries(sigma INTERFACE ${HEXL_INTERFACE_LINK_OPTIONS})
            endif()
            sigma_combine_archives(sigma HEXL::hexl)
        else()
            target_link_libraries(sigma PUBLIC HEXL::hexl)
        endif()
    endif()

    # Set secure compile options if SIGMA_SECURE_COMPILE_OPTIONS is ON; only supported on MSVC
    if(SIGMA_SECURE_COMPILE_OPTIONS)
        sigma_set_secure_compile_options(sigma PUBLIC)
    endif()

# Build only a shared library
else()
    add_library(sigma_shared SHARED ${SIGMA_SOURCE_FILES})
    sigma_set_soversion(sigma_shared)
    set_target_properties(sigma_shared PROPERTIES OUTPUT_NAME sigma)
    sigma_set_language(sigma_shared)
    sigma_set_include_directories(sigma_shared)
    sigma_set_version(sigma_shared)
    sigma_link_threads(sigma_shared)
    sigma_install_target(sigma_shared SIGMATargets)

    if(SIGMA_USE_MSGSL)
        if(SIGMA_BUILD_DEPS)
            target_include_directories(sigma_shared PUBLIC $<BUILD_INTERFACE:${msgsl_SOURCE_DIR}/include>)
        else()
            target_link_libraries(sigma_shared PUBLIC Microsoft.GSL::GSL)
        endif()
    endif()

    if(SIGMA_USE_ZLIB)
        target_link_libraries(sigma_shared PRIVATE ${zlib})
        target_include_directories(sigma_shared PRIVATE $<BUILD_INTERFACE:${zlib_SOURCE_DIR}>)
        target_include_directories(sigma_shared PRIVATE $<BUILD_INTERFACE:$<TARGET_PROPERTY:${zlib},BINARY_DIR>>)
    endif()

    if(SIGMA_USE_ZSTD)
        target_include_directories(sigma_shared PRIVATE $<BUILD_INTERFACE:${zstd_SOURCE_DIR}/lib>)
        target_include_directories(sigma_shared PRIVATE $<BUILD_INTERFACE:${zstd_SOURCE_DIR}/lib/common>)
        target_link_libraries(sigma_shared PRIVATE ${zstd_static})
    endif()

    if(SIGMA_USE_INTEL_HEXL)
        target_link_libraries(sigma_shared PRIVATE HEXL::hexl)
        target_compile_options(sigma_shared PRIVATE $<TARGET_PROPERTY:HEXL::hexl,INTERFACE_COMPILE_OPTIONS>)
        get_target_property(HEXL_INTERFACE_LINK_OPTIONS HEXL::hexl INTERFACE_LINK_OPTIONS)
        if (NOT "${HEXL_INTERFACE_LINK_OPTIONS}" STREQUAL "HEXL_INTERFACE_LINK_OPTIONS-NOTFOUND")
            target_link_libraries(sigma_shared INTERFACE ${HEXL_INTERFACE_LINK_OPTIONS})
        endif()

        # Install shared HEXL library to installation directory
        install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_LIBRARY_PREFIX}
            FILES_MATCHING PATTERN "*hexl*"
            )
    endif()
endif()

# Add standard alias targets for SIGMA::sigma and SIGMA::sigma_shared
if(TARGET sigma)
    add_library(SIGMA::sigma ALIAS sigma)
endif()
if(TARGET sigma_shared)
    add_library(SIGMA::sigma_shared ALIAS sigma_shared)
endif()

#########################
# SIGMA C export library #
#########################

# [option] SIGMA_BUILD_SIGMA_C (default: OFF)
set(SIGMA_BUILD_SIGMA_C_OPTION_STR "Build C export library for Microsoft SIGMA")
option(SIGMA_BUILD_SIGMA_C ${SIGMA_BUILD_SIGMA_C_OPTION_STR} OFF)
message(STATUS "SIGMA_BUILD_SIGMA_C: ${SIGMA_BUILD_SIGMA_C}")

set(SIGMA_BUILD_STATIC_SIGMA_C_OPTION_STR "Build static C library for Microsoft SIGMA")
cmake_dependent_option(SIGMA_BUILD_STATIC_SIGMA_C ${SIGMA_BUILD_STATIC_SIGMA_C_OPTION_STR} OFF "SIGMA_BUILD_SIGMA_C" OFF)
mark_as_advanced(SIGMA_BUILD_STATIC_SIGMA_C)

# Create SIGMA_C library but add no source files yet
if(SIGMA_BUILD_SIGMA_C)
    # Check that size_t is 8 bytes
    include(CheckTypeSize)
    check_type_size("size_t" SIZET LANGUAGE C)
    if(NOT ${SIZET} EQUAL 8)
        unset(SIZET CACHE)
        unset(HAVE_SIZET CACHE)
        message(FATAL_ERROR "SIGMA_C requires 64-bit platform")
    endif()
    unset(SIZET CACHE)
    unset(HAVE_SIZET CACHE)

    if(BUILD_SHARED_LIBS)
        message(FATAL_ERROR "SIGMA_C (shared or static) requires a static SIGMA; set `BUILD_SHARED_LIBS=OFF` and control "
                            "SIGMA_C by setting `SIGMA_BUILD_STATIC_SIGMA_C` to `OFF` (default) or `ON`")
    endif()

    if(NOT SIGMA_BUILD_DEPS)
        message(FATAL_ERROR "SIGMA_C requires `SIGMA_BUILD_DEPS=ON`")
    endif()

    if(NOT SIGMA_BUILD_STATIC_SIGMA_C AND SIGMA_USE_INTEL_HEXL AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND UNIX)
        message(WARNING "SIGMA_C (shared) cannot find Address Sanitizer (libasan) enabled by Intel HEXL in Debug mode "
                        "on Unix-like systems. SIGMANet does not work. Preload libasan for downstream applications.")
    endif()

    if(SIGMA_BUILD_STATIC_SIGMA_C)
        add_library(sigmac STATIC)
    else()
        add_library(sigmac SHARED)
    endif()

    # Add source files to library and header files to install
    add_subdirectory(${SIGMA_INCLUDES_DIR}/sigma/c)
    sigma_set_version(sigmac)
    if(SIGMA_BUILD_STATIC_SIGMA_C)
        sigma_set_version_filename(sigmac)
    else()
        sigma_set_soversion(sigmac)
    endif()
    sigma_set_language(sigmac)
    sigma_set_include_directories(sigmac)
    if (SIGMA_USE_INTEL_HEXL)
        target_include_directories(sigmac PRIVATE $<BUILD_INTERFACE:${hexl_SOURCE_DIR}/hexl/include>)
    endif()

    target_link_libraries(sigmac PUBLIC sigma)

    sigma_install_target(sigmac SIGMATargets)
endif()

# Add standard alias target for SIGMA::sigmac
if(TARGET sigmac)
    add_library(SIGMA::sigmac ALIAS sigmac)
endif()

#################################
# Installation and CMake config #
#################################

# Create the CMake config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${SIGMA_CONFIG_IN_FILENAME} ${SIGMA_CONFIG_FILENAME}
    INSTALL_DESTINATION ${SIGMA_CONFIG_INSTALL_DIR})

# Install the export
install(
    EXPORT SIGMATargets
    NAMESPACE SIGMA::
    DESTINATION ${SIGMA_CONFIG_INSTALL_DIR})

# Version file; we require exact version match for downstream
write_basic_package_version_file(
    ${SIGMA_CONFIG_VERSION_FILENAME}
    VERSION ${SIGMA_VERSION}
    COMPATIBILITY SameMinorVersion)

# Install config and module files
install(
    FILES
        ${SIGMA_CONFIG_FILENAME}
        ${SIGMA_CONFIG_VERSION_FILENAME}
    DESTINATION ${SIGMA_CONFIG_INSTALL_DIR})

# We export SIGMATargets from the build tree so it can be used by other projects
# without requiring an install.
export(
    EXPORT SIGMATargets
    NAMESPACE SIGMA::
    FILE ${SIGMA_TARGETS_FILENAME})

# Install header files of dependencies if SIGMA_BUILD_DEPS is ON
if(SIGMA_BUILD_DEPS)
    # MSGSL
    if(SIGMA_USE_MSGSL)
        install(
            DIRECTORY ${msgsl_SOURCE_DIR}/include/gsl
            DESTINATION ${SIGMA_INCLUDES_INSTALL_DIR})
    endif()
endif()

##############
# pkg-config #
##############

if(UNIX OR APPLE)
    # Set the requirements according to the current configuration if SIGMA_BUILD_DEPS=OFF
    if(NOT SIGMA_BUILD_DEPS)
        if(SIGMA_USE_MSGSL)
            # Extract the include directories from Microsoft.GSL::GSL
            get_target_property(
                SIGMA_PKGCONFIG_MSGSL_INCLUDE_DIR
                Microsoft.GSL::GSL
                INTERFACE_INCLUDE_DIRECTORIES)

            # Create the Microsoft GSL pkg-config file
            configure_file(${SIGMA_PKGCONFIG_MSGSL_IN_FILENAME} ${SIGMA_PKGCONFIG_MSGSL_FILENAME} @ONLY)

            set(SIGMA_PKGCONFIG_MSGSL_REQUIRES_STRING "sigma_msgsl >= ${Microsoft.GSL_VERSION_MAJOR}")
        endif()
        if(SIGMA_USE_ZLIB)
            set(SIGMA_PKGCONFIG_ZLIB_REQUIRES_STRING "zlib = ${ZLIB_VERSION_STRING}")
        endif()
        if(SIGMA_USE_ZSTD)
            set(SIGMA_PKGCONFIG_ZSTD_REQUIRES_STRING "libzstd >= ${zstd_VERSION_MAJOR}")
        endif()
        if(SIGMA_USE_INTEL_HEXL)
            set(SIGMA_PKGCONFIG_INTEL_HEXL_REQUIRES_STRING "hexl >= ${HEXL_VERSION_MAJOR}")
            message(STATUS "Intel HEXL does not have a pkg-config file")
        endif()
    endif()

    # Create the pkg-config file
    if(TARGET sigma)
        configure_file(${SIGMA_PKGCONFIG_IN_FILENAME} ${SIGMA_PKGCONFIG_FILENAME} @ONLY)
    endif()
    if(TARGET sigma_shared)
        configure_file(${SIGMA_PKGCONFIG_SHARED_IN_FILENAME} ${SIGMA_PKGCONFIG_SHARED_FILENAME} @ONLY)
    endif()

    # Install the pkg-config files
    if(EXISTS ${SIGMA_PKGCONFIG_FILENAME})
        install(
            FILES ${SIGMA_PKGCONFIG_FILENAME}
            DESTINATION ${SIGMA_PKGCONFIG_INSTALL_DIR})
    endif()
    if(EXISTS ${SIGMA_PKGCONFIG_SHARED_FILENAME})
        install(
            FILES ${SIGMA_PKGCONFIG_SHARED_FILENAME}
            DESTINATION ${SIGMA_PKGCONFIG_INSTALL_DIR})
    endif()
    if(EXISTS ${SIGMA_PKGCONFIG_MSGSL_FILENAME})
        install(
            FILES ${SIGMA_PKGCONFIG_MSGSL_FILENAME}
            DESTINATION ${SIGMA_PKGCONFIG_INSTALL_DIR})
    endif()
endif()

#######################################
# Configure SIGMANet and NuGet package #
#######################################

# In Windows we will set the SIGMA_C library path according to the CMake generator
set(SIGMA_WINDOWS_SIGMA_C_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
get_property(SIGMA_IS_MULTI_CONFIG_GENERATOR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(SIGMA_IS_MULTI_CONFIG_GENERATOR)
    # Is this the Visual Studio generator? If so, the output path will contain the configuration.
    # We shall use CMAKE_BUILD_TYPE here, which by default will be "Release". The user has the
    # option of changing this by explicitly specifying CMAKE_BUILD_TYPE, which multi-config
    # generators otherwise ignore.
    set(SIGMA_WINDOWS_SIGMA_C_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE})
endif()

set(NUGET_WINDOWS_SIGMA_C_PATH "")
set(NUGET_LINUX_SIGMA_C_PATH "")
set(NUGET_MACOS_SIGMA_C_PATH "")

# Supporting local building of NuGet package
if(WIN32)
    set(NUGET_WINDOWS_SIGMA_C_PATH ${SIGMA_WINDOWS_SIGMA_C_DIRECTORY}/sigmac.dll)
elseif(UNIX AND NOT APPLE)
    set(NUGET_LINUX_SIGMA_C_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsigmac.so)
elseif(APPLE)
    set(NUGET_MACOS_SIGMA_C_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsigmac.dylib)
endif()

