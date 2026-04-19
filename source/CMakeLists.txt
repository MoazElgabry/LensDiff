cmake_minimum_required(VERSION 3.20)

project(LensDiff VERSION 0.2.0 LANGUAGES CXX)

set(PLUGIN_BASENAME "LensDiff")
set(CUBE_DISPLAY_VERSION "0.2.0")
set(PLUGIN_IDENTIFIER "com.moazelgabry.lensdiff")
set(LOCAL_OFX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ofx-sdk")
set(GENERATED_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(COMMON_COLOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common/color")
set(SHARED_ICON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../icons")
set(PLUGIN_ICON_PNG "${SHARED_ICON_DIR}/LensDiff 2 to 1.png")

if(NOT EXISTS "${LOCAL_OFX_SDK_DIR}/include/ofxImageEffect.h")
  message(FATAL_ERROR "Missing local OFX SDK at ${LOCAL_OFX_SDK_DIR}.")
endif()

if(NOT EXISTS "${COMMON_COLOR_DIR}/ColorManagement.cpp" OR
   NOT EXISTS "${COMMON_COLOR_DIR}/ColorManagement.h")
  message(FATAL_ERROR "Missing vendored common color sources at ${COMMON_COLOR_DIR}.")
endif()

if(NOT EXISTS "${PLUGIN_ICON_PNG}")
  message(FATAL_ERROR "Missing LensDiff icon asset at ${PLUGIN_ICON_PNG}.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")
endif()

if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architectures to build")
endif()

include(CheckLanguage)

if(APPLE)
  enable_language(OBJCXX)
endif()

find_package(Threads REQUIRED)

if(WIN32 AND NOT DEFINED CUDAToolkit_ROOT)
  set(_LENSDIFF_WINDOWS_CUDA_CANDIDATES
    "$ENV{CUDA_PATH_V12_9}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
    "$ENV{CUDA_PATH_V12_8}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
    "$ENV{CUDA_PATH_V11_8}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
    "$ENV{CUDA_PATH}"
  )
  foreach(_cuda_candidate IN LISTS _LENSDIFF_WINDOWS_CUDA_CANDIDATES)
    if(_cuda_candidate AND EXISTS "${_cuda_candidate}/bin/nvcc.exe")
      set(CUDAToolkit_ROOT "${_cuda_candidate}" CACHE PATH "Preferred CUDA toolkit root" FORCE)
      break()
    endif()
  endforeach()
endif()

find_package(CUDAToolkit QUIET)

set(LENSDIFF_HAS_CUDA OFF)
if(CUDAToolkit_FOUND AND NOT APPLE)
  if((NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "-NOTFOUND$")
     AND DEFINED CUDAToolkit_NVCC_EXECUTABLE
     AND EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
  check_language(CUDA)
endif()
if(CUDAToolkit_FOUND AND NOT APPLE AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_LESS "13.0")
      set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 89)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
  endif()
  if(WIN32 AND CMAKE_CUDA_FLAGS MATCHES "(^| )--allow-unsupported-compiler( |$)")
  elseif(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
  endif()
  set(LENSDIFF_HAS_CUDA ON)
endif()

set(OFX_SUPPORT_SOURCES
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsCore.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsImageEffect.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsInteract.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsLog.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsMultiThread.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsParams.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsProperty.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsPropertyValidation.cpp
)

add_library(${PLUGIN_BASENAME} SHARED
  src/LensDiff.cpp
  src/core/LensDiffApertureImage.cpp
  src/core/LensDiffCpuReference.cpp
  $<$<NOT:$<PLATFORM_ID:Darwin>>:src/core/LensDiffFileDialog.cpp>
  src/core/LensDiffPhase.cpp
  src/core/LensDiffSpectrum.cpp
  src/core/LensDiffTransfer.cpp
  ${COMMON_COLOR_DIR}/ColorManagement.cpp
  $<$<BOOL:${LENSDIFF_HAS_CUDA}>:src/cuda/LensDiffCuda.cu>
  $<$<PLATFORM_ID:Darwin>:src/core/LensDiffFileDialog.mm>
  $<$<PLATFORM_ID:Darwin>:src/metal/LensDiffMetal.mm>
  ${OFX_SUPPORT_SOURCES}
)

target_compile_features(${PLUGIN_BASENAME} PRIVATE cxx_std_17)
target_include_directories(${PLUGIN_BASENAME} PRIVATE
  ${LOCAL_OFX_SDK_DIR}/include
  ${LOCAL_OFX_SDK_DIR}/Support/include
  ${LOCAL_OFX_SDK_DIR}/Support/Library
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${COMMON_COLOR_DIR}
)
target_link_libraries(${PLUGIN_BASENAME} PRIVATE Threads::Threads)

if(LENSDIFF_HAS_CUDA)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE LENSDIFF_HAS_CUDA)
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart CUDA::cufft)
  set_target_properties(${PLUGIN_BASENAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)
  if(UNIX AND NOT APPLE)
    set_target_properties(${PLUGIN_BASENAME} PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
  endif()
endif()

if(MSVC)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${PLUGIN_BASENAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /bigobj /MP /EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3,-bigobj,-EHsc>)
endif()

if(WIN32)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE windowscodecs)
endif()

if(APPLE)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework AppKit"
    "-framework ApplicationServices"
    "-framework ImageIO")
endif()

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  OUTPUT_NAME "${PLUGIN_BASENAME}"
)

set(OFX_STAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle")
if(WIN32)
  set(OFX_OUTPUT_DIR "${OFX_STAGE_ROOT}/Contents/Win64")
  set(OFX_RESOURCES_DIR "${OFX_STAGE_ROOT}/Contents/Resources")
elseif(APPLE)
  set(OFX_BUNDLE_CONTENTS_DIR "${OFX_STAGE_ROOT}/Contents")
  set(OFX_OUTPUT_DIR "${OFX_BUNDLE_CONTENTS_DIR}/MacOS")
  set(OFX_RESOURCES_DIR "${OFX_BUNDLE_CONTENTS_DIR}/Resources")
else()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LENSDIFF_LINUX_ARCH)
  if(LENSDIFF_LINUX_ARCH STREQUAL "x86_64" OR LENSDIFF_LINUX_ARCH STREQUAL "amd64")
    set(OFX_LINUX_ARCH_DIR "Linux-x86-64")
  elseif(LENSDIFF_LINUX_ARCH STREQUAL "x86" OR LENSDIFF_LINUX_ARCH STREQUAL "i686" OR LENSDIFF_LINUX_ARCH STREQUAL "i386")
    set(OFX_LINUX_ARCH_DIR "Linux-x86")
  else()
    set(OFX_LINUX_ARCH_DIR "Linux-${LENSDIFF_LINUX_ARCH}")
  endif()
  set(OFX_OUTPUT_DIR "${OFX_STAGE_ROOT}/Contents/${OFX_LINUX_ARCH_DIR}")
  set(OFX_RESOURCES_DIR "${OFX_STAGE_ROOT}/Contents/Resources")
endif()

set(BUILD_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
set(BUILD_PLUGIN_DIR "${BUILD_ARTIFACT_DIR}/plugin")

file(MAKE_DIRECTORY "${OFX_OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OFX_RESOURCES_DIR}")
file(MAKE_DIRECTORY "${GENERATED_ASSET_DIR}")
file(MAKE_DIRECTORY "${BUILD_PLUGIN_DIR}")

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
  COMPILE_PDB_NAME "${PLUGIN_BASENAME}"
  COMPILE_PDB_NAME_DEBUG "${PLUGIN_BASENAME}"
  COMPILE_PDB_NAME_RELEASE "${PLUGIN_BASENAME}"
  COMPILE_PDB_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
  PDB_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  PDB_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  PDB_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
)

if(WIN32 AND EXISTS "${PLUGIN_ICON_PNG}")
  set(PLUGIN_ICON_ICO "${GENERATED_ASSET_DIR}/${PLUGIN_BASENAME}.ico")
  set(PLUGIN_ICON_RC "${GENERATED_ASSET_DIR}/${PLUGIN_BASENAME}.rc")

  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/plugin_icon.rc.in" "${PLUGIN_ICON_RC}" @ONLY)

  add_custom_command(
    OUTPUT "${PLUGIN_ICON_ICO}"
    COMMAND powershell -ExecutionPolicy Bypass -File "${CMAKE_CURRENT_SOURCE_DIR}/cmake/png_to_ico.ps1" -InputPng "${PLUGIN_ICON_PNG}" -OutputIco "${PLUGIN_ICON_ICO}"
    DEPENDS "${PLUGIN_ICON_PNG}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/png_to_ico.ps1"
    COMMENT "Generating Windows ICO from ${PLUGIN_ICON_PNG}"
    VERBATIM)

  target_sources(${PLUGIN_BASENAME} PRIVATE "${PLUGIN_ICON_RC}" "${PLUGIN_ICON_ICO}")
endif()

if(APPLE)
  file(MAKE_DIRECTORY "${OFX_BUNDLE_CONTENTS_DIR}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist.in" "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" "${OFX_BUNDLE_CONTENTS_DIR}/Info.plist"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${PLUGIN_BASENAME}>" "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.ofx")
else()
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${PLUGIN_BASENAME}>" "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.ofx")
endif()

if(EXISTS "${PLUGIN_ICON_PNG}")
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_RESOURCES_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PLUGIN_ICON_PNG}" "${OFX_RESOURCES_DIR}/${PLUGIN_IDENTIFIER}.png")
endif()
