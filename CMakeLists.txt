cmake_minimum_required(VERSION 3.20)
project(vkdnn VERSION 0.1.0 LANGUAGES C CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

find_package(Vulkan REQUIRED)
get_filename_component(VKDNN_VULKAN_BIN "${Vulkan_GLSLC_EXECUTABLE}" DIRECTORY)
find_program(VKDNN_GLSLC NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "${VKDNN_VULKAN_BIN}" REQUIRED)
set(FORWARD_SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/forward_spv.inc")
add_custom_command(OUTPUT "${FORWARD_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${VKDNN_GLSLC}" --target-env=vulkan1.2 -mfmt=c "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/forward.comp" -o "${FORWARD_SPV}"
    DEPENDS src/ops/forward.comp VERBATIM)

set(GEMM_SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/gemm_spv.inc")
set(BACKWARD_SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/backward_spv.inc")
add_custom_command(OUTPUT "${BACKWARD_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${VKDNN_GLSLC}" --target-env=vulkan1.2 -mfmt=c "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/backward.comp" -o "${BACKWARD_SPV}"
    DEPENDS src/ops/backward.comp VERBATIM)
add_custom_command(OUTPUT "${GEMM_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${VKDNN_GLSLC}" --target-env=vulkan1.2 -mfmt=c "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/gemm.comp" -o "${GEMM_SPV}"
    DEPENDS src/ops/gemm.comp VERBATIM)
set(CONVOLUTION_SPV "${CMAKE_CURRENT_BINARY_DIR}/generated/convolution_spv.inc")
add_custom_command(OUTPUT "${CONVOLUTION_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${VKDNN_GLSLC}" --target-env=vulkan1.2 -mfmt=c "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/convolution.comp" -o "${CONVOLUTION_SPV}"
    DEPENDS src/ops/convolution.comp VERBATIM)
add_library(vkdnn SHARED src/core/vkdnn.cpp src/core/runtime.cpp src/ops/cnn_stubs.cpp src/ops/forward.cpp src/ops/gemm.cpp "${FORWARD_SPV}" "${GEMM_SPV}" "${CONVOLUTION_SPV}" "${BACKWARD_SPV}")
target_include_directories(vkdnn PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(vkdnn PRIVATE VKDNN_BUILDING_LIBRARY)
target_link_libraries(vkdnn PRIVATE Vulkan::Vulkan)
target_include_directories(vkdnn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(vkdnn PRIVATE cxx_std_17)
set_target_properties(vkdnn PROPERTIES CXX_EXTENSIONS OFF)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
