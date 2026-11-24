cmake_minimum_required(VERSION 3.20)
project(vkdnn VERSION 0.1.0 LANGUAGES C CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

find_package(Vulkan REQUIRED)

add_library(vkdnn SHARED src/core/vkdnn.cpp src/core/runtime.cpp)
target_compile_definitions(vkdnn PRIVATE VKDNN_BUILDING_LIBRARY)
target_link_libraries(vkdnn PRIVATE Vulkan::Vulkan)
target_include_directories(vkdnn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(vkdnn PRIVATE cxx_std_17)
set_target_properties(vkdnn PROPERTIES CXX_EXTENSIONS OFF)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
