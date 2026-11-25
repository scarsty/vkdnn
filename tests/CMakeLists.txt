add_executable(vkdnn_memory_test runtime_memory_test.c)
add_executable(vkdnn_forward_benchmark forward_benchmark.cpp)
add_executable(vkdnn_backward_benchmark backward_benchmark.cpp)
target_link_libraries(vkdnn_backward_benchmark PRIVATE vkdnn)
target_compile_features(vkdnn_backward_benchmark PRIVATE cxx_std_17)
set_target_properties(vkdnn_backward_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(vkdnn_forward_benchmark PRIVATE vkdnn)
target_compile_features(vkdnn_forward_benchmark PRIVATE cxx_std_17)
set_target_properties(vkdnn_forward_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(vkdnn_convolution_query_test convolution_query_test.c)
target_link_libraries(vkdnn_convolution_query_test PRIVATE vkdnn)
set_target_properties(vkdnn_convolution_query_test PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
add_test(NAME vkdnn.convolution_query COMMAND vkdnn_convolution_query_test)
set_tests_properties(vkdnn.convolution_query PROPERTIES SKIP_RETURN_CODE 77)
target_link_libraries(vkdnn_memory_test PRIVATE vkdnn)
set_target_properties(vkdnn_memory_test PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX
)
add_test(NAME vkdnn.runtime_memory COMMAND vkdnn_memory_test)
set_tests_properties(vkdnn.runtime_memory PROPERTIES SKIP_RETURN_CODE 77)

add_executable(vkdnn_device_query_test device_query_test.c)
target_link_libraries(vkdnn_device_query_test PRIVATE vkdnn)
set_target_properties(vkdnn_device_query_test PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX
)
add_test(NAME vkdnn.device_query COMMAND vkdnn_device_query_test)
set_tests_properties(vkdnn.device_query PROPERTIES SKIP_RETURN_CODE 77)

add_executable(vkdnn_cnn_stub_test cnn_stub_test.c)
target_link_libraries(vkdnn_cnn_stub_test PRIVATE vkdnn)
set_target_properties(vkdnn_cnn_stub_test PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF LINKER_LANGUAGE CXX
)
add_test(NAME vkdnn.cnn_stub COMMAND vkdnn_cnn_stub_test)
add_executable(vkdnn_cnn_forward_test cnn_forward_test.cpp)
target_link_libraries(vkdnn_cnn_forward_test PRIVATE vkdnn)
target_compile_features(vkdnn_cnn_forward_test PRIVATE cxx_std_17)
add_test(NAME vkdnn.cnn_forward COMMAND vkdnn_cnn_forward_test)
set_tests_properties(vkdnn.cnn_forward PROPERTIES SKIP_RETURN_CODE 77)

add_executable(vkdnn_gemm_test gemm_test.cpp)
target_link_libraries(vkdnn_gemm_test PRIVATE vkdnn)
target_compile_features(vkdnn_gemm_test PRIVATE cxx_std_17)
add_test(NAME vkdnn.gemm COMMAND vkdnn_gemm_test)
set_tests_properties(vkdnn.gemm PROPERTIES SKIP_RETURN_CODE 77)
add_executable(vkdnn_convolution_test convolution_test.cpp)
target_link_libraries(vkdnn_convolution_test PRIVATE vkdnn)
target_compile_features(vkdnn_convolution_test PRIVATE cxx_std_17)
add_test(NAME vkdnn.convolution COMMAND vkdnn_convolution_test)
set_tests_properties(vkdnn.convolution PROPERTIES SKIP_RETURN_CODE 77)

add_executable(vkdnn_activation_backward_test activation_backward_test.cpp)
target_link_libraries(vkdnn_activation_backward_test PRIVATE vkdnn)
target_compile_features(vkdnn_activation_backward_test PRIVATE cxx_std_17)
add_test(NAME vkdnn.activation_backward COMMAND vkdnn_activation_backward_test)
set_tests_properties(vkdnn.activation_backward PROPERTIES SKIP_RETURN_CODE 77)

add_executable(vkdnn_cnn_backward_test cnn_backward_test.cpp)
target_link_libraries(vkdnn_cnn_backward_test PRIVATE vkdnn)
target_compile_features(vkdnn_cnn_backward_test PRIVATE cxx_std_17)
add_test(NAME vkdnn.cnn_backward COMMAND vkdnn_cnn_backward_test)
set_tests_properties(vkdnn.cnn_backward PROPERTIES SKIP_RETURN_CODE 77)
if(WIN32)
    set_target_properties(vkdnn_memory_test vkdnn_device_query_test vkdnn_cnn_stub_test vkdnn_cnn_forward_test vkdnn_gemm_test vkdnn_convolution_test vkdnn_activation_backward_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    set_target_properties(vkdnn_cnn_backward_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
