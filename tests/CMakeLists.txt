add_executable(vkdnn_memory_test runtime_memory_test.c)
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

if(WIN32)
    set_target_properties(vkdnn_memory_test vkdnn_device_query_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
