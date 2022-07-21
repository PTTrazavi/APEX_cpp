

if(MSVC)
    file(GLOB_RECURSE tiny_dnn_headers "${CMAKE_SOURCE_DIR}/tiny_dnn/*.h")
    file(RELATIVE_PATH tiny_dnn_relpath ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/tiny_dnn")
    string(REPLACE "/" "\\" tiny_dnn_relpath "${tiny_dnn_relpath}")
    foreach(_source IN ITEMS ${tiny_dnn_headers})
        file(RELATIVE_PATH _source_rel ${CMAKE_CURRENT_SOURCE_DIR} "${_source}")
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        string(REPLACE ${tiny_dnn_relpath} "tiny_dnn" _source_path_msvc ${_source_path_msvc})
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endif()

if(USE_SERIALIZER)

    add_executable(example_mnist_train mnist/train.cpp ${tiny_dnn_headers})
    target_link_libraries(example_mnist_train
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

    add_executable(example_mnist_test mnist/test.cpp ${tiny_dnn_headers})
    target_link_libraries(example_mnist_test
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

    cotire(example_mnist_train example_mnist_test)
endif()