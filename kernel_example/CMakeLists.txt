configure_file(
        kernel_example.h.in
        "${PROJECT_BINARY_DIR}/kernel_example.h"
)
include_directories("${PROJECT_BINARY_DIR}")
if (GOOGLE_CUDA)
    find_package(CUDA QUIET REQUIRED)
    cuda_add_library(kernel_example SHARED kernel_example.cu.cc)
else()
    add_library(kernel_example SHARED kernel_example.cc kernel_example.h)
    message("TF Libs are " ${TF_LIBS})
    target_link_libraries(kernel_example PUBLIC ${TF_LIBS})
    set_target_properties(kernel_example PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
    set_target_properties(kernel_example PROPERTIES PREFIX "")
endif()

