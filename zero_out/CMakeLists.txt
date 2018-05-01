add_library(zero_out SHARED zero_out.cc)
target_link_libraries(zero_out PUBLIC ${TF_LIBS})
set_target_properties(zero_out PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(zero_out PROPERTIES PREFIX "")
