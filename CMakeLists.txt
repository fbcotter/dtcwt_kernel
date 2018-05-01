cmake_minimum_required(VERSION 2.8)
project(dtcwt_kernel)

# C++11 required for tensorflow
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -O2 ${CMAKE_CXX_FLAGS}")

# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC")

# Get the include and link directories for tensorflow
set(PYTHON_DIR "/home/fbc23/.pyenv/versions/phdnets/")
#execute_process(COMMAND
#        ${PYTHON_DIR}/bin/python -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))"
#        OUTPUT_VARIABLE TF_CFLAGS)
#execute_process(COMMAND
#        ${PYTHON_DIR}/bin/python -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))"
#        OUTPUT_VARIABLE TF_LFLAGS)
#string(REPLACE "-I" "" TF_CFLAGS ${TF_CFLAGS})
#string(REGEX MATCHALL "-L[/a-zA-z0-9.-]+" TF_LDIRS ${TF_LFLAGS})
#string(REPLACE "-L" "" TF_LDIRS ${TF_LDIRS})
#string(REGEX MATCHALL "-l[a-zA-z0-9.-]+" TF_LIBS ${TF_LFLAGS})
#string(REPLACE "-l" "" TF_LIBS ${TF_LIBS})
include_directories(${PYTHON_DIR}/lib/python3.5/site-packages/tensorflow/include
                    ${PYTHON_DIR}/lib/python3.5/site-packages/tensorflow/include/external/nsync/public)
link_directories(${PYTHON_DIR}/lib/python3.5/site-packages/tensorflow)
set(TF_LIBS "tensorflow_framework")
link_directories(/usr/local/cuda-9.0/lib64)

option (GOOGLE_CUDA "Compile kernels for gpu implementation" OFF)
message("Compile GPU is " ${GOOGLE_CUDA})

add_subdirectory(zero_out)
add_subdirectory(kernel_example)
