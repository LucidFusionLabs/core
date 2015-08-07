# $Id: CMakeLists.txt 323 2011-03-07 02:31:15Z justin $

add_definitions(-DLFL_BUILD)
include_directories(../lflpub/ ..)

set(CUDA_64_BIT_DEVICE_CODE OFF)
CUDA_ADD_LIBRARY(lfcuda lfcuda.cu)

# test
# add_executable(test test.cpp)
# target_link_libraries(test ${LFAPP_LIB})

