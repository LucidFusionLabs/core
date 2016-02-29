# $Id: CMakeLists.txt 323 2011-03-07 02:31:15Z justin $

include_directories(..)
add_definitions(-DLFL_BUILD)

if(LFL_CUDA)
set(CUDA_64_BIT_DEVICE_CODE OFF)
CUDA_ADD_LIBRARY(lfcuda lfcuda.cu)
endif()
