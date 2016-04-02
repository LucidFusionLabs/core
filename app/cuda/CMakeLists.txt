# $Id: CMakeLists.txt 323 2011-03-07 02:31:15Z justin $

if(LFL_CUDA)
  set(CUDA_64_BIT_DEVICE_CODE OFF)
  CUDA_ADD_LIBRARY(lfcuda lfcuda.cu)
  target_compile_definitions(lfcuda PUBLIC -DLFL_BUILD)
  target_include_directories(lfcuda PUBLIC ${LFL_SOURCE_DIR})
endif()
