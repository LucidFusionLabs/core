# $Id$

if(LFL_CUDA)
  lfl_add_target(app_cuda STATIC_LIBRARY SOURCES cuda.cpp
                 INCLUDE_DIRECTORIES ${CUDA_INCLUDE} LINK_LIBRARIES ${CUDA_LIB})

  set(CUDA_64_BIT_DEVICE_CODE OFF)
  CUDA_ADD_LIBRARY(cudaspeech speech.cu)
  target_compile_definitions(cudaspeech PUBLIC -DLFL_BUILD)
  target_include_directories(cudaspeech PUBLIC ${LFL_SOURCE_DIR})
endif()
