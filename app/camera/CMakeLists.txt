# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

include_directories(${LFL_SOURCE_DIR})

add_library(app_null_camera STATIC null_camera.cpp)

if(LFL_OSX AND LFL64)
  add_library(app_qtkit_camera STATIC qtkit_camera.mm)
  target_compile_options(app_qtkit_camera PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
  target_link_libraries(app_qtkit_camera PUBLIC "-framework QTKit" "-framework CoreVideo")
endif()

if(LFL_OPENCV)
  add_library(app_opencv_camera STATIC opencv_camera.cpp)
  target_include_directories(app_opencv_camera PUBLIC ${OPENCV_INCLUDE})
  target_link_libraries(app_opencv_camera PUBLIC ${OPENCV_LIB})
endif()
