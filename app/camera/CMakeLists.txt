# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_camera STATIC_LIBRARY SOURCES null_camera.cpp)

if(LFL_IPHONE)
  lfl_add_target(app_avcapture_camera STATIC_LIBRARY SOURCES avcapture_camera.mm
                 COMPILE_OPTIONS -fobjc-abi-version=2 -fobjc-legacy-dispatch)
endif()

if(LFL_OSX AND LFL64)
  lfl_add_target(app_qtkit_camera STATIC_LIBRARY SOURCES qtkit_camera.mm
                 COMPILE_OPTIONS -fobjc-abi-version=2 -fobjc-legacy-dispatch
                 LINK_LIBRARIES "-framework QTKit" "-framework CoreVideo")
endif()

if(LFL_OPENCV)
  lfl_add_target(app_opencv_camera STATIC_LIBRARY SOURCES opencv_camera.cpp
                 INCLUDE_DIRECTORIES ${OPENCV_INCLUDE} LINK_LIBRARIES ${OPENCV_LIB})
endif()
