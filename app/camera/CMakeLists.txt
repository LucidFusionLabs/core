# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_camera STATIC_LIBRARY SOURCES null_camera.cpp)

if(LFL_IOS)
  lfl_add_target(app_avcapture_camera STATIC_LIBRARY SOURCES avcapture_camera.mm
                 COMPILE_OPTIONS -fobjc-abi-version=2 -fobjc-legacy-dispatch)
endif()

if(0)
  add_library(app_qtkit_camera STATIC qtkit_camera.mm)
  target_include_directories(app_qtkit_camera PUBLIC ${LFL_APP_INCLUDE})
  target_compile_definitions(app_qtkit_camera PUBLIC ${LFL_APP_DEF})
  target_compile_options(app_qtkit_camera PUBLIC -fobjc-abi-version=2 -fobjc-legacy-dispatch)
  target_link_libraries(app_qtkit_camera PUBLIC "-framework QTKit" "-framework CoreVideo")
endif()

if(LFL_OPENCV)
  lfl_add_target(app_opencv_camera STATIC_LIBRARY SOURCES opencv_camera.cpp
                 INCLUDE_DIRECTORIES ${OPENCV_INCLUDE} LINK_LIBRARIES ${OPENCV_LIB})
endif()
