# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

include_directories(${LFL_SOURCE_DIR})
add_library(app_null_fft STATIC null_fft.cpp)
add_library(app_null_matrix STATIC null_matrix.cpp)

if(LFL_OPENCV)
  add_library(app_opencv_matrix STATIC opencv_matrix.cpp)
  target_include_directories(app_opencv_matrix PUBLIC ${OPENCV_INCLUDE})
  target_link_libraries(app_opencv_matrix PUBLIC ${OPENCV_LIB})
  add_external_dep(app_opencv_matrix OpenCV)
endif()

if(LFL_FFMPEG)
  add_library(app_ffmpeg_fft STATIC ffmpeg_fft.cpp)
  target_include_directories(app_ffmpeg_fft PUBLIC ${FFMPEG_INCLUDE})
  target_link_libraries(app_ffmpeg_fft PUBLIC ${FFMPEG_LIB})
endif()
