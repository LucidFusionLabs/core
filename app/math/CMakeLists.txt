# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_fft STATIC_LIBRARY SOURCES null_fft.cpp)
lfl_add_target(app_null_matrix STATIC_LIBRARY SOURCES null_matrix.cpp)

if(LFL_OPENCV)
  lfl_add_target(app_opencv_matrix STATIC_LIBRARY SOURCES opencv_matrix.cpp
                 INCLUDE_DIRECTORIES ${OPENCV_INCLUDE} LINK_LIBRARIES ${OPENCV_LIB})
endif()

if(LFL_FFMPEG)
  lfl_add_target(app_ffmpeg_fft STATIC_LIBRARY SOURCES ffmpeg_fft.cpp
                 INCLUDE_DIRECTORIES ${FFMPEG_INCLUDE} LINK_LIBRARIES ${FFMPEG_LIB})
endif()
