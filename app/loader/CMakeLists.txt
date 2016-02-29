# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

include_directories(${LFL_SOURCE_DIR})
add_library(app_simple_loader STATIC simple_loader.cpp)
add_library(app_simple_resampler STATIC simple_resampler.cpp)
add_library(app_null_png STATIC null_png.cpp)
add_library(app_null_jpeg STATIC null_jpeg.cpp)
add_library(app_null_gif STATIC null_gif.cpp)

if(LFL_PNG)
  add_library(app_libpng_png STATIC libpng_png.cpp)
  target_include_directories(app_libpng_png PUBLIC ${PNG_INCLUDE})
  target_link_libraries(app_libpng_png PUBLIC ${PNG_LIB})
endif()

if(LFL_GIF)
  add_library(app_giflib_gif STATIC giflib_gif.cpp)
  target_include_directories(app_giflib_gif PUBLIC ${PNG_INCLUDE})
  target_link_libraries(app_giflib_gif PUBLIC ${PNG_LIB})
endif()

if(LFL_JPEG)
  add_library(app_libjpeg_jpeg STATIC giflib_gif.cpp)
  target_include_directories(app_libjpeg_jpeg PUBLIC ${JPEG_INCLUDE})
  target_link_libraries(app_libjpeg_jpeg PUBLIC ${JPEG_LIB})
endif()

if(LFL_FFMPEG)
  add_library(app_ffmpeg_loader STATIC ffmpeg_loader.cpp)
  target_include_directories(app_ffmpeg_loader PUBLIC ${FFMPEG_INCLUDE})
  target_link_libraries(app_ffmpeg_loader PUBLIC ${FFMPEG_LIB})

  add_library(app_ffmpeg_resampler STATIC ffmpeg_loader.cpp)
  target_include_directories(app_ffmpeg_resampler PUBLIC ${FFMPEG_INCLUDE})
  target_link_libraries(app_ffmpeg_resampler PUBLIC ${FFMPEG_LIB})
endif()
