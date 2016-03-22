# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})
lfl_add_library(app_null_ssl STATIC null_ssl.cpp)

if(LFL_OPENSSL)
  lfl_add_library(app_openssl_ssl STATIC openssl_ssl.cpp)
  target_include_directories(app_openssl_ssl PUBLIC ${OPENSSL_INCLUDE})
  target_link_libraries(app_openssl_ssl PUBLIC ${OPENSSL_LIB})
endif()

if(LFL_FFMPEG)
  lfl_add_library(app_ffmpeg_stream STATIC ffmpeg_stream.cpp)
  target_include_directories(app_ffmpeg_stream PUBLIC ${FFMPEG_INCLUDE})
  target_link_libraries(app_ffmpeg_stream PUBLIC ${FFMPEG_LIB})
endif()
