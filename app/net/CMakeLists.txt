# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_ssl STATIC_LIBRARY SOURCES null_ssl.cpp)

if(LFL_OPENSSL)
  lfl_add_target(app_openssl_ssl STATIC_LIBRARY SOURCES openssl_ssl.cpp
                 INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE} LINK_LIBRARIES ${OPENSSL_LIB})
endif()

if(LFL_FFMPEG)
  lfl_add_target(app_ffmpeg_stream STATIC_LIBRARY SOURCES ffmpeg_stream.cpp
                 INCLUDE_DIRECTORIES ${FFMPEG_INCLUDE} LINK_LIBRARIES ${FFMPEG_LIB})
endif()
