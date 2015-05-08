# $Id: CMakeLists.txt 1004 2011-08-27 19:09:26Z justin $

if(WIN32)
 INCLUDE_EXTERNAL_MSPROJECT(libprotobuf-lite ${CMAKE_CURRENT_SOURCE_DIR}/vsprojects/libprotobuf-lite.vcxproj)
else(WIN32)
 set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --enable-static=yes --enable-shared=no)
 if(LFL_ANDROID OR LFL_IPHONE)
  set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --with-protoc=${CMAKE_CURRENT_SOURCE_DIR}/bin/osx64/protoc)
 endif()
 autoconf(./configure "${CONFIGURE_OPTIONS}" make)
endif(WIN32)
