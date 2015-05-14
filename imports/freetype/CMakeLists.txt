# $Id: CMakeLists.txt 1004 2011-08-27 19:09:26Z justin $

if(WIN32)
 INCLUDE_EXTERNAL_MSPROJECT(freetype ${CMAKE_CURRENT_SOURCE_DIR}/builds/win32/vc2008/freetype.vcxproj)
else(WIN32)
 set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --enable-static=yes --enable-shared=no)
 autoconf(./configure "${CONFIGURE_OPTIONS}" make)
endif(WIN32)
