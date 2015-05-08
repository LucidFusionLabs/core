# $Id: CMakeLists.txt 146 2009-10-13 05:36:56Z justin $

if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND NOT LFL_IPHONE)
 set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --host x86_64-apple-darwin)
endif()

autoconf(./configure "${CONFIGURE_OPTIONS}" make)

