# $Id: CMakeLists.txt 146 2009-10-13 05:36:56Z justin $

set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --prefix=${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --host x86_64-apple-darwin)
endif()

autoconf(./configure "${CONFIGURE_OPTIONS}" make)

