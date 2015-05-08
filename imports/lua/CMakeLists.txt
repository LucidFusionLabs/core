# $Id: CMakeLists.txt 146 2009-10-13 05:36:56Z justin $

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
autoconf_make("make" "Makefile" "macosx")
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
autoconf_make("make" "Makefile" "linux")
else(CMAKE_SYSTEM_NAME MATCHES "Darwin")
autoconf_make("make" "Makefile" "generic")
endif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
