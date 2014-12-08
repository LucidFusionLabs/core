# $Id: CMakeLists.txt 146 2009-10-13 05:36:56Z justin $

if(WIN32)
INCLUDE_EXTERNAL_MSPROJECT(libarchive ${CMAKE_CURRENT_SOURCE_DIR}/build/windows/vc90/libarchive.vcxproj)
else(WIN32)
autoconf("./configure" "" "make")
endif(WIN32)
