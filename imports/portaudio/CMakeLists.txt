# $Id: CMakeLists.txt 47 2009-07-19 22:26:30Z justin $

if(WIN32)
INCLUDE_EXTERNAL_MSPROJECT(portaudio ${CMAKE_CURRENT_SOURCE_DIR}/build/msvc/portaudio.vcxproj)
else(WIN32)
set(CONFIGURE_OPTIONS "--disable-mac-universal")
autoconf("./configure" ${CONFIGURE_OPTIONS} "make")
endif(WIN32)
