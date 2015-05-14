# $Id: CMakeLists.txt 1218 2011-11-06 02:23:33Z justin $

if(LFL_ANDROID)
else(LFL_ANDROID)
set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS})
endif(LFL_ANDROID)

autoconf(./configure "${CONFIGURE_OPTIONS}" make)

