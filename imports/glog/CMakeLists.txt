# $Id: CMakeLists.txt 1004 2011-08-27 19:09:26Z justin $

if(LFL_ANDROID)
 lfl_export_toolchain()
elseif(LFL_IPHONE)
 lfl_export_toolchain()
else(LFL_ANDROID)
 set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS})
endif(LFL_ANDROID)

autoconf(./configure "${CONFIGURE_OPTIONS}" make)

