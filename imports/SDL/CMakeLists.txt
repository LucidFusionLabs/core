# $Id: CMakeLists.txt 625 2010-12-13 04:47:37Z vmy $

if(LFL_HEADLESS)
set(CONFIGURE_OPTIONS --disable-video-x11 --disable-render-x11)
else(LFL_HEADLESS)
set(CONFIGURE_OPTIONS --disable-input-events)
endif(LFL_HEADLESS)

autoconf("./configure" "${CONFIGURE_OPTIONS}" "make")

