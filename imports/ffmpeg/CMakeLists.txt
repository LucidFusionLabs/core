# $Id: CMakeLists.txt 146 2009-10-13 05:36:56Z justin $

if(LFL_ANDROID)
set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --enable-cross-compile --enable-pic --disable-debug --disable-stripping --disable-bzlib --cross-prefix=${ANDROIDROOT}/bin/arm-linux-androideabi- --sysroot=${ANDROIDROOT}/sysroot --arch=arm5te --target-os=linux --disable-avdevice --enable-shared)
else(LFL_ANDROID)
set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --enable-gpl --enable-libx264 --enable-libmp3lame --extra-libs=-L../lame/libmp3lame/.libs --extra-libs=-L../x264 --extra-cflags=-I../lame/include --extra-cflags=-I../x264 --disable-sse --disable-outdev=sdl)
endif(LFL_ANDROID)

autoconf(./configure "${CONFIGURE_OPTIONS}" make)

