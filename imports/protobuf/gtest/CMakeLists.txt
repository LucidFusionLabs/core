# $Id: CMakeLists.txt 1004 2011-08-27 19:09:26Z justin $

set(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS} --enable-static=yes --enable-shared=no)
autoconf(./configure "${CONFIGURE_OPTIONS}" make)

