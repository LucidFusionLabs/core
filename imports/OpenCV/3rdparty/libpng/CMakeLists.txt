# $Id: CMakeLists.txt 1247 2011-11-19 01:48:49Z justin $

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
set(LIBPNG_SRC png.c pngerror.c pnggccrd.c pngget.c pngmem.c pngpread.c pngread.c pngrio.c pngrtran.c pngrutil.c pngset.c pngtrans.c pngvcrd.c pngwio.c pngwrite.c pngwtran.c pngwutil.c)
add_library(libpng STATIC ${LIBPNG_SRC})

