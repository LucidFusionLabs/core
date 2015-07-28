# $Id: CMakeLists.txt 146 2009-10-13 05:36:56Z justin $

PROJECT(giflib C)

SET(GIFLIB_PUBLIC_HEADERS
	lib/gif_lib.h
)
SET(GIFLIB_PRIVATE_HEADERS
	lib/gif_lib_private.h
	lib/gif_hash.h
)

SET(GIFLIB_SRCS
	lib/dgif_lib.c
	lib/egif_lib.c
	lib/gif_err.c
	lib/gif_font.c
	lib/gif_hash.c
	lib/gifalloc.c
	lib/quantize.c
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lib)
ADD_LIBRARY(libgif STATIC ${GIFLIB_SRCS} ${GIFLIB_PRIVATE_HEADERS} ${GIFLIB_PUBLIC_HEADERS})

