# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFAPP_DEF} -fobjc-abi-version=2 -fobjc-legacy-dispatch)
include_directories(${SDL_INCLUDE})
add_library(${LFL_PROJECT}_lfobjc STATIC lfobjc.mm camera.mm EAGLView.mm)

