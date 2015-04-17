# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFAPP_DEF} -fobjc-abi-version=2 -fobjc-legacy-dispatch)

if(LFL_SDL)
include_directories(${SDL_INCLUDE})
endif(LFL_SDL)

set(LFOBJC_SRC lfobjc.mm)
if(LFL_OSXVIDEO)
set(LFOBJC_SRC ${LFOBJC_SRC} osx.mm)
endif()
if(LFL_IPHONE)
set(LFOBJC_SRC ${LFOBJC_SRC} iphone.mm EAGLView.mm)
endif()
if(LFL_CAMERA)
set(LFOBJC_SRC ${LFOBJC_SRC} camera.mm)
endif()

add_library(${LFL_PROJECT}_lfobjc STATIC ${LFOBJC_SRC})

