# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFAPP_DEF} -fobjc-abi-version=2 -fobjc-legacy-dispatch)

set(LFOBJC_SRC lfobjc.mm)
if(LFL_OSXVIDEO)
set(LFOBJC_SRC ${LFOBJC_SRC} osx.mm)
endif()
if(LFL_IPHONE)
set(LFOBJC_SRC ${LFOBJC_SRC} iphone.mm)
endif()

add_library(lfapp_lfobjc STATIC ${LFOBJC_SRC})
