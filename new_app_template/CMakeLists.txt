# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_PNG 1)
set(LFL_GLES2 1)
set(LFL_LFLINPUT 1)
set(LFL_LFLVIDEO 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
    set(LFL_GLOG 1)
endif(LFL_ANDROID)

lfl_project($BINNAME)
add_subdirectory(../core/lfapp ${CMAKE_CURRENT_BINARY_DIR}/$BINNAME_lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

lfapp_add_target($BINNAME $BINNAME.cpp)
lfapp_post_build_start($BINNAME $BINNAME $PKGNAME)
