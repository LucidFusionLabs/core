# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_EASTL 1)
set(LFL_HEADLESS 1)

if(LFL_ANDROID)
elseif(LFL_IPHONE)
else(LFL_ANDROID)
 set(LFL_GLOG 1)
endif(LFL_ANDROID)

lfl_project(cluster)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

# trainer
add_executable(cluster-trainer trainer.cpp ${SDL_MAIN})
set_target_properties(cluster-trainer PROPERTIES OUTPUT_NAME trainer)
target_link_libraries(cluster-trainer ${LFAPP_LIB})
lfapp_post_build_start(cluster-trainer ClusterTrainer)

