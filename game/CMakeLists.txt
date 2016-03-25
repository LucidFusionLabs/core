# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

lfl_project(game)
add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})

lfl_add_library(game STATIC game.cpp)
