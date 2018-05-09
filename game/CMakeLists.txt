# $Id$

lfl_project(game)
lfl_add_target(game STATIC_LIBRARY SOURCES game.cpp)

add_subdirectory(master)
