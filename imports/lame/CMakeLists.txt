# $Id: CMakeLists.txt 47 2009-07-19 22:26:30Z justin $

autoconf("./configure" "" "make")

execute_process(COMMAND ln -s . include/lame WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

