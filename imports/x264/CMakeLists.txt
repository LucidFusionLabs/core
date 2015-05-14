# $Id: CMakeLists.txt 146 2009-10-13 05:36:56Z justin $

autoconf("./configure" "--enable-shared" "make")

execute_process(COMMAND ln -s libx264.so.142 libx264.so WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

