# $Id: CMakeLists.txt 148 2009-10-13 05:50:37Z justin $

execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMAND mkdir -p doc/man/man3
                COMMAND mkdir -p doc/ext)
autoconf("./configure" "" "make")

