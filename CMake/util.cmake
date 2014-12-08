# $Id: util.cmake 978 2011-08-14 03:13:42Z justin $

macro (copyfile _src _dst)

    execute_process(
        COMMAND
            cp ${_src} ${_dst}
        WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

endmacro (copyfile)

