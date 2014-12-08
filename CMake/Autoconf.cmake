# $Id: Autoconf.cmake 1190 2011-10-23 03:59:29Z justin $

macro (autoconf_make _dir _make _makefile _target)
    set(autoconf_built ${CMAKE_CURRENT_SOURCE_DIR}/.built)
    if(NOT EXISTS ${autoconf_built})
        execute_process(
                COMMAND
                        ${_make} -f ${_makefile} ${_target}
                WORKING_DIRECTORY
                        ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE
                        autoconf_failed
        )
        if(NOT autoconf_failed)
            execute_process(
                    COMMAND
                            touch .built
                    WORKING_DIRECTORY
                            ${CMAKE_CURRENT_SOURCE_DIR}
            )
        endif(NOT autoconf_failed)
    endif(NOT EXISTS ${autoconf_built})
endmacro (autoconf_make)

macro (autoconf _configure _options _make)
    set(autoconf_built ${CMAKE_CURRENT_SOURCE_DIR}/.built)
    if(NOT EXISTS ${autoconf_built})
        execute_process(
                COMMAND
                        ${_configure} ${CMAKE_CONFIGURE_OPTIONS} ${_options}
                WORKING_DIRECTORY
                        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        execute_process(
                COMMAND
                        ${_make} 
                WORKING_DIRECTORY
                        ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE
                        autoconf_failed
        )
        if(NOT autoconf_failed)
            execute_process(
                    COMMAND
                            touch .built
                    WORKING_DIRECTORY
                            ${CMAKE_CURRENT_SOURCE_DIR}
            )
        endif(NOT autoconf_failed)
    endif(NOT EXISTS ${autoconf_built})
endmacro (autoconf)

macro (autoconf_exec _cmd)
    set(autoconf_built ${CMAKE_CURRENT_SOURCE_DIR}/.built)
    if(NOT EXISTS ${autoconf_built})
    execute_process(COMMAND ${_cmd}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif(NOT EXISTS ${autoconf_built})
endmacro (autoconf_exec)

