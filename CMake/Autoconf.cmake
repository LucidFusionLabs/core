# $Id: Autoconf.cmake 1190 2011-10-23 03:59:29Z justin $

macro (autoconf_make _make _makefile _target)
    set(autoconf_built ${CMAKE_CURRENT_BINARY_DIR}/.built)
    if(NOT EXISTS ${autoconf_built})
        MESSAGE(STATUS "run ${_make} -f ${CMAKE_CURRENT_SOURCE_DIR}/${_makefile} ${_target} in ${CMAKE_CURRENT_BINARY_DIR}")
        execute_process(
                COMMAND
                        ${_make} -f ${CMAKE_CURRENT_SOURCE_DIR}/${_makefile} ${_target}
                WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}
                RESULT_VARIABLE
                        autoconf_failed
        )
        if(NOT autoconf_failed)
            execute_process(
                    COMMAND
                            touch .built
                    WORKING_DIRECTORY
                            ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif(NOT autoconf_failed)
    endif(NOT EXISTS ${autoconf_built})
endmacro (autoconf_make)

macro (autoconf _configure _options _make)
    set(autoconf_built ${CMAKE_CURRENT_BINARY_DIR}/.built)
    if(NOT EXISTS ${autoconf_built})
        MESSAGE(STATUS "run ${CMAKE_CURRENT_SOURCE_DIR}/${_configure} ${CMAKE_CONFIGURE_OPTIONS} ${_options} in ${CMAKE_CURRENT_BINARY_DIR}")
        execute_process(
                COMMAND
                        ${CMAKE_CURRENT_SOURCE_DIR}/${_configure} ${CMAKE_CONFIGURE_OPTIONS} ${_options}
                WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}
        )
        MESSAGE(STATUS "run ${_make} in ${CMAKE_CURRENT_BINARY_DIR}")
        execute_process(
                COMMAND
                        ${_make} 
                WORKING_DIRECTORY
                        ${CMAKE_CURRENT_BINARY_DIR}
                RESULT_VARIABLE
                        autoconf_failed
        )
        if(NOT autoconf_failed)
            execute_process(
                    COMMAND
                            touch .built
                    WORKING_DIRECTORY
                            ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif(NOT autoconf_failed)
    endif(NOT EXISTS ${autoconf_built})
endmacro (autoconf)

macro (autoconf_exec _cmd)
    set(autoconf_built ${CMAKE_CURRENT_BINARY_DIR}/.built)
    if(NOT EXISTS ${autoconf_built})
    MESSAGE(STATUS "run ${_cmd} in ${CMAKE_CURRENT_BINARY_DIR}")
    execute_process(COMMAND ${_cmd}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif(NOT EXISTS ${autoconf_built})
endmacro (autoconf_exec)
