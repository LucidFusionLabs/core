# $Id: Autoconf.cmake 1190 2011-10-23 03:59:29Z justin $

macro(autoconf _dir)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir})
  set(autoconf_built ${CMAKE_CURRENT_BINARY_DIR}/${_dir}/.built)
  if(NOT EXISTS ${autoconf_built})
    message(STATUS "run ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/configure ${CMAKE_CONFIGURE_OPTIONS} ${ARGN} in ${CMAKE_CURRENT_BINARY_DIR}/${_dir}")
    execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/configure ${CMAKE_CONFIGURE_OPTIONS} ${ARGN})
    message(STATUS "run make in ${CMAKE_CURRENT_BINARY_DIR}/${_dir}")
    execute_process(RESULT_VARIABLE autoconf_failed WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir}
                    COMMAND make)
    if(NOT autoconf_failed)
      execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir}
                      COMMAND touch .built)
    endif()
  endif()
endmacro()

macro(autoconf_exec _dir)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir})
  set(autoconf_built ${CMAKE_CURRENT_BINARY_DIR}/${_dir}/.built)
  if(NOT EXISTS ${autoconf_built})
    message(STATUS "run ${ARGN} in ${CMAKE_CURRENT_BINARY_DIR}/${_dir}")
    execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir}
                    COMMAND ${ARGN})
  endif()
endmacro()

macro(autoconf_exec_outfile _dir _file)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir})
  set(autoconf_built ${CMAKE_CURRENT_BINARY_DIR}/${_dir}/.built)
  if(NOT EXISTS ${autoconf_built})
    message(STATUS "run ${ARGN} ${_args} in ${CMAKE_CURRENT_BINARY_DIR}/${_dir}")
    execute_process(OUTPUT_FILE ${_file} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_dir}
                    COMMAND ${ARGN})
  endif()
endmacro()
