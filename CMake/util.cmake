# $Id: util.cmake 978 2011-08-14 03:13:42Z justin $

if(LFL_APPLE)
  set(so_prefix .)
  set(so_suffix .dylib)
else()
  set(so_prefix .so.)
  set(so_suffix)
endif()

macro(add_shared_library _var _file _ver)
  set(${_var} ${${_var}} ${_file}${so_prefix}${_ver}${so_suffix})
endmacro()

macro(add_dependency _target)
  if(${ARGN})
    add_dependencies(${_target} ${ARGN})
  endif() 
endmacro()

macro(copyfile _src _dst)
  execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND cp ${_src} ${_dst})
endmacro()
