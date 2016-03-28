# $Id: util.cmake 978 2011-08-14 03:13:42Z justin $

if(LFL_APPLE)
  set(so_prefix .)
  set(so_suffix .dylib)
else()
  set(so_prefix .so.)
  set(so_suffix)
endif()

macro(get_shared_library_name _var _file _ver)
  set(${_var} ${_file}${so_prefix}${_ver}${so_suffix})
endmacro()

macro(add_shared_library _var _file _ver)
  get_shared_library_name(SHARED_LIBRARY_NAME ${_file} ${_ver})
  set(${_var} ${${_var}} ${SHARED_LIBRARY_NAME})
endmacro()

macro(add_dependency _target)
  if(${ARGN})
    add_dependencies(${_target} ${ARGN})
  endif() 
endmacro()
