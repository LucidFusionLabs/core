# $Id: lfapp.h 1335 2014-12-02 04:13:46Z justin $
# Copyright (C) 2009 Lucid Fusion Labs

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

macro(get_shared_library_name _var _file _ver)
  if(LFL_APPLE)
    set(_so_prefix .)
    set(_so_suffix .dylib)
  else()
    set(_so_prefix .so.)
    set(_so_suffix)
  endif()
  set(${_var} ${_file}${_so_prefix}${_ver}${_so_suffix})
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

macro(lfl_project _name)
  project(${_name})
  set(LFL_PROJECT ${_name})
  set(LFL_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(LFL_PROJECT_BINDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro(lfl_project)

function(lfl_add_target _name)
  set(options EXECUTABLE SHARED_LIBRARY STATIC_LIBRARY WIN32)
  set(one_value_args)
  set(multi_value_args DEPENDENCIES COMPILE_DEFINITIONS COMPILE_OPTIONS INCLUDE_DIRECTORIES
      SOURCES LINK_LIBRARIES)
  cmake_parse_arguments("" "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT _WIN32)
    set(_WIN32)
  endif()

  if(_EXECUTABLE)
    add_executable(${_name} ${_WIN32} ${_SOURCES})
  elseif(_STATIC_LIBRARY)
    add_library(${_name} STATIC ${_SOURCES})
  elseif(_SHARED_LIBRARY)
    add_library(${_name} SHARED ${_SOURCES})
  else()
    message(FATAL_ERROR "lfl_add_target without EXECUTABLE, STATIC_LIBRARY, or SHARED_LIBRARY")
  endif()

  add_dependencies(${_name} lfl_app ${_DEPENDENCIES})
  target_include_directories(${_name} PUBLIC ${LFL_APP_INCLUDE} ${_INCLUDE_DIRECTORIES})
  target_compile_definitions(${_name} PUBLIC ${LFL_APP_DEF} ${_COMPILE_DEFINITIONS})
  target_compile_options(${_name} PUBLIC ${_COMPILE_OPTIONS})
  target_use_precompiled_header(${_name} core/app/app.h FORCEINCLUDE)
  if(_LINK_LIBRARIES)
    target_link_libraries(${_name} PUBLIC ${_LINK_LIBRARIES})
  endif()

  if(_EXECUTABLE AND LFL_ADD_BITCODE_TARGETS)
    add_bitcode(${_name}_bitcode ${_name})
    add_executable(${_name}_designer ${_WIN32} ${LFL_SOURCE_DIR}/core/app/bindings/llvm_jit.cpp)
    add_dependencies(${_name}_designer ${_name})
    target_compile_definitions(${_name}_designer PUBLIC ${LFL_APP_DEF} ${_COMPILE_DEFINITIONS})
    target_include_directories(${_name}_designer PUBLIC ${LFL_APP_INCLUDE} ${LIBCLANG_INCLUDE})
    target_compile_options(${_name}_designer PUBLIC ${_COMPILE_OPTIONS})
    target_link_libraries(${_name}_designer PUBLIC ${_LINK_LIBRARIES} app_libarchive_archive app_clang_tu)
  endif()
endfunction()
