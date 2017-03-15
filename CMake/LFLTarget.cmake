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

macro(get_static_library_name _var _file)
  if(LFL_WINDOWS)
    set(${_var} ${_file}.lib)
  else()
    set(${_var} lib${_file}.a)
  endif()
endmacro()

macro(get_unversioned_shared_library_name _var _file)
  if(LFL_APPLE)
    set(${_var} ${_file}.dylib)
  else()
    set(${_var} ${_file}.so)
  endif()
endmacro()

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

macro(add_windows_dependency _target)
  if(LFL_WINDOWS)
    add_dependencies(${_target} ${ARGN})
  endif() 
endmacro()

macro(add_interface_include_directory _target _dir)
  file(MAKE_DIRECTORY ${_dir})
  set_property(TARGET ${_target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${_dir})
endmacro()

macro(lfl_project _name)
  project(${_name})
  set(LFL_PROJECT ${_name})
  set(LFL_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(LFL_PROJECT_BINDIR ${CMAKE_CURRENT_BINARY_DIR})
  set(MACOSX_BUNDLE_GUI_IDENTIFIER ${ARGN})
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
    if(_WIN32)
      add_executable(${_name} WIN32 ${_SOURCES})
    else()
      add_executable(${_name} ${_SOURCES})
    endif()
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
  target_compile_options(${_name} PUBLIC ${LFL_APP_CFLAGS} ${_COMPILE_OPTIONS})
  target_use_precompiled_header(${_name} core/app/app.h FORCEINCLUDE)
  set_property(TARGET ${_name} PROPERTY NO_SYSTEM_FROM_IMPORTED ON)
  if(_LINK_LIBRARIES)
    target_link_libraries(${_name} PUBLIC ${_LINK_LIBRARIES})
  endif()

  if(_EXECUTABLE AND LFL_IOS AND LFL_XCODE)
    set_target_properties(${_name} PROPERTIES XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ${LFL_IOS_CERT})
  endif()

  if(_EXECUTABLE AND LFL_ADD_BITCODE_TARGETS)
    add_library(${_name}_bitcode STATIC ${_SOURCES})
    target_compile_options(${_name}_bitcode PUBLIC ${_COMPILE_OPTIONS} -emit-llvm -fno-use-cxa-atexit)
    target_compile_definitions(${_name}_bitcode PUBLIC ${LFL_APP_DEF} ${_COMPILE_DEFINITIONS})
    target_include_directories(${_name}_bitcode PUBLIC ${LFL_APP_INCLUDE} ${_INCLUDE_DIRECTORIES})
    target_use_precompiled_header(${_name}_bitcode core/app/app.h FORCEINCLUDE)
    set_property(TARGET ${_name}_bitcode PROPERTY NO_SYSTEM_FROM_IMPORTED ON)
    add_dependencies(${_name}_bitcode ${_name})

    add_executable(${_name}_designer ${_WIN32} ${LFL_SOURCE_DIR}/core/app/bindings/llvm_jit.cpp)
    target_compile_options(${_name}_designer PUBLIC ${_COMPILE_OPTIONS})
    target_compile_definitions(${_name}_designer PUBLIC ${LFL_APP_DEF} ${_COMPILE_DEFINITIONS})
    target_include_directories(${_name}_designer PUBLIC ${LFL_APP_INCLUDE} ${LIBCLANG_INCLUDE})
    target_link_libraries(${_name}_designer PUBLIC ${_LINK_LIBRARIES} app_libarchive_archive app_clang_tu)
    set_property(TARGET ${_name}_designer PROPERTY NO_SYSTEM_FROM_IMPORTED ON)
    add_dependencies(${_name}_designer ${_name})
  endif()
endfunction()
