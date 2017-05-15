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
endmacro(lfl_project)

function(lfl_set_xcode_properties _name)
  if(LFL_XCODE)
    if(LFL_DEBUG)
      set(XCODE_OPT_FLAG "0")
    else()
      set(XCODE_OPT_FLAG "2")
    endif()
    set_target_properties(${_name} PROPERTIES
                          XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS YES
                          XCODE_ATTRIBUTE_GCC_OPTIMIZATION_LEVEL ${XCODE_OPT_FLAG})
    if(LFL_IOS)
      set_target_properties(${_name} PROPERTIES
                            XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")
    endif()
  endif()
endfunction()

function(lfl_add_target _name)
  set(options EXECUTABLE SHARED_LIBRARY STATIC_LIBRARY WIN32)
  set(one_value_args)
  set(multi_value_args DEPENDENCIES COMPILE_DEFINITIONS COMPILE_OPTIONS INCLUDE_DIRECTORIES
      SOURCES LINK_LIBRARIES LIB_FILES ASSET_DIRS ASSET_FILES)
  cmake_parse_arguments("" "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT _WIN32)
    set(_WIN32)
  endif()
  set(${_name}_LIB_FILES ${_LIB_FILES} PARENT_SCOPE)
  set(${_name}_ASSET_DIRS ${_ASSET_DIRS} PARENT_SCOPE)
  set(${_name}_ASSET_FILES ${_ASSET_FILES} PARENT_SCOPE)

  if(_EXECUTABLE)
    if(LFL_IOS OR LFL_OSX)
      if(LFL_OSX)
        set(BUNDLE_PREFIX Resources/) 
      elseif(LFL_IOS AND LFL_XCODE)
        set(BUNDLE_PREFIX ${_name}.app/)
      endif()
      file(GLOB ASSET_FILES ${_ASSET_FILES})
      set_source_files_properties(${ASSET_FILES} PROPERTIES HEADER_FILE_ONLY ON
                                  MACOSX_PACKAGE_LOCATION ${BUNDLE_PREFIX}assets)
      foreach(_dir ${_ASSET_DIRS})
        file(GLOB ASSET_DIR_FILES ${_dir}/*)
        get_filename_component(_dirname ${_dir} NAME)
        set_source_files_properties(${ASSET_DIR_FILES} PROPERTIES HEADER_FILE_ONLY ON
                                    MACOSX_PACKAGE_LOCATION ${BUNDLE_PREFIX}${_dirname})
        set(ASSET_FILES ${ASSET_FILES} ${ASSET_DIR_FILES})
      endforeach()
    else()
      set(ASSET_FILES)
    endif()

    if(_WIN32)
      add_executable(${_name} WIN32 ${_SOURCES} ${ASSET_FILES})
    else()
      add_executable(${_name} ${_SOURCES} ${ASSET_FILES})
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
  lfl_set_xcode_properties(${_name})

  if(_EXECUTABLE AND LFL_ADD_BITCODE_TARGETS)
    add_library(${_name}_bitcode STATIC ${_SOURCES})
    target_compile_options(${_name}_bitcode PUBLIC ${_COMPILE_OPTIONS} -emit-llvm -fno-use-cxa-atexit)
    target_compile_definitions(${_name}_bitcode PUBLIC ${LFL_APP_DEF} ${_COMPILE_DEFINITIONS})
    target_include_directories(${_name}_bitcode PUBLIC ${LFL_APP_INCLUDE} ${_INCLUDE_DIRECTORIES})
    target_use_precompiled_header(${_name}_bitcode core/app/app.h FORCEINCLUDE)
    set_property(TARGET ${_name}_bitcode PROPERTY NO_SYSTEM_FROM_IMPORTED ON)
    add_dependencies(${_name}_bitcode ${_name})

    add_executable(${_name}_designer ${_WIN32} ${LFL_SOURCE_DIR}/core/ide/llvm_jit.cpp)
    target_compile_options(${_name}_designer PUBLIC ${_COMPILE_OPTIONS})
    target_compile_definitions(${_name}_designer PUBLIC ${LFL_APP_DEF} ${_COMPILE_DEFINITIONS})
    target_include_directories(${_name}_designer PUBLIC ${LFL_APP_INCLUDE} ${LIBCLANG_INCLUDE})
    target_link_libraries(${_name}_designer PUBLIC ${_LINK_LIBRARIES} app_libarchive_archive app_clang_tu)
    set_property(TARGET ${_name}_designer PROPERTY NO_SYSTEM_FROM_IMPORTED ON)
    add_dependencies(${_name}_designer ${_name})
  endif()
endfunction()
