# $Id: CMakeLists.txt 1327 2014-11-03 23:26:43Z justin $

set(LFL_APP_CFLAGS)
set(LFL_APP_DEF -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS)
set(LFL_APP_INCLUDE ${LFL_SOURCE_DIR} ${LFL_CORE_BINARY_DIR})
set(LFL_APP_LIB lfl_app app_browser app_layers app_font ${LFL_APP_REGEX})

macro(add_app_def _def)
  list_add_def(LFL_APP_DEF "${_def}")
endmacro()

macro(add_app_lib _def _inc)
  list_add_lib(LFL_APP_DEF LFL_APP_INCLUDE "${_def}" "${_inc}")
endmacro()

add_app_def(LFL32)
add_app_def(LFL64)
add_app_def(LFL_IPC)
add_app_def(LFL_DEBUG)
add_app_def(LFL_APPLE)
add_app_def(LFL_LINUX)
add_app_def(LFL_WINDOWS)
add_app_def(LFL_MOBILE)
add_app_def(LFL_ANDROID)
add_app_def(LFL_IOS)
add_app_def(LFL_IOS_SIM)
add_app_def(LFL_EMSCRIPTEN)
add_app_lib(LFL_FLATBUFFERS "${FLATBUFFERS_INCLUDE_DIR}" "")

if(LFL_QT)
  set(LFL_APP_DEF ${LFL_APP_DEF} ${QT_DEF})
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(LFL_APP_CFLAGS ${LFL_APP_CFLAGS} -fPIC)
  endif()
endif()

if(LFL_FLATBUFFERS)
  FLATBUFFERS_GENERATE_C_HEADERS(FLATBUFFERS_SRC ipc.fbs)
endif()

add_subdirectory(framework)
add_subdirectory(types)
add_subdirectory(math)
add_subdirectory(gl/device)
add_subdirectory(font)
add_subdirectory(audio)
add_subdirectory(loader)
add_subdirectory(net)
add_subdirectory(crypto)
add_subdirectory(bindings)
add_subdirectory(camera)
add_subdirectory(cuda)
add_subdirectory(db)

add_library(lfl_app ${LFL_APP_TYPE} gl/view.cpp gl/terminal.cpp gl/editor.cpp gl/toolkit.cpp
            font.cpp flow.cpp app.cpp types/types.cpp types/string.cpp types/time.cpp "file.cpp"
            assets.cpp loader.cpp audio.cpp gl.cpp input.cpp math.cpp ipc.cpp
            network.cpp net/http.cpp net/resolver.cpp
            camera.cpp scene.cpp shell.cpp crypto.cpp ${FLATBUFFERS_SRC_OUTPUTS})

set(_include ${LFL_APP_INCLUDE})
if(LFL_GLOG)
  set(_include ${_include} ${GLOG_INCLUDE})
  target_link_libraries(lfl_app ${GLOG_LIB})
  target_compile_definitions(lfl_app PUBLIC -DLFL_GLOG)
endif()
if(LFL_FLATBUFFERS)
  target_link_libraries(lfl_app ${FLATBUFFERS_LIB})
  add_dependencies(lfl_app flatbuffers ${FLATBUFFERS_SRC})
  if(NOT LFL_MOBILE AND NOT LFL_XCODE)
    add_dependencies(lfl_app flatc)
  endif()
endif()

target_compile_options(lfl_app PUBLIC ${LFL_APP_CFLAGS})
target_compile_definitions(lfl_app PUBLIC ${LFL_APP_DEF})
target_include_directories(lfl_app PUBLIC ${_include})

add_precompiled_header(lfl_app core/app/app.h c++-header COMPILE_OPTIONS ${LFL_APP_CFLAGS}
                       COMPILE_DEFINITIONS ${LFL_APP_DEF} INCLUDE_DIRECTORIES ${_include})
target_use_precompiled_header(lfl_app core/app/app.h FORCEINCLUDE)

lfl_add_target(app_layers STATIC_LIBRARY SOURCES layers.cpp)

set(LFL_APP_CFLAGS ${LFL_APP_CFLAGS} PARENT_SCOPE)
set(LFL_APP_DEF ${LFL_APP_DEF} PARENT_SCOPE)
set(LFL_APP_INCLUDE ${LFL_APP_INCLUDE} PARENT_SCOPE)
set(LFL_APP_LIB ${LFL_APP_LIB} PARENT_SCOPE)
set(LFL_APP_ASSET_FILES ${LFL_SOURCE_DIR}/core/app/shaders/default.*
    ${LFL_SOURCE_DIR}/core/app/assets/MenuAtlas*
    ${LFL_SOURCE_DIR}/core/app/assets/MobileAtlas* PARENT_SCOPE)

add_subdirectory(tests)
