# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_SOURCE_DIR})
lfl_add_library(app_null_audio STATIC null_audio.cpp)

if(LFL_OPENAL)
  lfl_add_library(app_openal_audio STATIC openal_audio.cpp)
endif()

if(LFL_PORTAUDIO)
  lfl_add_library(app_portaudio_audio STATIC portaudio_audio.cpp)
  target_include_directories(app_portaudio_audio PUBLIC ${PORTAUDIO_INCLUDE})
  target_link_libraries(app_portaudio_audio ${PORTAUDIO_LIB})
endif()

if(LFL_AUDIOQUEUE)
  lfl_add_library(app_audioqueue_audio STATIC audioqueue_audio.cpp)
  target_link_libraries(app_audioqueue_audio -framework AudioToolbox)
endif()

if(LFL_AUDIOUNIT)
  lfl_add_library(app_audiounit_audio STATIC audiounit_audio.cpp)
  target_link_libraries(app_audiounit_audio -framework AudioToolbox)
endif()
