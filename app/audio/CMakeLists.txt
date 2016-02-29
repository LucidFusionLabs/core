# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

include_directories(${LFL_SOURCE_DIR})
add_library(app_null_audio STATIC null_audio.cpp)

if(LFL_PORTAUDIO)
  add_library(app_portaudio_audio STATIC portaudio_audio.cpp)
  target_include_directories(app_portaudio_audio PUBLIC ${PORTAUDIO_INCLUDE})
  target_link_libraries(app_portaudio_audio ${PORTAUDIO_LIB})
endif()

if(LFL_AUDIOQUEUE)
  add_library(app_audioqueue_audio STATIC audioqueue_audio.cpp)
  target_link_libraries(app_audioqueue_audio -framework AudioToolbox)
endif()

if(LFL_AUDIOUNIT)
  add_library(app_audiounit_audio STATIC audiounit_audio.cpp)
  target_link_libraries(app_audiounit_audio -framework AudioToolbox)
endif()
