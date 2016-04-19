# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_audio STATIC_LIBRARY SOURCES null_audio.cpp)

if(LFL_OPENAL)
  if(LFL_APPLE)
    set(libs "-framework OpenAL")
  elseif(LFL_LINUX)
    set(libs "-lopenal")
  endif()
  lfl_add_target(app_openal_audio STATIC_LIBRARY SOURCES openal_audio.cpp
                 LINK_LIBRARIES ${libs})
endif()

if(LFL_PORTAUDIO)
  lfl_add_target(app_portaudio_audio STATIC_LIBRARY SOURCES portaudio_audio.cpp
                 INCLUDE_DIRECTORIES ${PORTAUDIO_INCLUDE} 
                 LINK_LIBRARIES ${PORTAUDIO_LIB})
endif()

if(LFL_AUDIOQUEUE)
  lfl_add_target(app_audioqueue_audio STATIC_LIBRARY SOURCES audioqueue_audio.cpp
                 LINK_LIBRARIES "-framework AudioToolbox")
endif()

if(LFL_AUDIOUNIT)
  lfl_add_target(app_audiounit_audio STATIC_LIBRARY SOURCES audiounit_audio.cpp
                 LINK_LIBRARIES "-framework AudioToolbox")
endif()
