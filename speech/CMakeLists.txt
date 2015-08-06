# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

set(LFL_EASTL 1)
set(LFL_LIBARCHIVE 1)
set(LFL_HEADLESS 1)

lfl_project(speech)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(${LFAPP_INCLUDE})

# speech
set(LFL_RECOGNITION_SOURCE ${LFL_CORE}/speech/speech.cpp ${LFL_CORE}/speech/recognition.cpp ${LFL_CORE}/speech/voice.cpp)
set(LFL_RECOGNITION_SOURCE ${LFL_RECOGNITION_SOURCE} PARENT_SCOPE)
add_library(recognition STATIC ${LFL_RECOGNITION_SOURCE})

# trainer
add_executable(speech-trainer trainer.cpp)
target_link_libraries(speech-trainer ${LFAPP_LIB} recognition)
lfapp_post_build_start(speech-trainer speec-trainer SpeechTrainer)

# mfcp
add_executable(mfcp mfcp.cpp)
target_link_libraries(mfcp ${LFAPP_LIB})

# vpcat
add_executable(vpcat vpcat.cpp)
target_link_libraries(vpcat ${LFAPP_LIB} recognition)

