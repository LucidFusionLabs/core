# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

lfl_project(speech)
add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})

# recognition
add_library(recognition STATIC speech.cpp recognition.cpp voice.cpp)

# trainer
add_executable(speech-trainer trainer.cpp)
target_link_libraries(speech-trainer ${LFAPP_LIB} recognition)
lfl_post_build_start(speech-trainer speec-trainer SpeechTrainer)

# mfcp
add_executable(mfcp mfcp.cpp)
target_link_libraries(mfcp ${LFAPP_LIB})

# vpcat
add_executable(vpcat vpcat.cpp)
target_link_libraries(vpcat ${LFAPP_LIB} recognition)

