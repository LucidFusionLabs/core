# $Id: CMakeLists.txt 1336 2014-12-08 09:29:59Z justin $

set(LFL_HEADLESS 1)

lfl_project(nlp)
add_subdirectory(../lfapp ${CMAKE_CURRENT_BINARY_DIR}/lfapp_obj)
add_definitions(${LFAPP_DEF})
include_directories(../lflpub ${LFAPP_INCLUDE})

# nlp
set(LFL_NLP_SOURCE ../nlp/nlp.cpp)
set(LFL_NLP_SOURCE ${LFL_NLP_SOURCE} PARENT_SCOPE)
add_library(nlp ${LFL_NLP_SOURCE})

# trainer
add_executable(nlp-trainer trainer.cpp ${SDL_MAIN})
set_target_properties(nlp-trainer PROPERTIES OUTPUT_NAME trainer)
target_link_libraries(nlp-trainer ${LFAPP_LIB} nlp)
lfapp_post_build_start(nlp-trainer trainer NLPTrainer)

