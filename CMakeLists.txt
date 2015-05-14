# $Id: CMakeLists.txt 1318 2014-10-23 06:13:52Z justin $

if(LFL_CUDA)
    add_subdirectory(lfcuda)
endif(LFL_CUDA)

# lfapp unit tests
add_subdirectory(lfapp_tests)

# crawler
add_subdirectory(crawler)

# nlp
add_subdirectory(nlp)

# speech
add_subdirectory(speech)
