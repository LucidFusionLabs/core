# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})
lfl_add_library(app_null_convert STATIC null_convert.cpp)
lfl_add_library(app_null_regex STATIC null_regex.cpp)
lfl_add_library(app_null_sregex STATIC null_sregex.cpp)

if(LFL_ICONV)
  lfl_add_library(app_iconv_convert STATIC iconv_convert.cpp)
endif()

if(LFL_REGEX)
  lfl_add_library(app_regexp_regex STATIC regexp_regex.cpp)
  target_include_directories(app_regexp_regex PUBLIC ${REGEX_INCLUDE})
  target_link_libraries(app_regexp_regex PUBLIC ${REGEX_LIB})
endif()

if(LFL_SREGEX)
  lfl_add_library(app_libsregex_sregex STATIC libsregex_sregex.cpp)
  target_include_directories(app_libsregex_sregex PUBLIC ${SREGEX_INCLUDE})
  target_link_libraries(app_libsregex_sregex PUBLIC ${SREGEX_LIB})
endif()
