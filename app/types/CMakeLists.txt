# $Id$

lfl_add_target(app_null_convert STATIC_LIBRARY SOURCES null_convert.cpp)
lfl_add_target(app_null_regex STATIC_LIBRARY SOURCES null_regex.cpp)
lfl_add_target(app_null_sregex STATIC_LIBRARY SOURCES null_sregex.cpp)

lfl_add_target(app_stdregex_regex STATIC_LIBRARY SOURCES stdregex_regex.cpp
               INCLUDE_DIRECTORIES ${REGEX_INCLUDE} LINK_LIBRARIES ${REGEX_LIB})

if(LFL_ICU)
  lfl_add_target(app_icu_regex STATIC_LIBRARY SOURCES icu_regex.cpp)
endif()

if(LFL_RE2)
  lfl_add_target(app_re2_regex STATIC_LIBRARY SOURCES re2_regex.cpp
                 INCLUDE_DIRECTORIES ${RE2_INCLUDE} LINK_LIBRARIES ${RE2_LIB})
endif()

if(LFL_SREGEX)
  lfl_add_target(app_libsregex_sregex STATIC_LIBRARY SOURCES libsregex_sregex.cpp
                 INCLUDE_DIRECTORIES ${SREGEX_INCLUDE} LINK_LIBRARIES ${SREGEX_LIB})
endif()

if(LFL_ICONV)
  lfl_add_target(app_iconv_convert STATIC_LIBRARY SOURCES iconv_convert.cpp)
endif()
