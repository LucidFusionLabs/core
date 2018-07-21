# $Id$

lfl_add_target(app_null_css STATIC_LIBRARY SOURCES null_css.cpp)
lfl_add_target(app_browser STATIC_LIBRARY SOURCES browser.cpp)

if(LFL_LIBCSS)
  lfl_add_target(app_libcss_css STATIC_LIBRARY SOURCES libcss_css.cpp
                 INCLUDE_DIRECTORIES ${LIBCSS_INCLUDE} ${LFL_CORE_BINARY_DIR}/app 
                 LINK_LIBRARIES ${LIBCSS_LIB})
  set(CMAKE_AUTOMOC)
endif()
