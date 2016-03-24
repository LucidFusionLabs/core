# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})
lfl_add_library(app_null_css STATIC null_css.cpp)

if(LFL_LIBCSS)
  lfl_add_library(app_libcss_css STATIC libcss_css.cpp)
  target_include_directories(app_libcss_css PUBLIC ${LIBCSS_INCLUDE} ${LFL_CORE_BINARY_DIR}/app)
  target_link_libraries(app_libcss_css PUBLIC ${LIBCSS_LIB})
  set(CMAKE_AUTOMOC)
endif()
