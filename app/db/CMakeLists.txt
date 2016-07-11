# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $
  
lfl_add_target(app_null_sqlite STATIC_LIBRARY SOURCES null_sqlite.cpp)

if(LFL_SQLITE)
  lfl_add_target(app_db_sqlite STATIC_LIBRARY SOURCES db_sqlite.cpp
                 INCLUDE_DIRECTORIES ${SQLITE_INCLUDE} LINK_LIBRARIES ${SQLITE_LIB})
endif()
