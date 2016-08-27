# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $
  
lfl_add_target(app_null_sqlite STATIC_LIBRARY SOURCES null_sqlite.cpp)

if(LFL_SQLITE)
  lfl_add_target(app_sqlite_sqlite STATIC_LIBRARY SOURCES db_sqlite.cpp
                 INCLUDE_DIRECTORIES ${SQLITE_INCLUDE} LINK_LIBRARIES ${SQLITE_LIB})
endif()

if(LFL_SQLCIPHER)
  lfl_add_target(app_sqlcipher_sqlite STATIC_LIBRARY SOURCES db_sqlite.cpp
                 COMPILE_DEFINITIONS -DLFL_SQLCIPHER ${SQLCIPHER_DEF}
                 INCLUDE_DIRECTORIES ${SQLCIPHER_INCLUDE} LINK_LIBRARIES ${SQLCIPHER_LIB})
endif()
