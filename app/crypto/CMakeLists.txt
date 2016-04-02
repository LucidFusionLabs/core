# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_crypto STATIC_LIBRARY SOURCES null_crypto.cpp)
lfl_add_target(app_null_dh STATIC_LIBRARY SOURCES null_dh.cpp)

if(LFL_APPLE)
  lfl_add_target(app_commoncrypto_crypto STATIC_LIBRARY SOURCES commoncrypto_crypto.cpp)
endif()

if(LFL_OPENSSL)
  lfl_add_target(app_openssl_crypto STATIC_LIBRARY SOURCES openssl_crypto.cpp
                 INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE} LINK_LIBRARIES ${OPENSSL_LIB})

  lfl_add_target(app_openssl_dh STATIC_LIBRARY SOURCES openssl_dh.cpp
                 INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE} LINK_LIBRARIES ${OPENSSL_LIB})
endif()
