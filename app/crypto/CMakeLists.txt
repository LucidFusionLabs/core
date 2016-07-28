# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

lfl_add_target(app_null_crypto STATIC_LIBRARY SOURCES null_crypto.cpp)
lfl_add_target(app_null_dh STATIC_LIBRARY SOURCES null_dh.cpp)
lfl_add_target(app_null_pk STATIC_LIBRARY SOURCES null_pk.cpp)

if(LFL_APPLE)
  lfl_add_target(app_commoncrypto_crypto STATIC_LIBRARY SOURCES commoncrypto_crypto.cpp)
endif()

if(LFL_OPENSSL)
  lfl_add_target(app_openssl_crypto STATIC_LIBRARY SOURCES openssl_crypto.cpp
                 INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE} LINK_LIBRARIES ${OPENSSL_LIB})

  lfl_add_target(app_openssl_dh STATIC_LIBRARY SOURCES openssl_dh.cpp
                 INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE} LINK_LIBRARIES ${OPENSSL_LIB})

  lfl_add_target(app_openssl_pk STATIC_LIBRARY SOURCES openssl_pk.cpp
                 INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE} LINK_LIBRARIES ${OPENSSL_LIB})
endif()

if(LFL_REF10EXTRACT)
  lfl_add_target(app_ref10_25519 STATIC_LIBRARY SOURCES ref10_25519.cpp
                 INCLUDE_DIRECTORIES ${REF10EXTRACT_INCLUDE} LINK_LIBRARIES ${REF10EXTRACT_LIB})
endif()
