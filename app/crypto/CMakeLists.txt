# $Id: CMakeLists.txt 1298 2011-12-11 18:46:06Z justin $

add_definitions(${LFL_APP_DEF})
include_directories(${LFL_APP_INCLUDE})
add_library(app_null_crypto STATIC null_crypto.cpp)
add_library(app_null_dh STATIC null_dh.cpp)

if(LFL_APPLE)
  add_library(app_commoncrypto_crypto STATIC commoncrypto_crypto.cpp)
endif()

if(LFL_OPENSSL)
  add_library(app_openssl_crypto STATIC openssl_crypto.cpp)
  target_include_directories(app_openssl_crypto PUBLIC ${OPENSSL_INCLUDE})
  target_link_libraries(app_openssl_crypto PUBLIC ${OPENSSL_LIB})

  add_library(app_openssl_dh STATIC openssl_dh.cpp)
  target_include_directories(app_openssl_dh PUBLIC ${OPENSSL_INCLUDE})
  target_link_libraries(app_openssl_dh PUBLIC ${OPENSSL_LIB})
endif()
