# $Id$

lfl_add_target(app_null_ssl STATIC_LIBRARY SOURCES null_ssl.cpp)

if(LFL_NET_SMTP)
  lfl_add_target(app_net_smtp STATIC_LIBRARY SOURCES smtp.cpp)
endif()

if(LFL_NET_SSH)
  lfl_add_target(app_net_ssh STATIC_LIBRARY SOURCES ssh.cpp)
endif()

if(LFL_NET_RFB)
  lfl_add_target(app_net_rfb STATIC_LIBRARY SOURCES rfb.cpp)
endif()

if(LFL_OPENSSL)
  lfl_add_target(app_openssl_ssl STATIC_LIBRARY SOURCES openssl_ssl.cpp
                 INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE} LINK_LIBRARIES ${OPENSSL_LIB})
endif()

if(LFL_APPLE OR LFL_IOS)
  lfl_add_target(app_securetransport_ssl STATIC_LIBRARY SOURCES securetransport_ssl.cpp
                 LINK_LIBRARIES "-framework Security")
endif()

if(LFL_FFMPEG)
  lfl_add_target(app_ffmpeg_stream STATIC_LIBRARY SOURCES ffmpeg_stream.cpp
                 INCLUDE_DIRECTORIES ${FFMPEG_INCLUDE} LINK_LIBRARIES ${FFMPEG_LIB})
endif()
