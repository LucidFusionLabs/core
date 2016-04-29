/*
 * $Id: crypto.cpp 1335 2014-12-02 04:13:46Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "openssl/bio.h"
#include "openssl/ssl.h"
#include "openssl/err.h"
#include "openssl/conf.h"
#ifndef LFL_ANDROID
#include "openssl/engine.h"
#endif

namespace LFL {
DEFINE_string(ssl_certfile, "", "SSL server certificate file");
DEFINE_string(ssl_keyfile,  "", "SSL server key file");

SSLSocket::~SSLSocket() { if (bio) BIO_free_all(bio); }
const char *SSLSocket::ErrorString() const { return ERR_reason_error_string(ERR_get_error()); }
Socket SSLSocket::GetSocket() const { Socket v=InvalidSocket; if (bio) BIO_get_fd(bio, &v); return v; }
ptrdiff_t SSLSocket::Write(const StringPiece &b) { return BIO_write(bio, b.buf, b.len); }
ptrdiff_t SSLSocket::Read(char *buf, int readlen) {
  ptrdiff_t len = BIO_read(bio, buf, readlen);
  if (len <= 0) return SSL_get_error(ssl, len) == SSL_ERROR_WANT_READ ? 0 : -1;
  return len;
}

Socket SSLSocket::Listen(int port, bool reuse) {
  bio = BIO_new_accept(const_cast<char*>(StrCat(port).c_str()));
  BIO_ctrl(bio, BIO_C_SET_ACCEPT, 1, Void("a"));
  if (reuse) BIO_set_bind_mode(bio, BIO_BIND_REUSEADDR);
  if (BIO_do_accept(bio) <= 0) return InvalidSocket;
  BIO_set_accept_bios(bio, BIO_new_ssl(app->net->ssl, 0));
  return GetSocket();
}

Socket SSLSocket::Connect(SSL_CTX *sslctx, const string &hostport) {
  bio = BIO_new_ssl_connect(sslctx);
  BIO_set_conn_hostname(bio, hostport.c_str());
  BIO_get_ssl(bio, &ssl);
  BIO_set_nbio(bio, 1);
  if (BIO_do_connect(bio) < 0 && !BIO_should_retry(bio)) return InvalidSocket;
  return GetSocket();
};

Socket SSLSocket::Connect(SSL_CTX *sslctx, IPV4::Addr addr, int port) {
  char addrbuf[sizeof(addr)], portbuf[sizeof(port)];
  memcpy(addrbuf, &addr, sizeof(addr));
  memcpy(portbuf, &port, sizeof(port));
  bio = BIO_new_ssl_connect(sslctx);
  BIO_set_conn_ip(bio, addrbuf);
  BIO_set_conn_int_port(bio, portbuf);
  BIO_get_ssl(bio, &ssl);
  BIO_set_nbio(bio, 1);
  if (BIO_do_connect(bio) < 0 && !BIO_should_retry(bio)) return InvalidSocket;
  return GetSocket();
}

Socket SSLSocket::Accept(SSLSocket *out) {
  if (BIO_do_accept(bio) <= 0) return InvalidSocket;
  out->bio = BIO_pop(bio);
  BIO_set_nbio(out->bio, 1);
  BIO_get_ssl(out->bio, &out->ssl);
  return out->GetSocket();
}

SSL_CTX *SSLSocket::Init() {
  SSL_CTX *ssl = nullptr;
  SSL_load_error_strings();
  SSL_library_init(); 

  if (bool client_only=0) ssl = SSL_CTX_new(SSLv23_client_method());
  else                    ssl = SSL_CTX_new(SSLv23_method());

  if (!ssl) FATAL("no SSL_CTX: ", ERR_reason_error_string(ERR_get_error()));
  SSL_CTX_set_verify(ssl, SSL_VERIFY_NONE, 0);

  if (FLAGS_ssl_certfile.size() && FLAGS_ssl_keyfile.size()) {
    if (!SSL_CTX_use_certificate_file(ssl, FLAGS_ssl_certfile.c_str(), SSL_FILETYPE_PEM)) return ERRORv(nullptr, "SSL_CTX_use_certificate_file ", ERR_reason_error_string(ERR_get_error()));
    if (!SSL_CTX_use_PrivateKey_file(ssl, FLAGS_ssl_keyfile.c_str(), SSL_FILETYPE_PEM)) return ERRORv(nullptr, "SSL_CTX_use_PrivateKey_file ",  ERR_reason_error_string(ERR_get_error()));
    if (!SSL_CTX_check_private_key(ssl)) return ERRORv(nullptr, "SSL_CTX_check_private_key ", ERR_reason_error_string(ERR_get_error()));
  }
  return ssl;
}

void SSLSocket::Free() {
  CONF_modules_free();
  ERR_remove_state(0);
#ifndef LFL_ANDROID
  ENGINE_cleanup();
#endif
  CONF_modules_unload(1);
  ERR_free_strings();
  EVP_cleanup();
  CRYPTO_cleanup_all_ex_data();
  sk_SSL_COMP_free(SSL_COMP_get_compression_methods());
}

}; // namespace LFL
