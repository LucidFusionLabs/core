/*
 * $Id$
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

SSLSocket::~SSLSocket() { if (bio) BIO_free_all(FromVoid<BIO*>(bio)); }
string SSLSocket::ErrorString() const { return BlankNull(ERR_reason_error_string(ERR_get_error())); }
ptrdiff_t SSLSocket::Write(const StringPiece &b) { return BIO_write(FromVoid<BIO*>(bio), b.buf, b.len); }
ptrdiff_t SSLSocket::Read(char *buf, int readlen) {
  ptrdiff_t len = BIO_read(FromVoid<BIO*>(bio), buf, readlen);
  if (len <= 0) return SSL_get_error(FromVoid<SSL*>(ssl), len) == SSL_ERROR_WANT_READ ? 0 : -1;
  return len;
}

Socket SSLSocket::Connect(CTXPtr sslctx, const string &hostport) {
  bio = BIO_new_ssl_connect(FromVoid<SSL_CTX*>(sslctx));
  BIO *b = FromVoid<BIO*>(bio);
  BIO_set_conn_hostname(b, hostport.c_str());
  BIO_get_ssl(b, &ssl);
  BIO_set_nbio(b, 1);
  if (BIO_do_connect(b) < 0 && !BIO_should_retry(b)) return InvalidSocket;
  BIO_get_fd(b, &socket);
  return socket;
};

Socket SSLSocket::Connect(CTXPtr sslctx, IPV4::Addr addr, int port) {
  char addrbuf[sizeof(addr)], portbuf[sizeof(port)];
  memcpy(addrbuf, &addr, sizeof(addr));
  memcpy(portbuf, &port, sizeof(port));
  bio = BIO_new_ssl_connect(FromVoid<SSL_CTX*>(sslctx));
  BIO *b = FromVoid<BIO*>(bio);
  BIO_set_conn_ip(b, addrbuf);
  BIO_set_conn_int_port(b, portbuf);
  BIO_get_ssl(b, &ssl);
  BIO_set_nbio(b, 1);
  if (BIO_do_connect(b) < 0 && !BIO_should_retry(b)) return InvalidSocket;
  BIO_get_fd(b, &socket);
  return socket;
}

Socket SSLSocket::Listen(CTXPtr sslctx, int port, bool reuse) {
  bio = BIO_new_accept(const_cast<char*>(StrCat(port).c_str()));
  BIO *b = FromVoid<BIO*>(bio);
  BIO_ctrl(b, BIO_C_SET_ACCEPT, 1, Void("a"));
  if (reuse) BIO_set_bind_mode(b, BIO_BIND_REUSEADDR);
  if (BIO_do_accept(b) <= 0) return InvalidSocket;
  BIO_set_accept_bios(b, BIO_new_ssl(FromVoid<SSL_CTX*>(sslctx), 0));
  BIO_get_fd(b, &socket);
  return socket;
}

Socket SSLSocket::Accept(SSLSocket *out) {
  BIO *b = FromVoid<BIO*>(bio);
  if (BIO_do_accept(b) <= 0) return InvalidSocket;
  out->bio = BIO_pop(b);

  b = FromVoid<BIO*>(out->bio);
  BIO_set_nbio(b, 1);
  BIO_get_ssl(b, &out->ssl);
  BIO_get_fd(b, &out->socket);
  return out->socket;
}

SSLSocket::CTXPtr SSLSocket::Init() {
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
