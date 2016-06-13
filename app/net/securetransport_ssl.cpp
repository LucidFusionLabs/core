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

#include <Security/SecureTransport.h>

namespace LFL {
DEFINE_string(ssl_certfile, "", "SSL server certificate file");
DEFINE_string(ssl_keyfile,  "", "SSL server key file");

static OSStatus SecureTransportRead(SSLConnectionRef conn, void *data, size_t *length) {
  OSStatus ret = 0;
  auto ssl_socket = static_cast<const SSLSocket*>(conn);
  StringPiece buf(static_cast<const char*>(data), *length);
  while (buf.len) {
#ifdef LFL_WINDOWS
    int len = recv(ssl_socket->socket, const_cast<char*>(buf.buf), buf.len, 0);
#else
    int len = read(ssl_socket->socket, const_cast<char*>(buf.buf), buf.len);
#endif
    if (len > 0) buf.pop_front(len);
    else if (!len) { ret = errSSLClosedGraceful; break; }
    else { ret = SystemNetwork::EWouldBlock() ? errSSLWouldBlock : errSSLClosedAbort; break; }
  }
  *length -= buf.len;
  return ret;
}

static OSStatus SecureTransportWrite(SSLConnectionRef conn, const void *data, size_t *length) {
  OSStatus ret = 0;
  auto ssl_socket = static_cast<const SSLSocket*>(conn);
  StringPiece buf(static_cast<const char*>(data), *length);
  while (buf.len) {
    int len = send(ssl_socket->socket, buf.buf, buf.len, 0);
    if (len > 0) buf.pop_front(len);
    else { ret = SystemNetwork::EWouldBlock() ? errSSLWouldBlock : errSSLClosedAbort; break; }
  }
  *length -= buf.len;
  return ret;
}

static int SecureTransportHandshake(SSLSocket *ssl_socket) {
  OSStatus status = SSLHandshake(FromVoid<SSLContextRef>(ssl_socket->ssl));
  if (!status) {
    ssl_socket->ready = true;
    if (ssl_socket->buf.empty()) return 0;
    int ret = ssl_socket->Write(ssl_socket->buf);
    ssl_socket->buf.clear();
    return ret; 
  } else return status == errSSLWouldBlock ? 0 : -1;
}

static Socket SecureTransportClose(SSLSocket *ssl_socket) {
  SystemNetwork::CloseSocket(ssl_socket->socket);
  return (ssl_socket->socket = InvalidSocket);
}

SSLSocket::~SSLSocket() { if (auto s = FromVoid<SSLContextRef>(ssl)) { /*SSLClose(s);*/ CFRelease(s); } }
string SSLSocket::ErrorString() const { return StrCat(last_error); }

ptrdiff_t SSLSocket::Write(const StringPiece &b) {
  if (!ready && SecureTransportHandshake(this) < 0) return -1;
  if (!ready) { buf.append(b.buf, b.len); return b.len; }

  size_t ret = 0;
  OSStatus status = SSLWrite(FromVoid<SSLContextRef>(ssl), b.buf, b.len, &ret);
  if (status) return status == errSSLWouldBlock ? 0 : -1;
  return ret;
}

ptrdiff_t SSLSocket::Read(char *buf, int readlen) { 
  size_t ret = 0;
  OSStatus status = SSLRead(FromVoid<SSLContextRef>(ssl), buf, readlen, &ret);
  if (!ready && SecureTransportHandshake(this) < 0) return -1;
  if (status) return status == errSSLWouldBlock ? 0 : -1;
  return ret;
}

Socket SSLSocket::Connect(CTXPtr sslctx, const string &hostport) {
  int port;
  IPV4::Addr addr;
  if (!HTTP::ResolveHost(hostport.c_str(), 0, &addr, &port, 0, 443))
    return ERRORv(InvalidSocket, "resolve ", hostport, " failed");
  return Connect(sslctx, addr, port);
};

Socket SSLSocket::Connect(CTXPtr sslctx, IPV4::Addr addr, int port) {
  int connected = 0;
  if ((socket = SystemNetwork::OpenSocket(Protocol::TCP)) == InvalidSocket) return InvalidSocket;
  if (SystemNetwork::SetSocketBlocking(socket, 0))                  return SecureTransportClose(this);
  if (SystemNetwork::Connect(socket, addr, port, &connected) == -1) return SecureTransportClose(this);
  CHECK_EQ(0, connected);
  ssl = SSLCreateContext(kCFAllocatorDefault, kSSLClientSide, kSSLStreamType);
  auto s = FromVoid<SSLContextRef>(ssl);
  SSLSetIOFuncs(s, SecureTransportRead, SecureTransportWrite);
  SSLSetConnection(s, this);
  return socket;
}

Socket SSLSocket::Listen(int port, bool reuse) {
  return (socket = SystemNetwork::Listen(Protocol::TCP, 0, port, 32, false));
}

Socket SSLSocket::Accept(SSLSocket *out) {
  if ((out->socket = SystemNetwork::Accept(socket, 0, 0)) == InvalidSocket) return InvalidSocket;
  out->ssl = SSLCreateContext(kCFAllocatorDefault, kSSLServerSide, kSSLStreamType);
  auto s = FromVoid<SSLContextRef>(out->ssl);
  SSLSetIOFuncs(s, SecureTransportRead, SecureTransportWrite);
  SSLSetConnection(s, this);
  // SSLSetCertificate(s, nullptr);
  if (SecureTransportHandshake(out) < 0) return SecureTransportClose(out);
  return out->socket;
}

SSLSocket::CTXPtr SSLSocket::Init() { return Void(0xdeadbeef); }
void SSLSocket::Free() {}

}; // namespace LFL
