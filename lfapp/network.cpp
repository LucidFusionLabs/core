/*
 * $Id: network.cpp 1334 2014-11-28 09:14:21Z justin $
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

extern "C" {
#ifdef LFL_FFMPEG
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/avfft.h>
#include <libswscale/swscale.h>
#define AVCODEC_MAX_AUDIO_FRAME_SIZE 192000 
#endif

#ifdef LFL_PCAP
#include "pcap/pcap.h"
#endif
};

#include "lfapp/lfapp.h"
#include "lfapp/crypto.h"
#include "lfapp/resolver.h"

#ifdef LFL_OPENSSL
#include "openssl/bio.h"
#include "openssl/ssl.h"
#include "openssl/err.h"
#endif

#ifndef WIN32
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#ifndef LFL_ANDROID
#include <ifaddrs.h>
#endif
#endif

#ifdef LFL_GEOIP
#include "GeoIPCity.h"
#endif

namespace LFL {
DECLARE_string(nameserver);
DEFINE_bool(dns_dump,       0,  "Print DNS responses");
DEFINE_bool(network_debug,  0,  "Print send()/recv() bytes");
DEFINE_int (udp_idle_sec,   15, "Timeout UDP connections idle for seconds");
#ifdef LFL_OPENSSL
DEFINE_string(ssl_certfile, "", "SSL server certificate file");
DEFINE_string(ssl_keyfile,  "", "SSL server key file");
SSL_CTX *lfapp_ssl = 0;
#endif

IPV4EndpointPool::IPV4EndpointPool(const string &ip_csv) {
  IPV4::ParseCSV(ip_csv, &source_addrs);
  source_ports.resize(source_addrs.size());
  for (int i=0; i<source_ports.size(); i++) {
    source_ports[i].resize(bytes_per_ip, 0);
    source_ports[i][bytes_per_ip-1] = 1<<7; // mark port 65536 as used
  }
  if (!source_addrs.size()) ERROR("warning empty address pool");
}

bool IPV4EndpointPool::Available() const {
  if (!source_addrs.size()) return true;
  for (int i=0; i<source_addrs.size(); i++)
    if (BitString::LastClear(source_ports[i].data(), source_ports[i].size()) != -1) return true;
  return false;
}

void IPV4EndpointPool::Close(IPV4::Addr addr, int port) {
  if (!source_addrs.size()) return;
  int bit = port - 1024;
  for (int i=0; i<source_addrs.size(); i++) if (source_addrs[i] == addr) {
    if (!BitString::Get(source_ports[i].data(), bit))
      ERROR("IPV4EndpointPool: Close unopened endpoint: ", IPV4::Text(addr, port));

    BitString::Clear(&source_ports[i][0], bit);
    return;
  }
  ERROR("IPV4EndpointPool: Close unknown endpoint: ", IPV4::Text(addr, port));
}

void IPV4EndpointPool::Get(IPV4::Addr addr, int *port) {
  *port=0;
  if (!source_addrs.size()) return;
  for (int i=0; i<source_addrs.size(); i++) if (source_addrs[i] == addr) { GetPort(i, port); return; }
  ERROR("IPV4EndpointPool: address full: ", IPV4::Text(addr));
}

void IPV4EndpointPool::Get(IPV4::Addr *addr, int *port) {
  *addr=0; *port=0;
  if (!source_addrs.size()) return;
  for (int i=0, max_retries=10; i<max_retries; i++) {
    int ind = Rand<int>(0, source_addrs.size()-1);
    *addr = source_addrs[ind];
    if (GetPort(ind, port)) return;
  }
  ERROR("IPV4EndpointPool: full");
}

bool IPV4EndpointPool::GetPort(int ind, int *port) {
  int zero_bit = BitString::FirstClear(source_ports[ind].data(), source_ports[ind].size());
  if (zero_bit == -1) return false;
  *port = 1025 + zero_bit;
  BitString::Set(&source_ports[ind][0], zero_bit);
  return true;
}

void IPV4EndpointPoolFilter::Get(IPV4::Addr *addr, int *port) { 
  *addr=0; *port=0;
  for (int i = 0; i < wrap->source_addrs.size(); i++) {
    if (Contains(filter, wrap->source_addrs[i])) continue;
    *addr = wrap->source_addrs[i];
    return Get(wrap->source_addrs[i], port);
  }
}

void SystemNetwork::CloseSocket(Socket fd) {
#ifdef WIN32
  closesocket(fd);
#else
  close(fd);
#endif
}

Socket SystemNetwork::OpenSocket(int protocol) {
  if (protocol == Protocol::TCP) return socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  else if (protocol == Protocol::UDP) {
    Socket ret;
    if ((ret = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) return ret;
#ifdef _WIN32
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR,12)
#endif
    DWORD dwBytesReturned = 0;
    BOOL bNewBehavior = FALSE;
    DWORD status = WSAIoctl(ret, SIO_UDP_CONNRESET, &bNewBehavior, sizeof(bNewBehavior), 0, 0, &dwBytesReturned, 0, 0);
#endif
    return ret;
  }
  else return -1;
}

bool SystemNetwork::OpenSocketPair(Socket *fd, bool close_on_exec) {
#ifdef WIN32
#ifdef LFL_NETWORK_MONOLITHIC_FRAME
  Socket l = -1;
  int listen_port = 0, connect_port = 0;
  IPV4::Addr listen_addr = 0, connect_addr = 0;
  if ((l = SystemNetwork::Listen(Protocol::TCP, IPV4::Parse("127.0.0.1"), 0, 1, true)) == -1) return -1;
  if ((fd[1] = SystemNetwork::OpenSocket(Protocol::TCP)) == -1) { SystemNetwork::CloseSocket(l); return ERRORv(-1, "OSP.OpenSocket ", SystemNetwork::LastError()); }
  if (SystemNetwork::GetSockName(l, &listen_addr, &listen_port)) { SystemNetwork::CloseSocket(l); SystemNetwork::CloseSocket(fd[1]); return -1; }
  if (SystemNetwork::Connect(fd[1], listen_addr, listen_port, 0)) { SystemNetwork::CloseSocket(l); SystemNetwork::CloseSocket(fd[1]); return -1; }
  if ((fd[0] = SystemNetwork::Accept(l, &connect_addr, &connect_port)) == -1) { SystemNetwork::CloseSocket(l); SystemNetwork::CloseSocket(fd[1]); return -1; }
  SystemNetwork::CloseSocket(l);
  SetSocketBlocking(fd[0], 0);
  SetSocketBlocking(fd[1], 0);
#else // LFL_NETWORK_MONOLITHIC_FRAME
  SECURITY_ATTRIBUTES sa;
  memset(&sa, 0, sizeof(sa));
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = 1;
  HANDLE handle[2];
  CHECK(CreatePipe(&handle[0], &handle[1], &sa, 0));
  // XXX use WFMO with HANDLE* instead of select with SOCKET
#endif // LFL_NETWORK_MONOLITHIC_FRAME
#else // WIN32
  CHECK(!socketpair(PF_LOCAL, SOCK_STREAM, 0, fd));
  SetSocketBlocking(fd[0], 0);
  SetSocketBlocking(fd[1], 0);
#endif
  if (close_on_exec) {
    SetSocketCloseOnExec(fd[0], true);
    SetSocketCloseOnExec(fd[1], true);
  }
  return true;
}

int SystemNetwork::SetSocketBlocking(Socket fd, int blocking) {
#ifdef _WIN32
  u_long ioctlarg = !blocking ? 1 : 0;
  if (ioctlsocket(fd, FIONBIO, &ioctlarg) < 0) return -1;
#else
  if (fcntl(fd, F_SETFL, !blocking ? O_NONBLOCK : 0) == -1) return -1;
#endif
  return 0;
}

int SystemNetwork::SetSocketCloseOnExec(Socket fd, int close) {
#ifdef _WIN32
#else
  if (fcntl(fd, F_SETFD, close ? FD_CLOEXEC : 0) == -1) return -1;
#endif
  return 0;
}

int SystemNetwork::SetSocketBroadcastEnabled(Socket fd, int optval) {
  if (setsockopt(fd, SOL_SOCKET, SO_BROADCAST, (const char*)&optval, sizeof(optval)))
    return ERRORv(-1, "setsockopt: ", SystemNetwork::LastError());
  return 0;
}

int SystemNetwork::SetSocketBufferSize(Socket fd, bool send_or_recv, int optval) {
  if (setsockopt(fd, SOL_SOCKET, send_or_recv ? SO_SNDBUF : SO_RCVBUF, (const char *)&optval, sizeof(optval)))
    return ERRORv(-1, "setsockopt: ", SystemNetwork::LastError());
  return 0;
}

int SystemNetwork::GetSocketBufferSize(Socket fd, bool send_or_recv) {
  int res=0, resSize=sizeof(res);
  if (getsockopt(fd, SOL_SOCKET, send_or_recv ? SO_SNDBUF : SO_RCVBUF, (char*)&res, (socklen_t*)&resSize))
    return ERRORv(-1, "getsockopt: ", SystemNetwork::LastError());
  return res;
}

int SystemNetwork::Bind(int fd, IPV4::Addr addr, int port) {
  sockaddr_in sin; int optval = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval, sizeof(optval)))
    return ERRORv(-1, "setsockopt: ", SystemNetwork::LastError());

  memset(&sin, 0, sizeof(sockaddr_in));
  sin.sin_family = PF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = addr ? addr : INADDR_ANY;

  if (FLAGS_network_debug) INFO("bind(", fd, ", ", IPV4::Text(addr, port), ")");
  if (SystemBind(fd, (const sockaddr *)&sin, (socklen_t)sizeof(sockaddr_in)) == -1)
    return ERRORv(-1, "bind: ", SystemNetwork::LastError());

  return 0;
}

Socket SystemNetwork::Accept(Socket listener, IPV4::Addr *addr, int *port) {
  struct sockaddr_in sin;
  int sinSize = sizeof(sin);
  Socket socket = ::accept(listener, (struct sockaddr *)&sin, (socklen_t*)&sinSize);
  if (socket == -1 && !SystemNetwork::EWouldBlock()) return ERRORv(-1, "accept: ", SystemNetwork::LastError());
  if (addr) *addr = sin.sin_addr.s_addr;
  if (port) *port = ntohs(sin.sin_port);
  return socket;
}

Socket SystemNetwork::Listen(int protocol, IPV4::Addr addr, int port, int backlog, bool blocking) {
  Socket fd;
  if ((fd = OpenSocket(protocol)) < 0) 
    return ERRORv(-1, "network_socket_open: ", SystemNetwork::LastError());

  if (Bind(fd, addr, port) == -1) { CloseSocket(fd); return -1; }

  if (protocol == Protocol::TCP) {
    if (::listen(fd, backlog) == -1)
    { ERROR("listen: ", SystemNetwork::LastError()); CloseSocket(fd); return -1; }
  }

  if (!blocking && SetSocketBlocking(fd, 0))
  { ERROR("Network::socket_blocking: ", SystemNetwork::LastError()); CloseSocket(fd); return -1; }

  INFO("listen(port=", port, ", protocol=", (protocol == Protocol::TCP) ? "TCP" : "UDP", ")");
  return fd;
}

int SystemNetwork::Connect(Socket fd, IPV4::Addr addr, int port, int *connected) {
  struct sockaddr_in sin;
  memset(&sin, 0, sizeof(struct sockaddr_in));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = addr;

  if (FLAGS_network_debug) INFO("connect(", fd, ", ", IPV4::Text(addr, port), ")");
  int ret = ::connect(fd, (struct sockaddr *)&sin, sizeof(struct sockaddr_in));
  if (ret == -1 && !SystemNetwork::EWouldBlock())
    return ERRORv(-1, "connect(", IPV4::Text(addr, port), "): ", SystemNetwork::LastError());

  if (connected) *connected = !ret;
  return 0;
}

int SystemNetwork::SendTo(Socket fd, IPV4::Addr addr, int port, const char *buf, int len) {
  sockaddr_in sin; int sinSize=sizeof(sin);
  sin.sin_family = PF_INET;
  sin.sin_addr.s_addr = addr;
  sin.sin_port = htons(port);
  return ::sendto(fd, buf, len, 0, (struct sockaddr*)&sin, sinSize);
}

int SystemNetwork::GetPeerName(Socket fd, IPV4::Addr *addr_out, int *port_out) {
  struct sockaddr_in sin; int sinSize=sizeof(sin);
  if (::getpeername(fd, (struct sockaddr *)&sin, (socklen_t*)&sinSize) < 0)
    return ERRORv(-1, "getpeername: ", strerror(errno));
  *addr_out = sin.sin_addr.s_addr;
  *port_out = ntohs(sin.sin_port);
  return 0;
}

int SystemNetwork::GetSockName(Socket fd, IPV4::Addr *addr_out, int *port_out) {
  struct sockaddr_in sin; int sinSize=sizeof(sin);
  if (::getsockname(fd, (struct sockaddr *)&sin, (socklen_t*)&sinSize) < 0)
    return ERRORv(-1, "getsockname: ", strerror(errno));
  *addr_out = sin.sin_addr.s_addr;
  *port_out = ntohs(sin.sin_port);
  return 0;
}

string SystemNetwork::GetHostByAddr(IPV4::Addr addr) {
#if defined(_WIN32) || defined(LFL_ANDROID)
  struct hostent *h = ::gethostbyaddr((const char *)&addr, sizeof(addr), PF_INET);
#else
  struct hostent *h = ::gethostbyaddr((const void *)&addr, sizeof(addr), PF_INET);
#endif
  return h ? h->h_name : "";
}

IPV4::Addr SystemNetwork::GetHostByName(const string &host) {
  in_addr a;
  if ((a.s_addr = IPV4::Parse(host)) != INADDR_NONE) return (int)a.s_addr;

  hostent *h = gethostbyname(host.c_str());
  if (h && h->h_length == 4) return *(int *)h->h_addr_list[0];

  ERROR("SystemNetwork::GetHostByName ", host);
  return -1;
}

int SystemNetwork::IOVLen(const iovec *iov, int len) {
  int ret = 0;
  if (iov) for (int i=0; i<len; i++) ret += iov[i].iov_len;
  return ret;
}

bool SystemNetwork::EWouldBlock() {
#ifdef _WIN32
  return WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == WSAEINPROGRESS;
#else
  return errno == EAGAIN || errno == EINPROGRESS;
#endif
};

string SystemNetwork::LastError() {
#ifdef _WIN32
  return StrCat(WSAGetLastError());
#else
  return strerror(errno);
#endif
}

int SelectSocketSet::Select(int wait_time) {
  int maxfd=-1, rc=0, wc=0, xc=0;
  struct timeval tv = Time2timeval(Time(wait_time));
  FD_ZERO(&rfds); FD_ZERO(&wfds); FD_ZERO(&xfds);
  for (auto &s : socket) {
    bool added = 0;
    if (s.second.first & READABLE)  { rc++; FD_SET(s.first, &rfds); added = 1; }
    if (s.second.first & WRITABLE)  { wc++; FD_SET(s.first, &wfds); added = 1; }
    if (s.second.first & EXCEPTION) { xc++; FD_SET(s.first, &xfds); added = 1; }
    if (added && s.first > maxfd) maxfd = s.first;
  }
  if (!rc && !wc && !xc) { MSleep(wait_time); return 0; }
  if ((select(maxfd+1, rc?&rfds:0, wc?&wfds:0, xc?&xfds:0, wait_time >= 0 ? &tv : 0)) == -1) return ERRORv(-1, "select: ", SystemNetwork::LastError());
  return 0;
}

SocketWakeupThread::~SocketWakeupThread() { SystemNetwork::CloseSocket(pipe[0]); SystemNetwork::CloseSocket(pipe[1]); }
void SocketWakeupThread::Start() {
  CHECK(SystemNetwork::OpenSocketPair(pipe));
  sockets.Add(pipe[0], SocketSet::READABLE, 0);
  thread.Start();
}
void SocketWakeupThread::Wakeup() { char c=0; if (pipe[1] >= 0) CHECK_EQ(send(pipe[1], &c, 1, 0), 1); }
void SocketWakeupThread::ThreadProc() {
  while (app->run) {
    if (frame_mutex) { ScopedMutex sm(*frame_mutex); }
    if (app->run) {
      SelectSocketSet my_sockets;
      { ScopedMutex sm(sockets_mutex); my_sockets = sockets; }
      my_sockets.Select(-1);
      if (my_sockets.GetReadable(pipe[0])) { char buf[128]; recv(pipe[0], buf, sizeof(buf), 0); }
      if (app->run) {
        if (!wakeup_each) app->scheduler.Wakeup(0);
        else for (auto &s : my_sockets.socket) if (my_sockets.GetReadable(s.first)) app->scheduler.Wakeup(s.second.second);
      }
    }
    if (wait_mutex) { ScopedMutex sm(*wait_mutex); }
  }
}

/* Connection */

int Connection::Read() {
  int readlen = rb.Remaining(), len = 0;
  if (readlen <= 0) return ERRORv(-1, Name(), ": read queue full, rl=", rb.size());

  if (ssl) {
#ifdef LFL_OPENSSL
    if ((len = BIO_read(bio, rb.end(), readlen)) <= 0) {
      if (SSL_get_error(ssl, len) != SSL_ERROR_WANT_READ) {
        const char *err_string = ERR_reason_error_string(ERR_get_error());
        return ERRORv(-1, Name(), ": BIO_read: ", err_string ? err_string : "read() zero");
      }
      return 0;
    }
#endif

#ifndef WIN32
  } else if (control_messages) {
    struct iovec iov;
    memzero(iov);
    iov.iov_base = rb.end();
    iov.iov_len = readlen;

    char control[CMSG_SPACE(sizeof (int))];
    struct msghdr msg;
    memzero(msg);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    if ((len = recvmsg(socket, &msg, 0)) <= 0) {
      if      (!len)                                     return ERRORv(-1, Name(), ": read() zero");
      else if (len < 0 && !SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": read(): ", SystemNetwork::LastError());
      return 0;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg && cmsg->cmsg_len == CMSG_LEN(sizeof(int)) && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
      Socket received_socket;
      memcpy(&received_socket, CMSG_DATA(cmsg), sizeof(int));
      transferred_socket.push_back({ received_socket, rb.size() });
    }
#endif

  } else { // XXX loop until read -1 with EAGAIN
#ifdef WIN32
    if ((len = recv(socket, rb.end(), readlen, 0)) <= 0) {
#else
    if ((len = read(socket, rb.end(), readlen)) <= 0) {
#endif
      if      (!len)                                     return ERRORv(-1, Name(), ": read() zero");
      else if (len < 0 && !SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": read(): ", SystemNetwork::LastError());
      return 0;
    }
  }

  rt = Now();
  rb.Added(len);
  rb.EnsureZeroTerminated();
  if (FLAGS_network_debug) INFO("read(", socket, ", ", len, ", '", rb.end()-len, "')");
  return len;
}

int Connection::Reads() {
  int len = 0, l;
  while ((l = Read()) > 0) len += l;
  return l < 0 ? l : len;
}

int Connection::ReadPacket() {
  int ret = Read();
  if (ret > 0) packets.push_back({ rb.size()-ret, ret });
  return ret;
}

int Connection::ReadPackets() {
  int ret=1;
  while (ret > 0) {
    ret = ReadPacket();
    if (ret < 0 && !SystemNetwork::EWouldBlock()) return ret;
  }
  return 0;
}

int Connection::Add(const char *buf, int len) {
  int readlen = rb.Remaining();
  if (readlen < len) return ERRORv(-1, Name(), ": read packet queue full");
  rt = Now();
  rb.Add(buf, len);
  rb.EnsureZeroTerminated();
  if (FLAGS_network_debug) INFO("add(", socket, ", ", len, ", '", rb.end()-len, "')");
  return len;
}

int Connection::AddPacket(const char *buf, int len) {
  int ret = Add(buf, len);
  if (ret <= 0) return ret;
  rt = Now();
  packets.push_back({ rb.size()-ret, ret });
  if (FLAGS_network_debug) INFO("addpacket(", Name(), ", ", len, ")");
  return ret;
}

int Connection::Write(const char *buf, int len) {
  if (!buf || len<0) return -1;
  if (!len) len = strlen(buf);
  if (wb.size() + len > wb.Capacity()) return ERRORv(-1, Name(), ": write queue full");

  if (!wb.size() && len) {
    writable = true;
    app->network->UpdateActive(this);
  }
  wb.Add(buf, len);
  wb.EnsureZeroTerminated();
  wt = Now();
  return len;
}

int Connection::ReadFlush(int len) {
  if (len<0) return -1;
  if (!len) return rb.size();
  if (rb.size()-len < 0) return ERRORv(-1, Name(), ": read queue underflow: ", len, " > ", rb.size());
  rb.Flush(len);
  rb.EnsureZeroTerminated();
  if (control_messages) for (auto &s : transferred_socket) { s.offset -= len; CHECK_GE(s.offset, 0); }
  return rb.size();
}

int Connection::WriteFlush(const char *buf, int len) {
  int wrote = 0;
  if (ssl) {
#ifdef LFL_OPENSSL
    if ((wrote = BIO_write(bio, buf, len)) < 0) {
      if (!SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": send: ", strerror(errno));
      wrote = 0;
    }
#endif
  }
  else {
    if ((wrote = send(socket, buf, len, 0)) < 0) {
      if (!SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": send: ", strerror(errno));
      wrote = 0;
    }
  }
  if (FLAGS_network_debug) INFO("write(", socket, ", ", wrote, ", '", buf, "')");
  return wrote;
}

int Connection::WriteVFlush(const iovec *iov, int len) {
  int wrote = 0;
  if (ssl) {
#ifdef LFL_OPENSSL
#endif
  }
  else {
    if ((wrote = writev(socket, iov, len)) < 0) {
      if (!SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": send: ", strerror(errno));
      wrote = 0;
    }
  }
  if (FLAGS_network_debug) INFO("writev(", socket, ", ", wrote, ", '", len, "')");
  return wrote;
}

int Connection::WriteFlush(const char *buf, int len, int transfer_socket) {
  struct iovec iov = { (void*)buf, static_cast<size_t>(len) };
  return WriteVFlush(&iov, 1);
}

int Connection::WriteVFlush(const iovec *iov, int len, int transfer_socket) {
  int wrote = 0;
#if defined(WIN32) || defined(LFL_MOBILE)
  return -1;
#else
  char control[CMSG_SPACE(sizeof (int))];
  memzero(control);
  struct msghdr msg;
  memzero(msg);
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  memcpy(CMSG_DATA(cmsg), &transfer_socket, sizeof(int));
  msg.msg_controllen = cmsg->cmsg_len;
  msg.msg_iov = const_cast<iovec*>(iov);
  msg.msg_iovlen = len;

  if ((wrote = sendmsg(socket, &msg, 0)) < 0) {
    if (!SystemNetwork::EWouldBlock())
      return ERRORv(-1, Name(), ": sendmsg(l=", SystemNetwork::IOVLen(iov, len), "): ", strerror(errno)); 
    wrote = 0;
  }
#endif
  if (FLAGS_network_debug) INFO("writev(", socket, ", ", wrote, ", '", len, "')");
  return wrote;
}

int Connection::WriteFlush() {
  int wrote = WriteFlush(wb.begin(), wb.size());
  if (wrote) wb.Flush(wrote);
  wb.EnsureZeroTerminated();
  return wrote;
}

int Connection::SendTo(const char *buf, int len) { return SystemNetwork::SendTo(socket, addr, port, buf, len); }

/* Service */

void Service::Close(Connection *c) {
  app->network->active.Del(c->socket);
  if (!c->detach) SystemNetwork::CloseSocket(c->socket);
  if (connect_src_pool && (c->src_addr || c->src_port)) connect_src_pool->Close(c->src_addr, c->src_port);
}

int Service::OpenSocket(Connection *c, int protocol, int blocking, IPV4EndpointSource* src_pool) {
  Socket fd = SystemNetwork::OpenSocket(protocol);
  if (fd == -1) return -1;

  if (!blocking) {
    if (SystemNetwork::SetSocketBlocking(fd, 0))
    { SystemNetwork::CloseSocket(fd); return -1; }
  }

  if (src_pool) {
    IPV4Endpoint last_src;
    for (int i=0, max_bind_attempts=10; /**/; i++) {
      src_pool->Get(&c->src_addr, &c->src_port);
      if (i >= max_bind_attempts || (i && c->src_addr == last_src.addr && c->src_port == last_src.port))
      { ERROR("connect-bind ", IPV4::Text(c->src_addr, c->src_port), ": ", strerror(errno)); SystemNetwork::CloseSocket(fd); return -1; }

      if (SystemNetwork::Bind(fd, c->src_addr, c->src_port) != -1) break;
      src_pool->BindFailed(c->src_addr, c->src_port);
      last_src = IPV4Endpoint(c->src_addr, c->src_port);
    }
  }

  c->socket = fd;
  return 0;
}

Socket Service::Listen(IPV4::Addr addr, int port, Listener *listener) {
  Socket fd = -1;
  if (listener->ssl) {
#ifdef LFL_OPENSSL
    listener->ssl = BIO_new_accept((char*)StrCat(port).c_str());
    BIO_ctrl(listener->ssl, BIO_C_SET_ACCEPT, 1, (void*)"a");
    BIO_set_bind_mode(listener->ssl, BIO_BIND_REUSEADDR);
    if (BIO_do_accept(listener->ssl) <= 0) return ERRORv(-1, "ssl_listen: ", -1);
    BIO_get_fd(listener->ssl, &listener->socket);
    BIO_set_accept_bios(listener->ssl, BIO_new_ssl(lfapp_ssl, 0));
#endif
  } else {
    if ((listener->socket = SystemNetwork::Listen(protocol, addr, port)) == -1)
      return ERRORv(-1, "SystemNetwork::Listen(", protocol, ", ", port, "): ", SystemNetwork::LastError());
  }
  app->network->active.Add(listener->socket, SocketSet::READABLE, &listener->self_reference);
  return listener->socket;
}

Connection *Service::Accept(int state, Socket socket, IPV4::Addr addr, int port) {
  Connection *c = new Connection(this, state, socket, addr, port);
  conn[c->socket] = c;
  app->network->active.Add(c->socket, SocketSet::READABLE, &c->self_reference);
  return c;
}

Connection *Service::Connect(IPV4::Addr addr, int port, IPV4::Addr src_addr, int src_port, Callback *detach) {
  SingleIPV4Endpoint src_pool(src_addr, src_port);
  return Connect(addr, port, &src_pool, detach);
}

Connection *Service::Connect(IPV4::Addr addr, int port, IPV4EndpointSource *src_pool, Callback *detach) {
  Connection *c = new Connection(this, Connection::Connecting, addr, port, detach);
  if (Service::OpenSocket(c, protocol, 0, src_pool ? src_pool : connect_src_pool))
  { ERROR(c->Name(), ": connecting: ", SystemNetwork::LastError()); delete c; return 0; }

  int connected = 0;
  if (SystemNetwork::Connect(c->socket, c->addr, c->port, &connected) == -1) {
    ERROR(c->Name(), ": connecting: ", SystemNetwork::LastError());
    SystemNetwork::CloseSocket(c->socket);
    delete c;
    return 0;
  }
  INFO(c->Name(), ": connecting");
  conn[c->socket] = c;

  if (connected) {
    /* connected 3 */ 
    c->SetConnected();
    c->SetSourceAddress();
    INFO(c->Name(), ": connected");
    if (this->Connected(c) < 0) c->SetError();
    if (c->handler) { if (c->handler->Connected(c) < 0) { ERROR(c->Name(), ": handler connected"); c->SetError(); } }
    if (c->detach) { conn.erase(c->socket); Detach(c); }
    app->network->UpdateActive(c);
  } else {
    app->network->active.Add(c->socket, SocketSet::READABLE|SocketSet::WRITABLE, &c->self_reference);
  }
  return c;
}

Connection *Service::Connect(const string &hostport, int default_port, Callback *detach) {
  IPV4::Addr addr; int port;
  if (!HTTP::ResolveHost(hostport.c_str(), 0, &addr, &port, 0, default_port)) return ERRORv(nullptr, "resolve ", hostport, " failed");
  return Connect(addr, port, NULL, detach);
}

Connection *Service::SSLConnect(SSL_CTX *sslctx, const string &hostport, int default_port, Callback *detach) {
#ifdef LFL_OPENSSL
  if (!sslctx) sslctx = lfapp_ssl;
  if (!sslctx) return ERRORv(nullptr, "no ssl: ", -1);

  Connection *c = new Connection(this, Connection::Connecting, 0, 0, detach);
  if (!HTTP::ResolveHost(hostport.c_str(), 0, &c->addr, &c->port, true, default_port)) return ERRORv(nullptr, "resolve: ", hostport);

  c->bio = BIO_new_ssl_connect(sslctx);
  BIO_set_conn_hostname(c->bio, hostport.c_str());
  BIO_get_ssl(c->bio, &c->ssl);
  BIO_set_nbio(c->bio, 1);

  int ret = BIO_do_connect(c->bio);
  if (ret < 0 && !BIO_should_retry(c->bio)) {
    ERROR(hostport, ": BIO_do_connect: ", ret);
    delete c;
    return 0;
  }

  BIO_get_fd(c->bio, &c->socket);

  INFO(c->Name(), ": connecting (fd=", c->socket, ")");
  conn[c->socket] = c;
  app->network->active.Add(c->socket, SocketSet::WRITABLE, &c->self_reference);
  return c;
#else
  return 0;
#endif
}

Connection *Service::SSLConnect(SSL_CTX *sslctx, IPV4::Addr addr, int port, Callback *detach) {
#ifdef LFL_OPENSSL
  if (!sslctx) sslctx = lfapp_ssl;
  if (!sslctx) return ERRORv(nullptr, "no ssl: ", -1);

  Connection *c = new Connection(this, Connection::Connecting, addr, port, detach);
  c->bio = BIO_new_ssl_connect(sslctx);
  BIO_set_conn_ip(c->bio, (char*)&addr);
  BIO_set_conn_int_port(c->bio, (char*)&port);
  BIO_get_ssl(c->bio, &c->ssl);
  BIO_set_nbio(c->bio, 1);

  int ret = BIO_do_connect(c->bio);
  if (ret < 0 && !BIO_should_retry(c->bio)) {
    ERROR(c->Name(), ": BIO_do_connect: ", ret);
    delete c;
    return 0;
  }

  BIO_get_fd(c->bio, &c->socket);

  INFO(c->Name(), ": connecting (fd=", c->socket, ")");
  conn[c->socket] = c;
  app->network->active.Add(c->socket, SocketSet::WRITABLE, &c->self_reference);
  return c;
#else
  return 0;
#endif
}

Connection *Service::EndpointConnect(const string &endpoint_name) {
  Connection *c = new Connection(this, Connection::Connected, -1);
  c->endpoint_name = endpoint_name;
  endpoint[endpoint_name] = c;

  /* connected 4 */
  if (this->Connected(c) < 0) c->SetError();
  INFO(Protocol::Name(protocol), "(", (void*)this, ") endpoint connect: ", endpoint_name);
  if (c->handler) { if (c->handler->Connected(c) < 0) { ERROR(c->Name(), ": handler connected"); c->SetError(); } }
  return c;
}

void Service::EndpointReadCB(string *endpoint_name, string *packet) {
  EndpointRead(*endpoint_name, packet->c_str(), packet->size());
  delete endpoint_name;
  delete packet;
}

void Service::EndpointRead(const string &endpoint_name, const char *buf, int len) {
  if (len) CHECK(buf);
  if (!MainThread()) return RunInMainThread(new Callback(bind(&Service::EndpointReadCB, this, new string(endpoint_name), new string(buf, len))));

  Service::EndpointMap::iterator ep = endpoint.find(endpoint_name);
  if (ep == endpoint.end()) { 
    if (!endpoint_read_autoconnect) return ERROR("unknown endpoint ", endpoint_name);
    if (!EndpointConnect(endpoint_name)) return ERROR("endpoint_read_autoconnect ", endpoint_name);
    ep = endpoint.find(endpoint_name);
    CHECK(ep != endpoint.end());
  }

  Connection *c = ep->second; int ret;
  if ((ret = c->AddPacket(buf, len)) != len) 
  { ERROR(c->Name(), ": addpacket(", len, ")"); c->SetError(); return; }
}

void Service::EndpointClose(const string &endpoint_name) {
  INFO(Protocol::Name(protocol), "(", (void*)this, ") endpoint close: ", endpoint_name);
  Service::EndpointMap::iterator ep = endpoint.find(endpoint_name);
  if (ep != endpoint.end()) ep->second->SetError();
}

void Service::Detach(Connection *c) {
  Service::Close(c);
  c->readable = c->writable = 0;
  RunInMainThread(new Callback([=]() { (*c->detach)(); }));
  app->scheduler.Wakeup(0);
}

/* Network */

int Network::Init() {
  INFO("Network::Init()");
#ifdef LFL_OPENSSL
  SSL_load_error_strings();
  SSL_library_init(); 

  if (bool client_only=0) lfapp_ssl = SSL_CTX_new(SSLv23_client_method());
  else                    lfapp_ssl = SSL_CTX_new(SSLv23_method());

  if (!lfapp_ssl) FATAL("no SSL_CTX: ", ERR_reason_error_string(ERR_get_error()));
  SSL_CTX_set_verify(lfapp_ssl, SSL_VERIFY_NONE, 0);

  if (FLAGS_ssl_certfile.size() && FLAGS_ssl_keyfile.size()) {
    if (!SSL_CTX_use_certificate_file(lfapp_ssl, FLAGS_ssl_certfile.c_str(), SSL_FILETYPE_PEM)) return ERRORv(-1, "SSL_CTX_use_certificate_file ", ERR_reason_error_string(ERR_get_error()));
    if (!SSL_CTX_use_PrivateKey_file(lfapp_ssl, FLAGS_ssl_keyfile.c_str(), SSL_FILETYPE_PEM)) return ERRORv(-1, "SSL_CTX_use_PrivateKey_file ",  ERR_reason_error_string(ERR_get_error()));
    if (!SSL_CTX_check_private_key(lfapp_ssl)) return ERRORv(-1, "SSL_CTX_check_private_key ", ERR_reason_error_string(ERR_get_error()));
  }
#endif
  Enable(Singleton<UDPClient>::Get());
  Enable(Singleton<HTTPClient>::Get());
  Enable(Singleton<UnixClient>::Get());

  vector<IPV4::Addr> nameservers;
  if (FLAGS_nameserver.empty()) {
    INFO("Network::Init(): Enable(new Resolver(defaultNameserver()))");
    Resolver::DefaultNameserver(&nameservers);
  } else {
    INFO("Network::Init(): Enable(new Resolver(", FLAGS_nameserver, "))");
    IPV4::ParseCSV(FLAGS_nameserver, &nameservers);
  }
  for (auto &n : nameservers) Singleton<Resolver>::Get()->Connect(n);

  return 0;
}

int Network::Shutdown(const vector<Service*> &st) { int ret = 0; for (auto s : st) if (Shutdown(s) < 0) ret = -1; return ret; }
int Network::Disable (const vector<Service*> &st) { int ret = 0; for (auto s : st) if (Disable (s) < 0) ret = -1; return ret; }
int Network::Enable  (const vector<Service*> &st) { int ret = 0; for (auto s : st) if (Enable  (s) < 0) ret = -1; return ret; }

int Network::Enable(Service *s) {

  /* listen */
  if (s->listen.size() && !s->initialized) {
    s->initialized = true;
    vector<string> removelist;
    for (Service::ListenMap::iterator i = s->listen.begin(); i != s->listen.end(); ++i) {
      const IPV4Endpoint *listen_addr = IPV4Endpoint::FromString(i->first.c_str());
      if (s->Listen(listen_addr->addr, listen_addr->port, i->second) == -1)
      { delete i->second; removelist.push_back(i->first); }
    }
    for (vector<string>::const_iterator i = removelist.begin(); i != removelist.end(); ++i) s->listen.erase(*i);
  }

  /* insert */
  auto i = find(service_table.begin(), service_table.end(), s);
  if (i != service_table.end()) return 0;
  service_table.push_back(s);
  return 0;
}

int Network::Disable(Service *s) {
  auto i = find(service_table.begin(), service_table.end(), s);
  if (i == service_table.end()) return -1;
  service_table.erase(i);
  return 0;
}

void Network::ConnClose(Service *svc, Connection *c, ServiceEndpointEraseList *removelist) {
  if (c->handler) c->handler->Close(c);
  svc->Close(c);
  if (removelist) removelist->AddSocket(svc, c->socket);
  delete c;
}
void Network::ConnCloseAll(Service *svc) {
  for (auto i = svc->conn.begin(), e = svc->conn.end(); i != e; ++i) ConnClose(svc, i->second, 0);
  svc->conn.clear();
}

void Network::EndpointRead(Service *svc, const char *name, const char *buf, int len) { return svc->EndpointRead(name, buf, len); }
void Network::EndpointClose(Service *svc, Connection *c, ServiceEndpointEraseList *removelist, const string &epk) {
  if (c->handler) c->handler->Close(c);
  if (svc->listen.empty()) svc->Close(c);
  if (removelist) removelist->AddEndpoint(svc, epk);
  delete c;
}
void Network::EndpointCloseAll(Service *svc) {
  for (Service::EndpointMap::iterator i = svc->endpoint.begin(); i != svc->endpoint.end(); ++i)
    EndpointClose(svc, i->second, 0, "");
  svc->endpoint.clear();
}

int Network::Shutdown(Service *svc) {
  ConnCloseAll(svc);
  EndpointCloseAll(svc);
  return 0;
}

int Network::Frame(unsigned clicks) {
  static const int listener_type = TypeId<Listener>(), connection_type = TypeId<Connection>();
  ServiceEndpointEraseList removelist;

  /* select */
  if (active.Select(select_time))
    return ERRORv(-1, "SocketSet.select: ", SystemNetwork::LastError());

#ifndef LFL_NETWORK_MONOLITHIC_FRAME
  /* iterate events */
  for (active.cur_event = 0; active.cur_event < active.num_events; active.cur_event++) {
    typed_ptr *tp = (typed_ptr*)active.events[active.cur_event].data.ptr;
    if      (tp->type == connection_type) { Connection *c=(Connection*)tp->value; active.cur_fd = c->socket; TCPConnectionFrame(c->svc, c, &removelist); }
    else if (tp->type == listener_type)   { Listener   *l=(Listener*)  tp->value; active.cur_fd = l->socket; AcceptFrame(l->svc, l); }
    else FATAL("unknown type", tp->type);
  }
#endif

  /* pre loop */
  for (int svc_i = 0, svc_l = service_table.size(); svc_i < svc_l; ++svc_i) {
    Service *svc = service_table[svc_i];

#ifdef LFL_NETWORK_MONOLITHIC_FRAME
    /* answer listening sockets & UDP server read */
    for (auto i = svc->listen.begin(), e = svc->listen.end(); i != e; ++i) 
      if (active.GetReadable(i->second->socket)) AcceptFrame(svc, i->second);

    /* iterate connections */
    for (auto i = svc->conn.begin(), e = svc->conn.end(); i != e; ++i)
      TCPConnectionFrame(svc, i->second, &removelist);
#endif

    /* iterate endpoints */
    for (auto i = svc->endpoint.begin(), e = svc->endpoint.end(); i != e; i++) {
      UDPConnectionFrame(svc, (*i).second, &removelist, (*i).first);
    }

    /* remove closed */
    removelist.Erase();

    if (svc->heartbeats) { /* connection heartbeats */
      for (auto i = svc->conn.begin(), e = svc->conn.end(); i != e; ++i) {
        Connection *c = i->second;
        int ret = c->handler ? c->handler->Heartbeat(c) : 0;
        if (c->state == Connection::Error || ret < 0) ConnClose(svc, c, &removelist);
      }

      for (auto i = svc->endpoint.begin(), e = svc->endpoint.end(); i != e; ++i) {
        Connection *c = i->second;
        int ret = c->handler ? c->handler->Heartbeat(c) : 0;
        if (c->state == Connection::Error || ret < 0) EndpointClose(svc, c, &removelist, c->endpoint_name);
      }

      removelist.Erase();
    }

    /* svc specific frame */
    svc->Frame();
  }
  return 0;
}

void Network::AcceptFrame(Service *svc, Listener *listener) {
  for (;;) {
    Connection *c = 0; bool inserted = false;
    if (listener->ssl) { /* SSL accept */
#ifdef LFL_OPENSSL
      struct sockaddr_in sin;
      int sinSize = sizeof(sin);
      Socket socket;
      if (BIO_do_accept(listener->ssl) <= 0) continue;
      BIO *bio = BIO_pop(listener->ssl);
      BIO_get_fd(bio, &socket);

      if (::getpeername(socket, (struct sockaddr *)&sin, (socklen_t*)&sinSize) < 0)
      { if (!SystemNetwork::EWouldBlock()) ERROR("getpeername: ", strerror(errno)); break; }

      /* insert socket */
      c = svc->Accept(Connection::Connected, socket, sin.sin_addr.s_addr, ntohs(sin.sin_port));
      if (!c) { SystemNetwork::CloseSocket(socket); continue; }
      c->bio = bio;
      BIO_set_nbio(c->bio, 1);
      BIO_get_ssl(c->bio, &listener->ssl);
#endif
    }
    else if (svc->protocol == Protocol::UDP) { /* UDP server read */
      struct sockaddr_in sin;
      int sinSize = sizeof(sin), inserted = 0, ret;
      char buf[2048];
      int len = recvfrom(listener->socket, buf, sizeof(buf)-1, 0, (struct sockaddr*)&sin, (socklen_t*)&sinSize);
      if (len <= 0) {
        if (SystemNetwork::EWouldBlock()) break;
        else { ERROR("recvfrom: ", SystemNetwork::LastError()); break; }
      }
      buf[len] = 0;

      IPV4Endpoint epk(sin.sin_addr.s_addr, ntohs(sin.sin_port));
      string epkstr = epk.ToString();
      Service::EndpointMap::iterator ep = svc->endpoint.find(epkstr);
      if (ep != svc->endpoint.end()) c = ep->second;
      else {
        svc->fake.addr = epk.addr;
        svc->fake.port = epk.port;
        svc->fake.socket = listener->socket;
        if (svc->UDPFilter(&svc->fake, buf, len)) continue;
        c = new Connection(svc, Connection::Connected, listener->socket, epk.addr, epk.port);
        svc->endpoint[epkstr] = c;
        inserted = true;
      }

      if ((ret = c->AddPacket(buf, len)) != len) 
      { ERROR(c->Name(), ": addpacket(", len, ")"); c->SetError(); continue; }
      if (!inserted) continue;
    }
    else { /* TCP accept */
      IPV4::Addr accept_addr = 0; 
      int accept_port = 0;
      Socket socket = SystemNetwork::Accept(listener->socket, &accept_addr, &accept_port);
      if (socket == -1) break;

      /* insert socket */
      c = svc->Accept(Connection::Connected, socket, accept_addr, accept_port);
      if (!c) { SystemNetwork::CloseSocket(socket); continue; }
    }
    if (!c) continue;

    /* connected 1 */
    c->SetSourceAddress();
    INFO(c->Name(), ": incoming connection (socket=", c->socket, ")");
    if (svc->Connected(c) < 0) c->SetError();
    if (c->handler) { if (c->handler->Connected(c) < 0) { ERROR(c->Name(), ": handler connected"); c->SetError(); } }
    if (c->detach) { svc->conn.erase(c->socket); svc->Detach(c); }
  }
}

void Network::TCPConnectionFrame(Service *svc, Connection *c, ServiceEndpointEraseList *removelist) {
  /* complete connecting sockets */
  if (c->state == Connection::Connecting && active.GetWritable(c->socket)) {
    int res=0, resSize=sizeof(res);
    if (!getsockopt(c->socket, SOL_SOCKET, SO_ERROR, (char*)&res, (socklen_t*)&resSize) && !res) {

      /* connected 2 */ 
      c->SetConnected();
      c->SetSourceAddress();
      INFO(c->Name(), ": connected");
      if (svc->Connected(c) < 0) c->SetError();
      if (c->handler) { if (c->handler->Connected(c) < 0) { ERROR(c->Name(), ": handler connected"); c->SetError(); } }
      if (c->detach) { removelist->AddSocket(svc, c->socket); svc->Detach(c); }
      UpdateActive(c);
      return;
    }

    errno = res;
    INFO(c->Name(), ": connect failed: ", SystemNetwork::LastError());
    c->SetError();
  }

  /* IO communicate */
  else if (c->state == Connection::Connected) do {
    if (svc->protocol == Protocol::UDP) {
      if (svc->listen.empty() && active.GetReadable(c->socket)) { /* UDP Client Read */
        if (c->ReadPackets()<0) { c->SetError(); break; }
      }
      if (c->packets.size()) {
        if (c->handler) { if (c->handler->Read(c) < 0) { ERROR(c->Name(), ": handler UDP read"); c->SetError(); } }
        c->packets.clear();
        c->ReadFlush(c->rb.size());
      }
    }
    else if (c->ssl || active.GetReadable(c->socket)) { /* TCP Read */
      if (c->Read()<0) { c->SetError(); break; }
      if (c->rb.size()) {
        if (c->handler) { if (c->handler->Read(c) < 0) { ERROR(c->Name(), ": handler read"); c->SetError(); } }
      }
    }

    if (c->wb.size() && active.GetWritable(c->socket)) {
      if (c->WriteFlush()<0) { c->SetError(); break; }
      if (!c->wb.size()) {
        c->writable = 0;
        if (c->handler) { if (c->handler->Flushed(c) < 0) { ERROR(c->Name(), ": handler flushed"); c->SetError(); } }
        UpdateActive(c);
      }
    }
  } while(0);

  /* error */
  if (c->state == Connection::Error) ConnClose(svc, c, removelist);
}

void Network::UDPConnectionFrame(Service *svc, Connection *c, ServiceEndpointEraseList *removelist, const string &epk) {
  if (c->state == Connection::Connected && c->packets.size()) {
    if (c->handler) { if (c->handler->Read(c) < 0) { ERROR(c->Name(), ": handler UDP read"); c->SetError(); } }
    c->packets.clear();
    c->ReadFlush(c->rb.size());
  }

  bool timeout; /* Timeout or error */
  if (c->state == Connection::Error || (timeout = (c->rt + Seconds(FLAGS_udp_idle_sec)) <= Now())) {
    INFO(c->Name(), ": ", timeout ? "timeout" : "error");
    EndpointClose(svc, c, removelist, epk);
  }
}

void Network::UpdateActive(Connection *c) {
  if (FLAGS_network_debug) INFO(c->Name(), " active = { ", c->readable?"READABLE":"", " , ", c->writable?"WRITABLE":"", " }");
  int flag = (c->readable?SocketSet::READABLE:0) | (c->writable?SocketSet::WRITABLE:0);
  active.Set(c->socket, flag, &c->self_reference);
}

/* NetworkThread */

NetworkThread::NetworkThread(Network *N, bool Init) : net(N), init(Init),
  rd(new Connection(Singleton<UnixClient>::Get(), new NetworkThread::ConnectionHandler())),
  wr(new Connection(Singleton<UnixClient>::Get(), new NetworkThread::ConnectionHandler())),
  thread(new Thread(bind(&NetworkThread::HandleMessagesLoop, this))) {
  Socket fd[2];
  CHECK(SystemNetwork::OpenSocketPair(fd));
  rd->state = wr->state = Connection::Connected;
  rd->socket = fd[0];
  wr->socket = fd[1];

  net->select_time = -1;
  rd->svc->conn[rd->socket] = rd;
  net->active.Add(rd->socket, SocketSet::READABLE, &rd->self_reference);
}

int NetworkThread::ConnectionHandler::Read(Connection *c) {
  int consumed = 0, s = sizeof(Callback*);
  for (; consumed + s <= c->rb.size(); consumed += s) HandleMessage(*reinterpret_cast<Callback**>(c->rb.begin() + consumed));
  if (consumed) c->ReadFlush(consumed);
  return 0;
}

/* UDP Client */

struct UDPClientHandler {
  struct PersistentConnection : public Connection::Handler {
    UDPClient::ResponseCB responseCB; UDPClient::HeartbeatCB heartbeatCB;
    PersistentConnection(UDPClient::ResponseCB RCB, UDPClient::HeartbeatCB HCB) : responseCB(RCB), heartbeatCB(HCB) {}

    int Heartbeat(Connection *c) { if (heartbeatCB) heartbeatCB(c); return 0; }
    void Close(Connection *c) { if (responseCB) responseCB(c, 0, 0); }
    int Read(Connection *c) {
      for (int i=0; i<c->packets.size() && responseCB; i++) {
        if (c->state != Connection::Connected) break;
        responseCB(c, c->rb.begin() + c->packets[i].offset, c->packets[i].len);
      }
      return 0;
    }
  };
};

Connection *UDPClient::PersistentConnection(const string &url, ResponseCB responseCB, HeartbeatCB heartbeatCB, int default_port) {
  IPV4::Addr ipv4_addr; int udp_port;
  if (!HTTP::ResolveURL(url.c_str(), (bool*)0, &ipv4_addr, &udp_port, (string*)0, (string*)0, default_port))
  { INFO(url, ": connect failed"); return 0; }

  Connection *c = Connect(ipv4_addr, udp_port);
  if (!c) { INFO(url, ": connect failed"); return 0; }

  c->handler = new UDPClientHandler::PersistentConnection(responseCB, heartbeatCB);
  return c;
}

/* HTTPClient */

struct HTTPClientHandler {
  struct Protocol : public Connection::Handler {
    int readHeaderLength, readContentLength, currentChunkLength, currentChunkRead;
    bool chunkedEncoding, fullChunkCBs;
    string content_type;

    Protocol(bool fullChunks) : fullChunkCBs(fullChunks) { reset(); }
    void reset() { readHeaderLength=0; readContentLength=0; currentChunkLength=0; currentChunkRead=0; chunkedEncoding=0; content_type.clear();  }

    int Read(Connection *c) {
      char *cur = c->rb.begin();
      if (!readHeaderLength) {
        StringPiece ct, cl, te;
        char *headers = cur, *headersEnd = HTTP::FindHeadersEnd(headers);
        if (!headersEnd) return 1;

        readHeaderLength = HTTP::GetHeaderLen(headers, headersEnd);
        HTTP::GrepHeaders(headers, headersEnd, 3, "Content-Type", &ct, "Content-Length", &cl, "Transfer-Encoding", &te);
        currentChunkLength = readContentLength = atoi(BlankNull(cl.data()));
        chunkedEncoding = te.str() == "chunked";
        content_type = ct.str();

        Headers(c, headers, readHeaderLength);
        cur += readHeaderLength;
      }
      for (;;) {
        if (chunkedEncoding && !currentChunkLength) {
          char *cur_in = cur;
          cur += IsNewline(cur);
          char *chunkHeader = cur;
          if (!(cur = (char*)NextLine(cur))) { cur=cur_in; break; }
          currentChunkLength = strtoul(chunkHeader, 0, 16);
        }

        int rb_left = c->rb.size() - (cur - c->rb.begin());
        if (rb_left <= 0) break;
        if (chunkedEncoding) {
          int chunk_left = currentChunkLength - currentChunkRead;
          if (chunk_left < rb_left) rb_left = chunk_left;
          if (rb_left < chunk_left && fullChunkCBs) break;
        }

        if (rb_left) Content(c, cur, rb_left);
        cur += rb_left;
        currentChunkRead += rb_left;
        if (currentChunkRead == currentChunkLength) currentChunkRead = currentChunkLength = 0;
      }
      if (cur != c->rb.begin()) c->ReadFlush(cur - c->rb.begin());
      return 0;
    }
    virtual void Headers(Connection *c, const char *headers, int len) {}
    virtual void Content(Connection *c, const char *headers, int len) {}
  };

  struct WGet : public Protocol {
    Service *svc;
    bool ssl;
    string host, path;
    int port;
    File *out;
    HTTPClient::ResponseCB cb;

    virtual ~WGet() { if (out) INFO("close ", out->Filename()); delete out; }
    WGet(Service *Svc, bool SSL, const string &Host, int Port, const string &Path, File *Out, HTTPClient::ResponseCB CB=HTTPClient::ResponseCB()) :
      Protocol(false), svc(Svc), ssl(SSL), host(Host), path(Path), port(Port), out(Out), cb(CB) {}

    int Connected(Connection *c) { return HTTPClient::request(c, HTTPServer::Method::GET, host.c_str(), path.c_str(), 0, 0, 0, false); }
    void Close(Connection *c) { if (cb) cb(c, 0, content_type, 0, 0); }

    void Headers(Connection *c, const char *headers, int len) {
      if (cb) cb(c, headers, content_type, 0, readContentLength);
    }

    void Content(Connection *c, const char *content, int len) {
      if (out) { if (out->Write(content, len) != len) ERROR("write ", out->Filename()); }
      if (cb) cb(c, 0, content_type, content, len);
    }

    void ResolverResponseCB(IPV4::Addr ipv4_addr, DNS::Response*) {
      Connection *c = 0;
      if (ipv4_addr != (IPV4::Addr)-1) {
        c =
#ifdef LFL_OPENSSL
          ssl ? svc->SSLConnect(lfapp_ssl, ipv4_addr, port) :
#endif
          svc->Connect(ipv4_addr, port);
      }
      if (!c) { if (cb) cb(0, 0, string(), 0, 0); delete this; }
      else c->handler = this;
    }
  };

  struct WPost : public WGet {
    string mimetype, postdata;
    WPost(Service *Svc, bool SSL, const string &Host, int Port, const string &Path, const string &Mimetype, const char *Postdata, int Postlen,
          HTTPClient::ResponseCB CB=HTTPClient::ResponseCB()) : WGet(Svc, SSL, Host, Port, Path, 0, CB), mimetype(Mimetype), postdata(Postdata,Postlen) {}

    int Connected(Connection *c) { return HTTPClient::request(c, HTTPServer::Method::POST, host.c_str(), path.c_str(), mimetype.data(), postdata.data(), postdata.size(), false); }
  };

  struct PersistentConnection : public Protocol {
    HTTPClient::ResponseCB responseCB;
    PersistentConnection(HTTPClient::ResponseCB RCB) : Protocol(true), responseCB(RCB) {}

    void Close(Connection *c) { if (responseCB) responseCB(c, 0, content_type, 0, 0); }
    void Content(Connection *c, const char *content, int len) {
      if (!readContentLength) FATAL("chunked transfer encoding not supported");
      if (responseCB) responseCB(c, 0, content_type, content, len);
      Protocol::reset();
    }
  };
};

int HTTPClient::request(Connection *c, int method, const char *host, const char *path, const char *postmime, const char *postdata, int postlen, bool persist) {
  string hdr, posthdr;

  if (postmime && postdata && postlen)
    posthdr = StringPrintf("Content-Type: %s\r\nContent-Length: %d\r\n", postmime, postlen);

  hdr = StringPrintf("%s /%s HTTP/1.1\r\nHost: %s\r\n%s%s\r\n",
                     HTTPServer::Method::name(method), path, host, persist?"":"Connection: close\r\n", posthdr.c_str());

  int ret = c->Write(hdr.data(), hdr.size());
  if (posthdr.empty()) return ret;
  if (ret != hdr.size()) return -1;
  return c->Write(postdata, postlen);
}

bool HTTPClient::WGet(const string &url, File *out, ResponseCB cb) {
  bool ssl; int tcp_port; string host, path, prot;
  if (!HTTP::ResolveURL(url.c_str(), &ssl, 0, &tcp_port, &host, &path, 0, &prot)) {
    if (prot != "file") return 0;
    string fn = StrCat(!host.empty() ? "/" : "", host , "/", path), content = LocalFile::FileContents(fn);
    if (!content.empty() && cb) cb(0, 0, string(), content.data(), content.size());
    if (cb)                     cb(0, 0, string(), 0,              0);
    return true;
  }

  if (!out && !cb) {
    string fn = BaseName(path);
    if (fn.empty()) fn = "index.html";
    out = new LocalFile(StrCat(LFAppDownloadDir(), fn), "w");
    if (!out->Opened()) { ERROR("open file"); delete out; return 0; }
  }

  HTTPClientHandler::WGet *handler = new HTTPClientHandler::WGet(this, ssl, host, tcp_port, path, out, cb);
  Singleton<Resolver>::Get()->NSLookup(host, bind(&HTTPClientHandler::WGet::ResolverResponseCB, handler, _1, _2));
  return true;
}

bool HTTPClient::WPost(const string &url, const string &mimetype, const char *postdata, int postlen, ResponseCB cb) {
  bool ssl; int tcp_port; string host, path;
  if (!HTTP::ResolveURL(url.c_str(), &ssl, 0, &tcp_port, &host, &path)) return 0;

  HTTPClientHandler::WPost *handler = new HTTPClientHandler::WPost(this, ssl, host, tcp_port, path, mimetype, postdata, postlen, cb);
  if (!Singleton<Resolver>::Get()->Resolve(Resolver::Request(host, DNS::Type::A, bind(&HTTPClientHandler::WGet::ResolverResponseCB, handler, _1, _2))))
  { ERROR("resolver: ", url); delete handler; return 0; }

  return true;
}

Connection *HTTPClient::PersistentConnection(const string &url, string *host, string *path, ResponseCB responseCB) {
  bool ssl; IPV4::Addr ipv4_addr; int tcp_port;
  if (!HTTP::ResolveURL(url.c_str(), &ssl, &ipv4_addr, &tcp_port, host, path)) return 0;

  Connection *c = 
#ifdef LFL_OPENSSL
    ssl ? SSLConnect(lfapp_ssl, ipv4_addr, tcp_port) : 
#endif
    Connect(ipv4_addr, tcp_port);

  if (!c) return 0;

  c->handler = new HTTPClientHandler::PersistentConnection(responseCB);
  return c;
}

/* HTTPServer */

struct HTTPServerConnection : public Connection::Handler {
  HTTPServer *server;
  bool persistent;
  Connection::Handler *refill;

  struct ClosedCallback {
    HTTPServer::ConnectionClosedCB cb;
    ClosedCallback(HTTPServer::ConnectionClosedCB CB) : cb(CB) {}
    void thunk(Connection *c) { cb(c); }
  };
  typedef vector<ClosedCallback> ClosedCB;
  ClosedCB closedCB;

  struct Dispatcher {
    int type; const char *url, *args, *headers, *postdata; int reqlen, postlen;
    void clear() { type=0; url=args=headers=postdata=0; reqlen=postlen=0; }
    bool empty() { return !type; }
    Dispatcher() { clear() ; }
    Dispatcher(int T, const char *U, const char *A, const char *H, int L) : type(T), url(U), args(A), headers(H), postdata(0), reqlen(L), postlen(0) {}
    int Thunk(HTTPServerConnection *httpserv, Connection *c) {
      if (c->rb.size() < reqlen) return 0;
      int ret = httpserv->Dispatch(c, type, url, args, headers, postdata, postlen);
      c->ReadFlush(reqlen);
      clear();
      return ret;
    }
  } dispatcher;

  ~HTTPServerConnection() { delete refill; }
  HTTPServerConnection(HTTPServer *s) : server(s), persistent(true), refill(0) {}
  void Closed(Connection *c) { for (ClosedCB::iterator i = closedCB.begin(); i != closedCB.end(); i++) (*i).thunk(c); }

  int Read(Connection *c) {
    for (;;) {
      if (!dispatcher.empty()) return dispatcher.Thunk(this, c);

      char *end = HTTP::FindHeadersEnd(c->rb.begin());
      if (!end) return 0;

      char *start = HTTP::FindHeadersStart(c->rb.begin());
      if (!start) return -1;

      char *headers = start;
      int headersLen = HTTP::GetHeaderLen(headers, end);
      int cmdLen = start - c->rb.begin();

      char *method, *url, *args, *ver;
      if (HTTP::ParseRequest(c->rb.begin(), &method, &url, &args, &ver) == -1) return -1;

      int type;
      if      (!strcasecmp(method, "GET"))  type = HTTPServer::Method::GET;
      else if (!strcasecmp(method, "POST")) type = HTTPServer::Method::POST;
      else return -1;

      dispatcher = Dispatcher(type, url, args, headers, cmdLen+headersLen);

      StringPiece cnhv;
      if (type == HTTPServer::Method::POST) {
        StringPiece ct, cl;
        HTTP::GrepHeaders(headers, end, 3, "Connection", &cnhv, "Content-Type", &ct, "Content-Length", &cl);
        dispatcher.postlen = atoi(BlankNull(cl.data()));
        dispatcher.reqlen += dispatcher.postlen;
        if (dispatcher.postlen) dispatcher.postdata = headers + headersLen;
      }
      else {
        HTTP::GrepHeaders(headers, end, 1, "Connection", &cnhv);
      }
      persistent = PrefixMatch(BlankNull(cnhv.data()), "close\r\n");

      int ret = dispatcher.Thunk(this, c);
      if (ret < 0) return ret;
    }
  }

  int Dispatch(Connection *c, int type, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
    /* process request */
    Timer timer;
    HTTPServer::Response response = server->Request(c, type, url, args, headers, postdata, postlen);
    INFOf("%s %s %s %d cl=%d %f ms", c->Name().c_str(), HTTPServer::Method::name(type), url, response.code, response.content_length, timer.GetTime()); 
    if (response.refill) c->readable = 0;

    /* write response/headers */
    if (response.write_headers) {
      if (WriteHeaders(c, &response) < 0) return -1;
    }

    /* prepare/deliver content */
    if (response.content) {
      if (c->Write(response.content, response.content_length) < 0) return -1;
    }
    else if (response.refill) {
      if (refill) return -1;
      refill = response.refill;
    }
    else return -1;

    return 0;
  }

  int Flushed(Connection *c) { 
    if (refill) {
      int ret;
      if ((ret = refill->Flushed(c))) return ret;
      Replace<Connection::Handler>(&refill, 0);
      c->readable = 1;
      return 0;
    }
    if (!persistent) return -1;
    return 0;
  }

  static int WriteHeaders(Connection *c, HTTPServer::Response *r) {
    const char *code;
    if      (r->code == 200) code = "OK";
    else if (r->code == 400) code = "Bad Request";
    else return -1;

    char date[64];
    httptime(date, sizeof(date));

    char h[16384]; int hl=0;
    hl += sprint(h+hl, sizeof(h)-hl, "HTTP/1.1 %d %s\r\nDate: %s\r\n", r->code, code, date);

    if (r->content_length >= 0) hl += sprint(h+hl, sizeof(h)-hl, "Content-Length: %d\r\n", r->content_length);

    hl += sprint(h+hl, sizeof(h)-hl, "Content-Type: %s\r\n\r\n", r->type);

    return c->Write(h, hl);
  }
};

int HTTPServer::Connected(Connection *c) { c->handler = new HTTPServerConnection(this); return 0; }

void HTTPServer::connectionClosedCB(Connection *c, ConnectionClosedCB cb) {
  ((HTTPServerConnection*)c->handler)->closedCB.push_back(HTTPServerConnection::ClosedCallback(cb));
} 

HTTPServer::Response HTTPServer::Response::_400
(400, "text/html; charset=iso-8859-1", StringPiece::FromString
 ("<!DOCTYPE HTML PUBLIC \"-//IETF//DTD HTML 2.0//EN\">\r\n"
  "<html><head>\r\n"
  "<title>400 Bad Request</title>\r\n"
  "</head><body>\r\n"
  "<h1>Bad Request</h1>\r\n"
  "<p>Your browser sent a request that this server could not understand.<br />\r\n"
  "</p>\r\n"
  "<hr>\r\n"
  "</body></html>\r\n"));

/* FileResource */

struct HTTPServerFileResourceHandler : public Connection::Handler {
  LocalFile f;
  HTTPServerFileResourceHandler(const string &fn) : f(fn, "r") {}
  int Flushed(Connection *c) {
    if (!f.Opened()) return 0;
    c->writable = 1;
    c->wb.buf.len = f.Read(c->wb.begin(), c->wb.Capacity());
    if (c->wb.buf.len < c->wb.Capacity()) return 0;
    return 1;
  }
};

HTTPServer::Response HTTPServer::FileResource::Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  if (!size) return HTTPServer::Response::_400;
  return Response(type, size, new HTTPServerFileResourceHandler(filename));
}

/* ConsoleResource */

HTTPServer::Response HTTPServer::ConsoleResource::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  StringPiece v;
  if (args) HTTP::GrepURLArgs(args, 0, 1, "v", &v);
  app->shell.Run(v.str());
  string response = StrCat("<html>Shell::run('", v.str(), "')<br/></html>\n");
  return HTTPServer::Response("text/html; charset=UTF-8", &response);
}

/* StreamResource */

#ifdef LFL_FFMPEG
struct StreamResourceClient : public Connection::Handler {
  Connection *conn;
  HTTPServer::StreamResource *resource;
  AVFormatContext *fctx;
  microseconds start;

  StreamResourceClient(Connection *c, HTTPServer::StreamResource *r) : conn(c), resource(r), start(0) {
    resource->subscribers[this] = conn;
    fctx = avformat_alloc_context();
    CopyAVFormatContextStreams(fctx, resource->fctx);
    fctx->max_delay = (int)(0.7*AV_TIME_BASE);
  }
  virtual ~StreamResourceClient() {
    resource->subscribers.erase(this);
    FreeAVFormatContext(fctx);
  }

  int Flushed(Connection *c) { return 1; }
  void Open() { if (avio_open_dyn_buf(&fctx->pb)) ERROR("avio_open_dyn_buf"); }

  void Write(AVPacket *pkt, microseconds timestamp) {        
    Open();
    if (start == microseconds(0)) start = timestamp;
    if (timestamp != microseconds(0)) {
      AVStream *st = fctx->streams[pkt->stream_index];
      AVRational r = {1, 1000000};
      unsigned t = (timestamp - start).count();
      pkt->pts = av_rescale_q(t, r, st->time_base);
    }
    int ret;
    if ((ret = av_interleaved_write_frame(fctx, pkt))) ERROR("av_interleaved_write_frame: ", ret);
    Flush();
  }

  void WriteHeader() {
    Open();
    if (avformat_write_header(fctx, 0)) ERROR("av_write_header");
    avio_flush(fctx->pb);
    Flush();
  }

  void Flush() {
    char *buf=0; int len=0;
    if (!(len = avio_close_dyn_buf(fctx->pb, (uint8_t**)&buf))) return;
    if (len < 0) return ERROR("avio_close_dyn_buf");
    if (conn->Write(buf, len) < 0) conn->SetError();
    av_free(buf);
  }

  static void FreeAVFormatContext(AVFormatContext *fctx) {
    for (int i=0; i<fctx->nb_streams; i++) av_freep(&fctx->streams[i]);
    av_free(fctx);
  }

  static void CopyAVFormatContextStreams(AVFormatContext *dst, AVFormatContext *src) {
    if (!dst->streams) {
      dst->nb_streams = src->nb_streams;
      dst->streams = (AVStream**)av_mallocz(sizeof(AVStream*) * src->nb_streams);
    }

    for (int i=0; i<src->nb_streams; i++) {
      AVStream *s = (AVStream*)av_mallocz(sizeof(AVStream));
      *s = *src->streams[i];
      s->priv_data = 0;
      s->codec->frame_number = 0;
      dst->streams[i] = s;
    }

    dst->oformat = src->oformat;
    dst->nb_streams = src->nb_streams;
  }

  static AVFrame *AllocPicture(enum PixelFormat pix_fmt, int width, int height) {
    AVFrame *picture = avcodec_alloc_frame();
    if (!picture) return 0;
    int size = avpicture_get_size(pix_fmt, width, height);
    uint8_t *picture_buf = (uint8_t*)av_malloc(size);
    if (!picture_buf) { av_free(picture); return 0; }
    avpicture_fill((AVPicture *)picture, picture_buf, pix_fmt, width, height);
    return picture;
  } 
  static void FreePicture(AVFrame *picture) {
    av_free(picture->data[0]);
    av_free(picture);
  }

  static AVFrame *AllocSamples(int num_samples, int num_channels, short **samples_out) {
    AVFrame *samples = avcodec_alloc_frame();
    if (!samples) return 0;
    samples->nb_samples = num_samples;
    int size = 2 * num_samples * num_channels;
    uint8_t *samples_buf = (uint8_t*)av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE);
    if (!samples_buf) { av_free(samples); return 0; }
    avcodec_fill_audio_frame(samples, num_channels, AV_SAMPLE_FMT_S16, samples_buf, size, 1);
    memset(samples_buf+size, 0, FF_INPUT_BUFFER_PADDING_SIZE);
    if (samples_out) *samples_out = (short*)samples_buf;
    return samples;
  }
  static void FreeSamples(AVFrame *picture) {
    av_free(picture->data[0]);
    av_free(picture);
  }
};

HTTPServer::StreamResource::~StreamResource() {
  delete resampler.out;
  if (audio && audio->codec) avcodec_close(audio->codec);
  if (video && video->codec) avcodec_close(video->codec);
  if (picture) StreamResourceClient::FreePicture(picture);
  if (samples) StreamResourceClient::FreeSamples(picture);
  StreamResourceClient::FreeAVFormatContext(fctx);
}

HTTPServer::StreamResource::StreamResource(const char *oft, int Abr, int Vbr) : fctx(0), open(0), abr(Abr), vbr(Vbr), 
  audio(0), samples(0), sample_data(0), frame(0), channels(0), resamples_processed(0), video(0), picture(0), conv(0) {
  fctx = avformat_alloc_context();
  fctx->oformat = av_guess_format(oft, 0, 0);
  if (!fctx->oformat) { ERROR("guess_format '", oft, "' failed"); return; }
  INFO("StreamResource: format ", fctx->oformat->mime_type);
  OpenStreams(FLAGS_lfapp_audio, FLAGS_lfapp_camera);
}

HTTPServer::Response HTTPServer::StreamResource::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  if (!open) return HTTPServer::Response::_400;
  Response response(fctx->oformat->mime_type, -1, new StreamResourceClient(c, this), false);
  if (HTTPServerConnection::WriteHeaders(c, &response) < 0) { c->SetError(); return response; }
  ((StreamResourceClient*)response.refill)->WriteHeader();
  return response;
}

void HTTPServer::StreamResource::OpenStreams(bool A, bool V) {
  if (V) {
    CHECK(!video);
    video = avformat_new_stream(fctx, 0);
    video->id = fctx->nb_streams;
    AVCodecContext *vc = video->codec;

    vc->codec_type = AVMEDIA_TYPE_VIDEO;
    vc->codec_id = CODEC_ID_H264;
    vc->codec_tag = av_codec_get_tag(fctx->oformat->codec_tag, vc->codec_id);

    vc->width = 576;
    vc->height = 342;
    vc->bit_rate = vbr;
    vc->time_base.num = 1;
    vc->time_base.den = FLAGS_camera_fps;
    vc->pix_fmt = PIX_FMT_YUV420P;

    /* x264 defaults */
    vc->me_range = 16;
    vc->max_qdiff = 4;
    vc->qmin = 10;
    vc->qmax = 51;
    vc->qcompress = 0.6;

    if (fctx->oformat->flags & AVFMT_GLOBALHEADER) vc->flags |= CODEC_FLAG_GLOBAL_HEADER;

    AVCodec *codec = avcodec_find_encoder(vc->codec_id);
    if (avcodec_open2(vc, codec, 0) < 0) return ERROR("avcodec_open2");
    if (!vc->codec) return ERROR("no video codec");

    if (vc->pix_fmt != PIX_FMT_YUV420P) return ERROR("pix_fmt ", vc->pix_fmt, " != ", PIX_FMT_YUV420P);
    if (!(picture = StreamResourceClient::AllocPicture(vc->pix_fmt, vc->width, vc->height))) return ERROR("AllocPicture");
  }

  if (0 && A) {
    audio = avformat_new_stream(fctx, 0);
    audio->id = fctx->nb_streams;
    AVCodecContext *ac = audio->codec;

    ac->codec_type = AVMEDIA_TYPE_AUDIO;
    ac->codec_id = CODEC_ID_MP3;
    ac->codec_tag = av_codec_get_tag(fctx->oformat->codec_tag, ac->codec_id);

    ac->channels = FLAGS_chans_in;
    ac->bit_rate = abr;
    ac->sample_rate = 22050;
    ac->sample_fmt = AV_SAMPLE_FMT_S16P;
    ac->channel_layout = AV_CH_LAYOUT_STEREO;

    if (fctx->oformat->flags & AVFMT_GLOBALHEADER) ac->flags |= CODEC_FLAG_GLOBAL_HEADER;

    AVCodec *codec = avcodec_find_encoder(ac->codec_id);
    if (avcodec_open2(ac, codec, 0) < 0) return ERROR("avcodec_open2");
    if (!ac->codec) return ERROR("no audio codec");

    if (!(frame = ac->frame_size)) return ERROR("empty frame size");
    channels = ac->channels;

    if (!(samples = StreamResourceClient::AllocSamples(frame, channels, &sample_data))) return ERROR("AllocPicture");
  }

  open = 1;
}

void HTTPServer::StreamResource::Update(int audio_samples, bool video_sample) {
  if (!open || !subscribers.size()) return;

  AVCodecContext *vc = video ? video->codec : 0;
  AVCodecContext *ac = audio ? audio->codec : 0;

  if (ac && audio_samples) {
    if (!resampler.out) {
      resampler.out = new RingBuf(ac->sample_rate, ac->sample_rate*channels);
      resampler.Open(resampler.out, FLAGS_chans_in, FLAGS_sample_rate, Sample::S16,
                     channels,       ac->sample_rate,   Sample::FromFFMpegId(ac->channel_layout));
    };
    RingBuf::Handle L(app->audio->IL, app->audio->IL->ring.back-audio_samples, audio_samples);
    RingBuf::Handle R(app->audio->IR, app->audio->IR->ring.back-audio_samples, audio_samples);
    if (resampler.Update(audio_samples, &L, FLAGS_chans_in > 1 ? &R : 0)) open=0;
  }

  for (;;) {
    bool asa = ac && resampler.output_available >= resamples_processed + frame * channels;
    bool vsa = vc && video_sample;
    if (!asa && !vsa) break;
    if (vc && !vsa) break;

    if (!vsa) { SendAudio(); continue; }       
    if (!asa) { SendVideo(); video_sample=0; continue; }

    int audio_behind = resampler.output_available - resamples_processed;
    microseconds audio_timestamp = resampler.out->ReadTimestamp(0, resampler.out->ring.back - audio_behind);

    if (audio_timestamp < app->camera->image_timestamp) SendAudio();
    else { SendVideo(); video_sample=0; }
  }
}

void HTTPServer::StreamResource::SendAudio() {
  int behind = resampler.output_available - resamples_processed, got = 0;
  resamples_processed += frame * channels;

  AVCodecContext *ac = audio->codec;
  RingBuf::Handle H(resampler.out, resampler.out->ring.back - behind, frame * channels);

  /* linearize */
  for (int i=0; i<frame; i++) 
    for (int c=0; c<channels; c++)
      sample_data[i*channels + c] = H.Read(i*channels + c) * 32768.0;

  /* broadcast */
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  avcodec_encode_audio2(ac, &pkt, samples, &got);
  if (got) Broadcast(&pkt, H.ReadTimestamp(0));

  av_free_packet(&pkt);
}

void HTTPServer::StreamResource::SendVideo() {
  AVCodecContext *vc = video->codec;

  /* convert video */
  if (!conv)
    conv = sws_getContext(FLAGS_camera_image_width, FLAGS_camera_image_height, (PixelFormat)Pixel::ToFFMpegId(app->camera->image_format),
                          vc->width, vc->height, vc->pix_fmt, SWS_BICUBIC, 0, 0, 0);

  int camera_linesize[4] = { app->camera->image_linesize, 0, 0, 0 }, got = 0;
  sws_scale(conv, (uint8_t**)&app->camera->image, camera_linesize, 0, FLAGS_camera_image_height, picture->data, picture->linesize);

  /* broadcast */
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  avcodec_encode_video2(vc, &pkt, picture, &got);
  if (got) Broadcast(&pkt, app->camera->image_timestamp);

  av_free_packet(&pkt);
}

void HTTPServer::StreamResource::Broadcast(AVPacket *pkt, microseconds timestamp) {
  for (SubscriberMap::iterator i = subscribers.begin(); i != subscribers.end(); i++) {
    StreamResourceClient *client = (StreamResourceClient*)(*i).first;
    client->Write(pkt, timestamp);
  }
}
#endif /* LFL_FFMPEG */

#ifdef LFL_SSH_DEBUG
#define SSHTrace(...) INFO(__VA_ARGS__)
#else
#define SSHTrace(...)
#endif

struct SSHClientConnection : public Connection::Handler {
  enum { INIT=0, FIRST_KEXINIT=1, FIRST_KEXREPLY=2, FIRST_NEWKEYS=3, KEXINIT=4, KEXREPLY=5, NEWKEYS=6 };

  SSHClient::ResponseCB cb;
  Vault::LoadPasswordCB load_password_cb;
  Vault::SavePasswordCB save_password_cb;
  string V_C, V_S, KEXINIT_C, KEXINIT_S, H_text, session_id, integrity_c2s, integrity_s2c, decrypt_buf, host, user, pw;
  int state=0, packet_len=0, packet_MAC_len=0, MAC_len_c=0, MAC_len_s=0, encrypt_block_size=0, decrypt_block_size=0;
  unsigned sequence_number_c2s=0, sequence_number_s2c=0, password_prompts=0, userauth_fail=0;
  bool guessed_c=0, guessed_s=0, guessed_right_c=0, guessed_right_s=0, loaded_pw=0;
  unsigned char padding=0, packet_id=0;
  pair<int, int> pty_channel;
  std::mt19937 rand_eng;
  BigNumContext ctx;
  BigNum K;
  Crypto::DiffieHellman dh;
  Crypto::EllipticCurveDiffieHellman ecdh;
  Crypto::DigestAlgo kex_hash;
  Crypto::Cipher encrypt, decrypt;
  Crypto::CipherAlgo cipher_algo_c2s=0, cipher_algo_s2c=0;
  Crypto::MACAlgo mac_algo_c2s=0, mac_algo_s2c=0;
  ECDef curve_id;
  int kex_method=0, hostkey_type=0, mac_prefix_c2s=0, mac_prefix_s2c=0, window_c=0, window_s=0;
  int initial_window_size=1048576, max_packet_size=32768, term_width=80, term_height=25;

  SSHClientConnection(const SSHClient::ResponseCB &CB, const string &H) : cb(CB), V_C("SSH-2.0-LFL_1.0"), host(H), rand_eng(std::random_device{}()),
    pty_channel(1,-1), ctx(NewBigNumContext()), K(NewBigNum()) { Crypto::CipherInit(&encrypt); Crypto::CipherInit(&decrypt); }
  virtual ~SSHClientConnection() { ClearPassword(); FreeBigNumContext(ctx); FreeBigNum(K); Crypto::CipherFree(&encrypt); Crypto::CipherFree(&decrypt); }

  void Close(Connection *c) { cb(c, StringPiece()); }
  int Connected(Connection *c) {
    if (state != INIT) return -1;
    string version_text = StrCat(V_C, "\r\n");
    if (c->WriteFlush(version_text) != version_text.size()) return ERRORv(-1, c->Name(), ": write");
    if (!WriteKeyExchangeInit(c, false)) return ERRORv(-1, c->Name(), ": write");
    return 0;
  }
  int Read(Connection *c) {
    if (state == INIT) {
      int processed = 0;
      StringLineIter lines(c->rb.buf, StringLineIter::Flag::BlankLines);
      for (string line = IterNextString(&lines); !lines.Done(); line = IterNextString(&lines)) {
        SSHTrace(c->Name(), ": SSH_INIT: ", line);
        processed = lines.next_offset;
        if (PrefixMatch(line, "SSH-")) { V_S=line; state++; break; }
      }
      c->ReadFlush(processed);
      if (state == INIT) return 0;
    }
    for (;;) {
      bool encrypted = state > FIRST_NEWKEYS;
      if (!packet_len) {
        packet_MAC_len = MAC_len_s ? X_or_Y(mac_prefix_s2c, MAC_len_s) : 0;
        if (c->rb.size() < SSH::BinaryPacketHeaderSize || (encrypted && c->rb.size() < decrypt_block_size)) return 0;
        if (encrypted) decrypt_buf = ReadCipher(c, StringPiece(c->rb.begin(), decrypt_block_size));
        const char *packet_text = encrypted ? decrypt_buf.data() : c->rb.begin();
        packet_len = 4 + SSH::BinaryPacketLength(packet_text, &padding, &packet_id) + packet_MAC_len;
      }
      if (c->rb.size() < packet_len) return 0;
      if (encrypted) decrypt_buf +=
        ReadCipher(c, StringPiece(c->rb.begin() + decrypt_block_size, packet_len - decrypt_block_size - packet_MAC_len));

      sequence_number_s2c++;
      const char *packet_text = encrypted ? decrypt_buf.data() : c->rb.begin();
      Serializable::ConstStream s(packet_text + SSH::BinaryPacketHeaderSize,
                                  packet_len  - SSH::BinaryPacketHeaderSize - packet_MAC_len);
      if (encrypted && packet_MAC_len) {
        string mac = SSH::MAC(mac_algo_s2c, MAC_len_s, StringPiece(decrypt_buf.data(), packet_len - packet_MAC_len),
                              sequence_number_s2c-1, integrity_s2c, mac_prefix_s2c);
        if (mac != string(c->rb.begin() + packet_len - packet_MAC_len, packet_MAC_len))
          return ERRORv(-1, c->Name(), ": verify MAC failed");
      }

      int v;
      switch (packet_id) {
        case SSH::MSG_DISCONNECT::ID: {
          SSH::MSG_DISCONNECT msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_DISCONNECT");
          SSHTrace(c->Name(), ": MSG_DISCONNECT ", msg.reason_code, " ", msg.description.str());
        } break;

        case SSH::MSG_DEBUG::ID: {
          SSH::MSG_DEBUG msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_DEBUG");
          SSHTrace(c->Name(), ": MSG_DEBUG ", msg.message.str());
        } break;

        case SSH::MSG_KEXINIT::ID: {
          state = state == FIRST_KEXINIT ? FIRST_KEXREPLY : KEXREPLY;
          SSH::MSG_KEXINIT msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEXINIT");
          SSHTrace(c->Name(), ": MSG_KEXINIT ", msg.DebugString());

          int cipher_id_c2s=0, cipher_id_s2c=0, mac_id_c2s=0, mac_id_s2c=0;
          guessed_s = msg.first_kex_packet_follows;
          KEXINIT_S.assign(packet_text, packet_len - packet_MAC_len);
          if (!SSH::KEX   ::PreferenceIntersect(msg.kex_algorithms,                         &kex_method))    return ERRORv(-1, c->Name(), ": negotiate kex");
          if (!SSH::Key   ::PreferenceIntersect(msg.server_host_key_algorithms,             &hostkey_type))  return ERRORv(-1, c->Name(), ": negotiate hostkey");
          if (!SSH::Cipher::PreferenceIntersect(msg.encryption_algorithms_client_to_server, &cipher_id_c2s)) return ERRORv(-1, c->Name(), ": negotiate c2s cipher");
          if (!SSH::Cipher::PreferenceIntersect(msg.encryption_algorithms_server_to_client, &cipher_id_s2c)) return ERRORv(-1, c->Name(), ": negotiate s2c cipher");
          if (!SSH::MAC   ::PreferenceIntersect(msg.mac_algorithms_client_to_server,        &mac_id_c2s))    return ERRORv(-1, c->Name(), ": negotiate c2s mac");
          if (!SSH::MAC   ::PreferenceIntersect(msg.mac_algorithms_server_to_client,        &mac_id_s2c))    return ERRORv(-1, c->Name(), ": negotiate s2c mac");
          guessed_right_s = kex_method == SSH::KEX::Id(Split(msg.kex_algorithms, iscomma)) && hostkey_type == SSH::Key::Id(Split(msg.server_host_key_algorithms, iscomma));
          guessed_right_c = kex_method == 1                                                && hostkey_type == 1;
          cipher_algo_c2s = SSH::Cipher::Algo(cipher_id_c2s, &encrypt_block_size);
          cipher_algo_s2c = SSH::Cipher::Algo(cipher_id_s2c, &decrypt_block_size);
          mac_algo_c2s = SSH::MAC::Algo(mac_id_c2s, &mac_prefix_c2s);
          mac_algo_s2c = SSH::MAC::Algo(mac_id_s2c, &mac_prefix_s2c);
          INFO(c->Name(), ": ssh negotiated { kex=", SSH::KEX::Name(kex_method), ", hostkey=", SSH::Key::Name(hostkey_type),
               cipher_algo_c2s == cipher_algo_s2c ? StrCat(", cipher=", Crypto::CipherAlgos::Name(cipher_algo_c2s)) : StrCat(", cipher_c2s=", Crypto::CipherAlgos::Name(cipher_algo_c2s), ", cipher_s2c=", Crypto::CipherAlgos::Name(cipher_algo_s2c)),
               mac_id_c2s      == mac_id_s2c      ? StrCat(", mac=",               SSH::MAC::Name(mac_id_c2s))      : StrCat(", mac_c2s=",               SSH::MAC::Name(mac_id_c2s),      ", mac_s2c=",               SSH::MAC::Name(mac_id_s2c)),
               " }");
          SSHTrace(c->Name(), ": block_size=", encrypt_block_size, ",", decrypt_block_size, " mac_len=", MAC_len_c, ",", MAC_len_s);

          if (SSH::KEX::EllipticCurveDiffieHellman(kex_method)) {
            switch (kex_method) {
              case SSH::KEX::ECDH_SHA2_NISTP256: curve_id=Crypto::EllipticCurve::NISTP256(); kex_hash=Crypto::DigestAlgos::SHA256(); break;
              case SSH::KEX::ECDH_SHA2_NISTP384: curve_id=Crypto::EllipticCurve::NISTP384(); kex_hash=Crypto::DigestAlgos::SHA384(); break;
              case SSH::KEX::ECDH_SHA2_NISTP521: curve_id=Crypto::EllipticCurve::NISTP521(); kex_hash=Crypto::DigestAlgos::SHA512(); break;
              default:                           return ERRORv(-1, c->Name(), ": ecdh curve");
            }
            if (!ecdh.GeneratePair(curve_id, ctx)) return ERRORv(-1, c->Name(), ": generate ecdh key");
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_ECDH_INIT(ecdh.c_text))) return ERRORv(-1, c->Name(), ": write");

          } else if (SSH::KEX::DiffieHellmanGroupExchange(kex_method)) {
            if      (kex_method == SSH::KEX::DHGEX_SHA1)   kex_hash = Crypto::DigestAlgos::SHA1();
            else if (kex_method == SSH::KEX::DHGEX_SHA256) kex_hash = Crypto::DigestAlgos::SHA256();
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_DH_GEX_REQUEST(dh.gex_min, dh.gex_max, dh.gex_pref)))
              return ERRORv(-1, c->Name(), ": write");

          } else if (SSH::KEX::DiffieHellman(kex_method)) {
            int secret_bits=0;
            if      (kex_method == SSH::KEX::DH14_SHA1) dh.p = Crypto::DiffieHellman::Group14Modulus(dh.g, dh.p, &secret_bits);
            else if (kex_method == SSH::KEX::DH1_SHA1)  dh.p = Crypto::DiffieHellman::Group1Modulus (dh.g, dh.p, &secret_bits);
            kex_hash = Crypto::DigestAlgos::SHA1();
            if (!dh.GeneratePair(secret_bits, ctx)) return ERRORv(-1, c->Name(), ": generate dh key");
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEXDH_INIT(dh.e))) return ERRORv(-1, c->Name(), ": write");

          } else return ERRORv(-1, c->Name(), "unkown kex method: ", kex_method);
        } break;

        case SSH::MSG_KEXDH_REPLY::ID:
        case SSH::MSG_KEX_DH_GEX_REPLY::ID: {
          if (state != FIRST_KEXREPLY && state != KEXREPLY) return ERRORv(-1, c->Name(), ": unexpected state ", state);
          if (guessed_s && !guessed_right_s && !(guessed_s=0)) { INFO(c->Name(), ": server guessed wrong, ignoring packet"); break; }
          if (packet_id == SSH::MSG_KEXDH_REPLY::ID && SSH::KEX::EllipticCurveDiffieHellman(kex_method)) {
            SSH::MSG_KEX_ECDH_REPLY msg; // MSG_KEX_ECDH_REPLY and MSG_KEXDH_REPLY share ID 31
            if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEX_ECDH_REPLY");
            SSHTrace(c->Name(), ": MSG_KEX_ECDH_REPLY");

            ecdh.s_text = msg.q_s.str();
            ECPointSetData(ecdh.g, ecdh.s, ecdh.s_text);
            if (!ecdh.ComputeSecret(&K, ctx)) return ERRORv(-1, c->Name(), ": ecdh");
            if ((v = ComputeExchangeHashAndVerifyHostKey(c, msg.k_s, msg.h_sig)) != 1) return ERRORv(-1, c->Name(), ": verify hostkey failed: ", v);
            // fall forward

          } else if (packet_id == SSH::MSG_KEXDH_REPLY::ID && SSH::KEX::DiffieHellmanGroupExchange(kex_method)) {
            SSH::MSG_KEX_DH_GEX_GROUP msg(dh.p, dh.g); // MSG_KEX_DH_GEX_GROUP and MSG_KEXDH_REPLY share ID 31
            if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEX_DH_GEX_GROUP");
            SSHTrace(c->Name(), ": MSG_KEX_DH_GEX_GROUP");

            if (!dh.GeneratePair(256, ctx)) return ERRORv(-1, c->Name(), ": generate dh_gex key");
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_DH_GEX_INIT(dh.e))) return ERRORv(-1, c->Name(), ": write");
            break;

          } else {
            SSH::MSG_KEXDH_REPLY msg(dh.f);
            if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEXDH_REPLY");
            SSHTrace(c->Name(), ": MSG_KEXDH_REPLY");
            if (!dh.ComputeSecret(&K, ctx)) return ERRORv(-1, c->Name(), ": dh");
            if ((v = ComputeExchangeHashAndVerifyHostKey(c, msg.k_s, msg.h_sig)) != 1) return ERRORv(-1, c->Name(), ": verify hostkey failed: ", v);
            // fall forward
          }

          state = state == FIRST_KEXREPLY ? FIRST_NEWKEYS : NEWKEYS;
          if (!WriteClearOrEncrypted(c, SSH::MSG_NEWKEYS())) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_NEWKEYS::ID: {
          if (state != FIRST_NEWKEYS && state != NEWKEYS) return ERRORv(-1, c->Name(), ": unexpected state ", state);
          SSHTrace(c->Name(), ": MSG_NEWKEYS");
          int key_len_c = Crypto::CipherAlgos::KeySize(cipher_algo_c2s), key_len_s = Crypto::CipherAlgos::KeySize(cipher_algo_s2c);
          if ((v = InitCipher(c, &encrypt, cipher_algo_c2s, DeriveKey(kex_hash, 'A', 24), DeriveKey(kex_hash, 'C', key_len_c), true))  != 1) return ERRORv(-1, c->Name(), ": init c->s cipher ", v, " keylen=", key_len_c);
          if ((v = InitCipher(c, &decrypt, cipher_algo_s2c, DeriveKey(kex_hash, 'B', 24), DeriveKey(kex_hash, 'D', key_len_s), false)) != 1) return ERRORv(-1, c->Name(), ": init s->c cipher ", v, " keylen=", key_len_s);
          if ((MAC_len_c = Crypto::MACAlgos::HashSize(mac_algo_c2s)) <= 0) return ERRORv(-1, c->Name(), ": invalid maclen ", encrypt_block_size);
          if ((MAC_len_s = Crypto::MACAlgos::HashSize(mac_algo_s2c)) <= 0) return ERRORv(-1, c->Name(), ": invalid maclen ", encrypt_block_size);
          integrity_c2s = DeriveKey(kex_hash, 'E', MAC_len_c);
          integrity_s2c = DeriveKey(kex_hash, 'F', MAC_len_s);
          state = NEWKEYS;
          if (!WriteCipher(c, SSH::MSG_SERVICE_REQUEST("ssh-userauth"))) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_SERVICE_ACCEPT::ID: {
          SSHTrace(c->Name(), ": MSG_SERVICE_ACCEPT");
          if (!WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "keyboard-interactive", "", "", "")))
            return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_USERAUTH_FAILURE::ID: {
          SSH::MSG_USERAUTH_FAILURE msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_USERAUTH_FAILURE");
          SSHTrace(c->Name(), ": MSG_USERAUTH_FAILURE: auth_left='", msg.auth_left.str(), "'");

          if (!loaded_pw) ClearPassword();
          if (!userauth_fail++) { cb(c, "Password:"); if ((password_prompts=1)) LoadPassword(c); }
          else return ERRORv(-1, c->Name(), ": authorization failed");
        } break;

        case SSH::MSG_USERAUTH_SUCCESS::ID: {
          SSHTrace(c->Name(), ": MSG_USERAUTH_SUCCESS");
          window_s = initial_window_size;
          if (!loaded_pw) { if (save_password_cb) save_password_cb(host, user, pw); ClearPassword(); }
          if (!WriteCipher(c, SSH::MSG_CHANNEL_OPEN("session", pty_channel.first, initial_window_size, max_packet_size)))
            return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_USERAUTH_INFO_REQUEST::ID: {
          SSH::MSG_USERAUTH_INFO_REQUEST msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_USERAUTH_INFO_REQUEST");
          SSHTrace(c->Name(), ": MSG_USERAUTH_INFO_REQUEST prompts=", msg.prompt.size());
          if (!msg.instruction.empty()) { SSHTrace(c->Name(), ": instruction: ", msg.instruction.str()); cb(c, msg.instruction); }
          for (auto &i : msg.prompt)    { SSHTrace(c->Name(), ": prompt: ",      i.text.str());          cb(c, i.text); }

          if ((password_prompts = msg.prompt.size())) LoadPassword(c);
          else if (!WriteCipher(c, SSH::MSG_USERAUTH_INFO_RESPONSE())) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_CHANNEL_OPEN_CONFIRMATION::ID: {
          SSH::MSG_CHANNEL_OPEN_CONFIRMATION msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_OPEN_CONFIRMATION");
          SSHTrace(c->Name(), ": MSG_CHANNEL_OPEN_CONFIRMATION");
          pty_channel.second = msg.sender_channel;
          window_c = msg.initial_win_size;

          if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST
                           (pty_channel.second, "pty-req", point(term_width, term_height),
                            point(term_width*8, term_height*12), "screen", "", true))) return ERRORv(-1, c->Name(), ": write");

          if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST(pty_channel.second, "shell", "", true))) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_CHANNEL_WINDOW_ADJUST::ID: {
          SSH::MSG_CHANNEL_WINDOW_ADJUST msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_WINDOW_ADJUST");
          SSHTrace(c->Name(), ": MSG_CHANNEL_WINDOW_ADJUST add ", msg.bytes_to_add, " to channel ", msg.recipient_channel);
          window_c += msg.bytes_to_add;
        } break;

        case SSH::MSG_CHANNEL_DATA::ID: {
          SSH::MSG_CHANNEL_DATA msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_DATA");
          SSHTrace(c->Name(), ": MSG_CHANNEL_DATA: channel ", msg.recipient_channel, ": ", msg.data.size(), " bytes");

          window_s -= (packet_len - packet_MAC_len - 4);
          if (window_s < initial_window_size / 2) {
            if (!WriteClearOrEncrypted(c, SSH::MSG_CHANNEL_WINDOW_ADJUST(pty_channel.second, initial_window_size)))
              return ERRORv(-1, c->Name(), ": write");
            window_s += initial_window_size;
          }

          cb(c, msg.data);
        } break;

        case SSH::MSG_CHANNEL_SUCCESS::ID: {
          SSHTrace(c->Name(), ": MSG_CHANNEL_SUCCESS");
        } break;

        case SSH::MSG_CHANNEL_FAILURE::ID: {
          SSHTrace(c->Name(), ": MSG_CHANNEL_FAILURE");
        } break;

        default: {
          ERROR(c->Name(), " unknown packet number ", (int)packet_id, " len ", packet_len);
        } break;
      }
      c->ReadFlush(packet_len);
      packet_len = 0;
    }
    return 0;
  }

  bool WriteKeyExchangeInit(Connection *c, bool guess) {
    string cipher_pref = SSH::Cipher::PreferenceCSV(), mac_pref = SSH::MAC::PreferenceCSV();
    KEXINIT_C = SSH::MSG_KEXINIT(RandBytes(16, rand_eng), SSH::KEX::PreferenceCSV(), SSH::Key::PreferenceCSV(),
                                 cipher_pref, cipher_pref, mac_pref, mac_pref, "none", "none",
                                 "", "", guess).ToString(rand_eng, 8, &sequence_number_c2s);
    SSHTrace(c->Name(), " wrote KEXINIT_C { kex=", SSH::KEX::PreferenceCSV(), " key=", SSH::Key::PreferenceCSV(),
             " cipher=", cipher_pref, " mac=", mac_pref, " }");
    return c->WriteFlush(KEXINIT_C) == KEXINIT_C.size();
  }
  int ComputeExchangeHashAndVerifyHostKey(Connection *c, const StringPiece &k_s, const StringPiece &h_sig) {
    H_text = SSH::ComputeExchangeHash(kex_method, kex_hash, V_C, V_S, KEXINIT_C, KEXINIT_S, k_s, K, &dh, &ecdh);
    if (state == FIRST_KEXREPLY) session_id = H_text;
    SSHTrace(c->Name(), ": H = \"", CHexEscape(H_text), "\"");
    return SSH::VerifyHostKey(H_text, hostkey_type, k_s, h_sig);
  }
  string DeriveKey(Crypto::DigestAlgo algo, char ID, int bytes) {
    return SSH::DeriveKey(algo, session_id, H_text, K, ID, bytes);
  }
  int InitCipher(Connection *c, Crypto::Cipher *cipher, Crypto::CipherAlgo algo, const string &IV, const string &key, bool dir) {
    SSHTrace(c->Name(), ": ", dir ? "C->S" : "S->C", " IV  = \"", CHexEscape(IV),  "\"");
    SSHTrace(c->Name(), ": ", dir ? "C->S" : "S->C", " key = \"", CHexEscape(key), "\"");
    Crypto::CipherFree(cipher);
    Crypto::CipherInit(cipher);
    return Crypto::CipherOpen(cipher, algo, dir, key, IV);
  }
  string ReadCipher(Connection *c, const StringPiece &m) {
    string dec_text(m.size(), 0);
    if (Crypto::CipherUpdate(&decrypt, m, &dec_text[0], dec_text.size()) != 1) return ERRORv("", c->Name(), ": decrypt failed");
    return dec_text;
  }
  int WriteCipher(Connection *c, const string &m) {
    string enc_text(m.size(), 0);
    if (Crypto::CipherUpdate(&encrypt, m, &enc_text[0], enc_text.size()) != 1) return ERRORv(-1, c->Name(), ": encrypt failed");
    enc_text += SSH::MAC(mac_algo_c2s, MAC_len_c, m, sequence_number_c2s-1, integrity_c2s, mac_prefix_c2s);
    return c->WriteFlush(enc_text) == m.size() + X_or_Y(mac_prefix_c2s, MAC_len_c);
  }
  bool WriteCipher(Connection *c, const SSH::Serializable &m) {
    string text = m.ToString(rand_eng, encrypt_block_size, &sequence_number_c2s);
    return WriteCipher(c, text);
  }
  bool WriteClearOrEncrypted(Connection *c, const SSH::Serializable &m) {
    if (state > FIRST_NEWKEYS) return WriteCipher(c, m);
    string text = m.ToString(rand_eng, encrypt_block_size, &sequence_number_c2s);
    return c->WriteFlush(text) == text.size();
  }
  int WriteChannelData(Connection *c, const StringPiece &b) {
    if (!password_prompts) {
      if (!WriteCipher(c, SSH::MSG_CHANNEL_DATA(pty_channel.second, b))) return ERRORv(-1, c->Name(), ": write");
      window_c -= (b.size() - 4);
    } else {
      bool cr = b.len && b.back() == '\r';
      pw.append(b.data(), b.size() - cr);
      if (cr && !WritePassword(c)) return ERRORv(-1, c->Name(), ": write");
    }
    return b.size();
  }
  bool WritePassword(Connection *c) {
    cb(c, "\r\n");
    bool success = false;
    if (userauth_fail) {
      success = WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "password", "", pw, ""));
    } else {
      vector<StringPiece> prompt(password_prompts);
      prompt.back() = StringPiece(pw.data(), pw.size());
      success = WriteCipher(c, SSH::MSG_USERAUTH_INFO_RESPONSE(prompt));
    }
    password_prompts = 0;
    return success;
  }
  void LoadPassword(Connection *c) {
    if ((loaded_pw = load_password_cb && load_password_cb(host, user, &pw))) WritePassword(c);
    if (loaded_pw) ClearPassword();
  }
  void ClearPassword() { pw.assign(pw.size(), ' '); pw.clear(); }
  void SetPasswordCB(const Vault::LoadPasswordCB &L, const Vault::SavePasswordCB &S) { load_password_cb=L; save_password_cb=S; }
  int SetTerminalWindowSize(Connection *c, int w, int h) {
    term_width = w;
    term_height = h;
    if (!c || c->state != Connection::Connected || state <= FIRST_NEWKEYS || pty_channel.second < 0) return 0;
    if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST(pty_channel.second, "window-change", point(term_width, term_height),
                                                 point(term_width*8, term_height*12),
                                                 "", "", false))) return ERRORv(-1, c->Name(), ": write");
    return 0;
  }
};

Connection *SSHClient::Open(const string &hostport, const SSHClient::ResponseCB &cb, Callback *detach) { 
  Connection *c = Connect(hostport, 22, detach);
  if (!c) return 0;
  c->handler = new SSHClientConnection(cb, hostport);
  return c;
}
int  SSHClient::WriteChannelData     (Connection *c, const StringPiece &b)                            { return dynamic_cast<SSHClientConnection*>(c->handler)->WriteChannelData(c, b); }
int  SSHClient::SetTerminalWindowSize(Connection *c, int w, int h)                                    { return dynamic_cast<SSHClientConnection*>(c->handler)->SetTerminalWindowSize(c, w, h); }
void SSHClient::SetUser              (Connection *c, const string &user)                                     { dynamic_cast<SSHClientConnection*>(c->handler)->user = user; }
void SSHClient::SetPasswordCB(Connection *c, const Vault::LoadPasswordCB &L, const Vault::SavePasswordCB &S) { dynamic_cast<SSHClientConnection*>(c->handler)->SetPasswordCB(L, S); }

struct SMTPClientConnection : public Connection::Handler {
  enum { INIT=0, SENT_HELO=1, READY=2, MAIL_FROM=3, RCPT_TO=4, SENT_DATA=5, SENDING=6, RESETING=7, QUITING=8 };
  SMTPClient *server;
  SMTPClient::DeliverableCB deliverable_cb;
  SMTPClient::DeliveredCB delivered_cb;
  int state=0, rcpt_index=0;
  string greeting, helo_domain, ehlo_response, response_lines;
  SMTP::Message mail;
  string RcptTo(int index) const { return StrCat("RCPT TO: <", mail.rcpt_to[index], ">\r\n"); }
  SMTPClientConnection(SMTPClient *S, SMTPClient::DeliverableCB CB1, SMTPClient::DeliveredCB CB2)
    : server(S), deliverable_cb(CB1), delivered_cb(CB2) {}

  int Connected(Connection *c) { helo_domain = server->HeloDomain(c->src_addr); return 0; }

  void Close(Connection *c) {
    server->total_disconnected++;
    if (DeliveringState(state)) delivered_cb(0, mail, 0, "");
    deliverable_cb(c, helo_domain, 0);
  }

  int Read(Connection *c) {
    int processed = 0;
    StringLineIter lines(c->rb.buf, StringLineIter::Flag::BlankLines);
    for (string line = IterNextString(&lines); !lines.Done(); line = IterNextString(&lines)) {
      processed = lines.next_offset;
      if (!response_lines.empty()) response_lines.append("\r\n");
      response_lines.append(line);

      const char *dash = FindChar(line.c_str(), notnum);
      bool multiline = dash && *dash == '-';
      if (multiline) continue;

      int code = atoi(line), need_code=0; string response;
      if      (state == INIT)        { response=StrCat("EHLO ", helo_domain, "\r\n"); greeting=response_lines; }
      else if (state == SENT_HELO)   { need_code=250; ehlo_response=response_lines; }
      else if (state == READY)       { ERROR("read unexpected line: ", response_lines); return -1; }
      else if (state == MAIL_FROM)   { need_code=250; response=RcptTo(rcpt_index++); }
      else if (state == RCPT_TO)     { need_code=250; response="DATA\r\n";
        if (rcpt_index < mail.rcpt_to.size()) { response=RcptTo(rcpt_index++); state--; }
      }
      else if (state == SENT_DATA)   { need_code=354; response=StrCat(mail.content, "\r\n.\r\n"); }
      else if (state == SENDING)     { delivered_cb(c, mail, code, response_lines); server->delivered++; state=READY-1; }
      else if (state == RESETING)    { need_code=250; state=READY-1; }
      else if (state == QUITING)     { /**/ }
      else { ERROR("unknown state ", state); return -1; }

      if (need_code && code != need_code) {
        ERROR(StateString(state), " failed: ", response_lines);
        if (state == SENT_HELO || state == RESETING || state == QUITING) return -1;

        if (DeliveringState(state)) delivered_cb(c, mail, code, response_lines);
        response="RSET\r\n"; server->failed++; state=RESETING-1;
      }
      if (!response.empty()) if (c->WriteFlush(response) != response.size()) return -1;

      response_lines.clear();
      state++;
    }
    c->ReadFlush(processed);

    if (state == READY) {
      mail.clear();
      rcpt_index = 0;
      if (deliverable_cb(c, helo_domain, &mail)) Deliver(c);
    }
    return 0;
  }

  void Deliver(Connection *c) {
    string response;
    if (!mail.mail_from.size() || !mail.rcpt_to.size() || !mail.content.size()) { response="QUIT\r\n"; state=QUITING; }
    else { response=StrCat("MAIL FROM: <", mail.mail_from, ">\r\n"); state++; }
    if (c->WriteFlush(response) != response.size()) c->SetError();
  }

  static const char *StateString(int n) {
    static const char *s[] = { "INIT", "SENT_HELO", "READY", "MAIL_FROM", "RCPT_TO", "SENT_DATA", "SENDING", "RESETING", "QUITTING" };
    return (n >= 0 && n < sizeofarray(s)) ? s[n] : "";
  }
  static bool DeliveringState(int state) { return (state >= MAIL_FROM && state <= SENDING); }
};

Connection *SMTPClient::DeliverTo(IPV4::Addr ipv4_addr, IPV4EndpointSource *src_pool,
                                  DeliverableCB deliverable_cb, DeliveredCB delivered_cb) {
  static const int tcp_port = 25;
  Connection *c = Connect(ipv4_addr, tcp_port, src_pool);
  if (!c) return 0;

  c->handler = new SMTPClientConnection(this, deliverable_cb, delivered_cb);
  return c;
}

void SMTPClient::DeliverDeferred(Connection *c) { ((SMTPClientConnection*)c->handler)->Deliver(c); }

struct SMTPServerConnection : public Connection::Handler {
  SMTPServer *server;
  string my_domain, client_domain;
  SMTP::Message message;
  bool in_data;

  SMTPServerConnection(SMTPServer *s) : server(s), in_data(0) {}
  void ClearStateTable() { message.mail_from.clear(); message.rcpt_to.clear(); message.content.clear(); in_data=0; }

  int Connected(Connection *c) {
    my_domain = server->HeloDomain(c->src_addr);
    string greeting = StrCat("220 ", my_domain, " Simple Mail Transfer Service Ready\r\n");
    return (c->Write(greeting) == greeting.size()) ? 0 : -1;
  }
  int Read(Connection *c) {
    int offset = 0, processed;
    while (c->state == Connection::Connected) {
      bool last_in_data = in_data;
      if (in_data) { if ((processed = ReadData    (c, c->rb.begin()+offset, c->rb.size()-offset)) < 0) return -1; }
      else         { if ((processed = ReadCommands(c, c->rb.begin()+offset, c->rb.size()-offset)) < 0) return -1; }
      offset += processed;
      if (last_in_data == in_data) break;
    }
    if (offset) c->ReadFlush(offset);
    return 0;
  }

  int ReadCommands(Connection *c, const char *in, int len) {
    int processed = 0;
    StringLineIter lines(StringPiece(in, len), StringLineIter::Flag::BlankLines);
    for (const char *line = lines.Next(); line && lines.next_offset>=0 && !in_data; line = lines.Next()) {
      processed = lines.next_offset;
      StringWordIter words(line, lines.cur_len, isint3<' ', '\t', ':'>);
      string cmd = toupper(IterNextString(&words));
      string a1_orig = IterNextString(&words);
      string a1 = toupper(a1_orig), response="500 unrecognized command\r\n";

      if (cmd == "MAIL" && a1 == "FROM") {
        ClearStateTable();
        message.mail_from = IterRemainingString(&words);
        response = "250 OK\r\n";
      }
      else if (cmd == "RCPT" && a1 == "TO") {
        message.rcpt_to.push_back(IterRemainingString(&words));
        response="250 OK\r\n"; }
      else if (cmd == "DATA") {
        if      (!message.rcpt_to.size())   response = "503 valid RCPT command must precede DATA\r\n";
        else if (!message.mail_from.size()) response = "503 valid FROM command must precede DATA\r\n";
        else                 { in_data = 1; response = "354 Start mail input; end with <CRLF>.<CRLF>\r\n"; }
      }
      else if (cmd == "EHLO") { response=StrCat("250 ", server->domain, " greets ", a1_orig, "\r\n"); client_domain=a1_orig; }
      else if (cmd == "HELO") { response=StrCat("250 ", server->domain, " greets ", a1_orig, "\r\n"); client_domain=a1_orig; }
      else if (cmd == "RSET") { response="250 OK\r\n"; ClearStateTable(); }
      else if (cmd == "NOOP") { response="250 OK\r\n"; }
      else if (cmd == "VRFY") { response="250 OK\r\n"; }
      else if (cmd == "QUIT") { c->WriteFlush(StrCat("221 ", server->domain, " closing connection\r\n")); c->SetError(); }
      if (!response.empty()) if (c->Write(response) != response.size()) return -1;
    }
    return processed;
  }

  int ReadData(Connection *c, const char *in, int len) {
    int processed = 0;
    StringLineIter lines(StringPiece(in, len), StringLineIter::Flag::BlankLines);
    for (const char *line = lines.Next(); line && lines.next_offset>=0; line = lines.Next()) {
      processed = lines.next_offset;
      if (lines.cur_len == 1 && *line == '.') { in_data=0; break; }
      message.content.append(line, lines.cur_len);
      message.content.append("\r\n");
    }
    if (!in_data) {
      c->Write("250 OK\r\n");
      server->ReceiveMail(c, message); 
      ClearStateTable();
    }
    return processed;
  }
};

int SMTPServer::Connected(Connection *c) { total_connected++; c->handler = new SMTPServerConnection(this); return 0; }

void SMTPServer::ReceiveMail(Connection *c, const SMTP::Message &mail) {
  INFO("SMTPServer::ReceiveMail FROM=", mail.mail_from, ", TO=", mail.rcpt_to, ", content=", mail.content);
}

/* GPlusClient */

struct GPlusClientHandler {
  struct PersistentConnection : public Connection::Handler {
    UDPClient::ResponseCB responseCB;
    UDPClient::HeartbeatCB heartbeatCB;
    PersistentConnection(UDPClient::ResponseCB RCB, UDPClient::HeartbeatCB HCB) : responseCB(RCB), heartbeatCB(HCB) {}

    int Heartbeat(Connection *c) { if (heartbeatCB) heartbeatCB(c); return 0; }
    void Close(Connection *c) { if (responseCB) responseCB(c, 0, 0); }
    int Read(Connection *c) {
      for (int i=0; i<c->packets.size() && responseCB; i++) {
        if (c->state != Connection::Connected) break;
        responseCB(c, c->rb.begin() + c->packets[i].offset, c->packets[i].len);
      }
      return 0;
    }
  };
};

Connection *GPlusClient::PersistentConnection(const string &name, UDPClient::ResponseCB responseCB, UDPClient::HeartbeatCB HCB) {
  Connection *c = EndpointConnect(name);
  c->handler = new GPlusClientHandler::PersistentConnection(responseCB, HCB);
  return c;
}

/* Sniffer */

#ifdef LFL_PCAP
void Sniffer::GetDeviceAddressSet(set<IPV4::Addr> *out) {
  static IPV4::Addr localhost = IPV4::Parse("127.0.0.1");
  char errbuf[PCAP_ERRBUF_SIZE]; pcap_if_t *devs = 0; int ret;
  if ((ret = pcap_findalldevs(&devs, errbuf))) FATAL("pcap_findalldevs: ", ret);
  for (pcap_if_t *d = devs; d; d = d->next) {
    for (pcap_addr *a = d->addresses; a; a = a->next) {
      struct sockaddr_in *sin = (struct sockaddr_in *)a->addr;
      if (sin->sin_family != PF_INET || sin->sin_addr.s_addr == localhost) continue;
      out->insert(sin->sin_addr.s_addr);
    }
  }
}

void Sniffer::PrintDevices(vector<string> *out) {
  char errbuf[PCAP_ERRBUF_SIZE]; pcap_if_t *devs = 0; int ret;
  if ((ret = pcap_findalldevs(&devs, errbuf))) FATAL("pcap_findalldevs: ", ret);
  for (pcap_if_t *d = devs; d; d = d->next) {
    if (out) out->push_back(d->name);
    INFO(ret++, ": ", d->name, " : ", d->description ? d->description : "(none)");
  }
}

void Sniffer::Threadproc() {
  pcap_pkthdr *pkthdr; const unsigned char *packet; int ret;
  while (Running() && (ret = pcap_next_ex((pcap_t*)handle, &pkthdr, &packet)) >= 0) {
    if (!ret) continue;
    cb((const char *)packet, pkthdr->caplen, pkthdr->len);
  }
}

Sniffer *Sniffer::Open(const string &dev, const string &filter, int snaplen, CB cb) {
  char errbuf[PCAP_ERRBUF_SIZE];
  bpf_u_int32 ip, mask, ret;
  pcap_t *handle;
  if (pcap_lookupnet(dev.c_str(), &ip, &mask, errbuf)) return ERRORv(nullptr, "no netmask for ", dev);
  if (!(handle = pcap_open_live(dev.c_str(), snaplen, 1, 1000, errbuf))) return ERRORv(nullptr, "open failed: ", dev, ": ", errbuf);
  if (filter.size()) {
    bpf_program fp;
    if (pcap_compile(handle, &fp, filter.c_str(), 0, ip)) return ERRORv(nullptr, "parse filter: ", filter, ": ", pcap_geterr(handle));
    if (pcap_setfilter(handle, &fp)) return ERRORv(nullptr, "install filter: ", filter, ": ", pcap_geterr(handle));
  }
  Sniffer *sniffer = new Sniffer(handle, ip, mask, cb);
  sniffer->thread.Open(bind(&Sniffer::Threadproc, sniffer));
  sniffer->thread.Start();
  return sniffer;
}
#else /* LFL_PCAP */
void Sniffer::GetDeviceAddressSet(set<IPV4::Addr> *out) {}
void Sniffer::PrintDevices(vector<string> *out) {}
void Sniffer::Threadproc() {}
Sniffer *Sniffer::Open(const string &dev, const string &filter, int snaplen, CB cb) { return ERRORv(nullptr, "sniffer not implemented"); }
#endif /* LFL_PCAP */
void Sniffer::GetIPAddress(IPV4::Addr *out) {
  static IPV4::Addr localhost = IPV4::Parse("127.0.0.1");
  *out = 0;
#if defined(_WIN32)
#elif defined(LFL_ANDROID)
  *out = ntohl(AndroidIPV4Address());
#else
  ifaddrs* ifap = NULL;
  int r = getifaddrs(&ifap);
  if (r) return ERROR("getifaddrs ", r);
  for (ifaddrs *i = ifap; i; i = i->ifa_next) {
    if (!i->ifa_dstaddr || i->ifa_dstaddr->sa_family != AF_INET) continue;
    IPV4::Addr addr = ((struct sockaddr_in*)i->ifa_addr)->sin_addr.s_addr;
    if (addr == localhost) continue;
    *out = addr;
    break;
  }
#endif
}
void Sniffer::GetBroadcastAddress(IPV4::Addr *out) {
  static IPV4::Addr localhost = IPV4::Parse("127.0.0.1");
  *out = 0;
#if defined(_WIN32)
#elif defined(LFL_ANDROID)
  *out = ntohl(AndroidIPV4BroadcastAddress());
#else
  ifaddrs* ifap = NULL;
  int r = getifaddrs(&ifap);
  if (r) return ERROR("getifaddrs ", r);
  for (ifaddrs *i = ifap; i; i = i->ifa_next) {
    if (!i->ifa_dstaddr || i->ifa_dstaddr->sa_family != AF_INET) continue;
    IPV4::Addr addr = ((struct sockaddr_in*)i->ifa_dstaddr)->sin_addr.s_addr;
    if (addr == localhost) continue;
    *out = addr;
    break;
  }
#endif
}

#ifdef LFL_GEOIP
GeoResolution *GeoResolution::Open(const string &db) {
  void *impl = GeoIP_open(db.c_str(), GEOIP_INDEX_CACHE);
  if (!impl) return 0;
  return new GeoResolution(impl);
}

bool GeoResolution::resolve(const string &addr, string *country, string *region, string *city, float *lat, float *lng) {
  GeoIPRecord *gir = GeoIP_record_by_name((GeoIP*)impl, addr.c_str());
  if (!gir) return false;
  if (country) *country = gir->country_code ? gir->country_code : "";
  if (region) *region = gir->region ? gir->region : "";
  if (city) *city = gir->city ? gir->city : "";
  if (lat) *lat = gir->latitude;
  if (lng) *lng = gir->longitude;
  GeoIPRecord_delete(gir);
  return true;
}
#else
GeoResolution *GeoResolution::Open(const string &db) { return 0; }
bool GeoResolution::resolve(const string &addr, string *country, string *region, string *city, float *lat, float *lng) { FATAL("not implemented"); }
#endif

}; // namespace LFL
