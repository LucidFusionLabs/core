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
#ifdef LFL_PCAP
#include "pcap/pcap.h"
#endif
};

#include "core/app/crypto.h"
#include "core/app/net/resolver.h"

#ifndef LFL_WINDOWS
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
DEFINE_bool(dns_dump,       0,  "Print DNS responses");
DEFINE_bool(network_debug,  0,  "Print send()/recv() bytes");
DEFINE_int (udp_idle_sec,   15, "Timeout UDP connections idle for seconds");

#ifdef LFL_WINDOWS
const Socket InvalidSocket = INVALID_SOCKET;
#else
const Socket InvalidSocket = -1;
#endif

const int SocketType::Stream    = SOCK_STREAM;
const int SocketType::Datagram  = SOCK_DGRAM; 
const int SocketType::SeqPacket = SOCK_SEQPACKET;
const int SocketType::Raw       = SOCK_RAW;

const char *Protocol::Name(int p) {
  switch (p) {
    case TCP:       return "TCP";
    case UDP:       return "UDP";
    case UNIX:      return "UNIX";
    case GPLUS:     return "GPLUS";
    case InProcess: return "InProcess";
    default:        return "";
  }
}

const IPV4::Addr IPV4::ANY = INADDR_ANY;

IPV4::Addr IPV4::Parse(const string &ip) { return inet_addr(ip.c_str()); }

void IPV4::ParseCSV(const string &text, vector<IPV4::Addr> *out) {
  vector<string> addrs; IPV4::Addr addr;
  Split(text, iscomma, &addrs);
  for (int i = 0; i < addrs.size(); i++) {
    if ((addr = Parse(addrs[i])) == INADDR_NONE) FATAL("unknown addr ", addrs[i]);
    out->push_back(addr);
  }
}

void IPV4::ParseCSV(const string &text, set<IPV4::Addr> *out) {
  vector<string> addrs; IPV4::Addr addr;
  Split(text, iscomma, &addrs);
  for (int i = 0; i < addrs.size(); i++) {
    if ((addr = Parse(addrs[i])) == INADDR_NONE) FATAL("unknown addr ", addrs[i]);
    out->insert(addr);
  }
}

string IPV4::MakeCSV(const vector<IPV4::Addr> &in) {
  string ret;
  for (vector<Addr>::const_iterator i = in.begin(); i != in.end(); ++i) StrAppend(&ret, ret.size()?",":"", IPV4::Text(*i));
  return ret;
}

string IPV4::MakeCSV(const set<IPV4::Addr> &in) {
  string ret;
  for (set<Addr>::const_iterator i = in.begin(); i != in.end(); ++i) StrAppend(&ret, ret.size()?",":"", IPV4::Text(*i));
  return ret;
}

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
#ifdef LFL_WINDOWS
  closesocket(fd);
#else
  close(fd);
#endif
}

Socket SystemNetwork::OpenSocket(int protocol) {
  Socket ret = InvalidSocket;
  if (protocol == Protocol::TCP) ret = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  else if (protocol == Protocol::UDP) {
    if ((ret = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) return InvalidSocket;
#ifdef LFL_WINDOWS
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR,12)
#endif
    DWORD dwBytesReturned = 0;
    BOOL bNewBehavior = FALSE;
    DWORD status = WSAIoctl(ret, SIO_UDP_CONNRESET, &bNewBehavior, sizeof(bNewBehavior), 0, 0, &dwBytesReturned, 0, 0);
#endif
  }
  return ret;
}

bool SystemNetwork::OpenSocketPair(Socket *fd, int socket_type, bool close_on_exec) {
#ifdef LFL_WINDOWS
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
#else // LFL_WINDOWS
  CHECK(!socketpair(PF_LOCAL, socket_type, 0, fd));
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
#ifdef LFL_WINDOWS
  u_long ioctlarg = !blocking ? 1 : 0;
  if (ioctlsocket(fd, FIONBIO, &ioctlarg) < 0) return -1;
#else
  if (fcntl(fd, F_SETFL, !blocking ? O_NONBLOCK : 0) == -1) return -1;
#endif
  return 0;
}

int SystemNetwork::SetSocketCloseOnExec(Socket fd, int close) {
#ifdef LFL_WINDOWS
#else
  if (fcntl(fd, F_SETFD, close ? FD_CLOEXEC : 0) == -1) return -1;
#endif
  return 0;
}

int SystemNetwork::SetSocketBroadcastEnabled(Socket fd, int optval) {
  if (setsockopt(fd, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<const char*>(&optval), sizeof(optval)))
    return ERRORv(-1, "setsockopt: ", SystemNetwork::LastError());
  return 0;
}

int SystemNetwork::SetSocketBufferSize(Socket fd, bool send_or_recv, int optval) {
  if (setsockopt(fd, SOL_SOCKET, send_or_recv ? SO_SNDBUF : SO_RCVBUF, reinterpret_cast<const char*>(&optval), sizeof(optval)))
    return ERRORv(-1, "setsockopt: ", SystemNetwork::LastError());
  return 0;
}

int SystemNetwork::GetSocketBufferSize(Socket fd, bool send_or_recv) {
  int res=0;
  socklen_t resSize=sizeof(res);
  if (getsockopt(fd, SOL_SOCKET, send_or_recv ? SO_SNDBUF : SO_RCVBUF, reinterpret_cast<char*>(&res), &resSize))
    return ERRORv(-1, "getsockopt: ", SystemNetwork::LastError());
  return res;
}

int SystemNetwork::Bind(int fd, IPV4::Addr addr, int port) {
#ifndef LFL_EMSCRIPTEN
  int optval = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&optval), sizeof(optval)))
    return ERRORv(-1, "setsockopt: ", SystemNetwork::LastError());
#endif

  sockaddr_in sin;
  memset(&sin, 0, sizeof(sockaddr_in));
  sin.sin_family = PF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = addr ? addr : INADDR_ANY;

  if (FLAGS_network_debug) INFO("bind(", fd, ", ", IPV4::Text(addr, port), ")");
  if (::bind(fd, reinterpret_cast<const sockaddr*>(&sin), socklen_t(sizeof(sockaddr_in))) == -1)
    return ERRORv(-1, "bind: ", SystemNetwork::LastError());

  return 0;
}

Socket SystemNetwork::Accept(Socket listener, IPV4::Addr *addr, int *port, bool blocking) {
  struct sockaddr_in sin;
  socklen_t sinSize = sizeof(sin);
  Socket socket = ::accept(listener, reinterpret_cast<struct sockaddr*>(&sin), &sinSize);
  if (socket == -1 && !SystemNetwork::EWouldBlock()) return ERRORv(InvalidSocket, "accept: ", SystemNetwork::LastError());
  if (addr) *addr = sin.sin_addr.s_addr;
  if (port) *port = ntohs(sin.sin_port);
  if (!blocking && SetSocketBlocking(socket, 0))
  { ERROR("Network::socket_blocking: ", SystemNetwork::LastError()); CloseSocket(socket); return InvalidSocket; }
  return socket;
}

Socket SystemNetwork::Listen(int protocol, IPV4::Addr addr, int port, int backlog, bool blocking) {
  Socket fd;
  if ((fd = OpenSocket(protocol)) < 0) 
    return ERRORv(InvalidSocket, "network_socket_open: ", SystemNetwork::LastError());

  if (Bind(fd, addr, port) == -1) { CloseSocket(fd); return InvalidSocket; }

  if (protocol == Protocol::TCP) {
    if (::listen(fd, backlog) == -1)
    { ERROR("listen: ", SystemNetwork::LastError()); CloseSocket(fd); return InvalidSocket; }
  }

  if (!blocking && SetSocketBlocking(fd, 0))
  { ERROR("Network::socket_blocking: ", SystemNetwork::LastError()); CloseSocket(fd); return InvalidSocket; }

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
  int ret = ::connect(fd, reinterpret_cast<struct sockaddr*>(&sin), sizeof(struct sockaddr_in));
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
  return ::sendto(fd, buf, len, 0, reinterpret_cast<struct sockaddr*>(&sin), sinSize);
}

int SystemNetwork::GetPeerName(Socket fd, IPV4::Addr *addr_out, int *port_out) {
  struct sockaddr_in sin;
  socklen_t sinSize=sizeof(sin);
  if (::getpeername(fd, reinterpret_cast<struct sockaddr*>(&sin), &sinSize) < 0)
    return ERRORv(-1, "getpeername: ", strerror(errno));
  *addr_out = sin.sin_addr.s_addr;
  *port_out = ntohs(sin.sin_port);
  return 0;
}

int SystemNetwork::GetSockName(Socket fd, IPV4::Addr *addr_out, int *port_out) {
  struct sockaddr_in sin;
  socklen_t sinSize=sizeof(sin);
  if (::getsockname(fd, reinterpret_cast<struct sockaddr*>(&sin), &sinSize) < 0)
    return ERRORv(-1, "getsockname: ", strerror(errno));
  *addr_out = sin.sin_addr.s_addr;
  *port_out = ntohs(sin.sin_port);
  return 0;
}

string SystemNetwork::GetHostByAddr(IPV4::Addr addr) {
#if defined(LFL_WINDOWS) || defined(LFL_ANDROID)
  struct hostent *h = ::gethostbyaddr(reinterpret_cast<const char*>(&addr), sizeof(addr), PF_INET);
#else
  struct hostent *h = ::gethostbyaddr(reinterpret_cast<const void*>(&addr), sizeof(addr), PF_INET);
#endif
  return h ? h->h_name : "";
}

IPV4::Addr SystemNetwork::GetHostByName(const string &host) {
  in_addr a;
  if ((a.s_addr = IPV4::Parse(host)) != INADDR_NONE) return int(a.s_addr);

  hostent *h = gethostbyname(host.c_str());
  if (h && h->h_length == 4) return *reinterpret_cast<const int*>(h->h_addr_list[0]);

  ERROR("SystemNetwork::GetHostByName ", host);
  return -1;
}

int SystemNetwork::IOVLen(const iovec *iov, int len) {
  int ret = 0;
  if (iov) for (int i=0; i<len; i++) ret += iov[i].iov_len;
  return ret;
}

bool SystemNetwork::EWouldBlock() {
#ifdef LFL_WINDOWS
  return WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == WSAEINPROGRESS;
#else
  return errno == EAGAIN || errno == EINPROGRESS;
#endif
};

string SystemNetwork::LastError() {
#ifdef LFL_WINDOWS
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
  if ((select(maxfd+1, rc?&rfds:0, wc?&wfds:0, xc?&xfds:0, wait_time >= 0 ? &tv : 0)) == -1)
    return ERRORv(-1, "select: ", SystemNetwork::LastError(), " maxfd=", maxfd, " ", DebugString());
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
      if (app->run) app->scheduler.Wakeup(app->focused);
    }
    if (wait_mutex) { ScopedMutex sm(*wait_mutex); }
  }
}

/* Connection */

Connection::~Connection() {}

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

int Connection::ReadFlush(int len) {
  if (len<0) return -1;
  if (!len) return rb.size();
  if (rb.size()-len < 0) return ERRORv(-1, Name(), ": read queue underflow: ", len, " > ", rb.size());
  rb.Flush(len);
  rb.EnsureZeroTerminated();
  if (control_messages) for (auto &s : transferred_socket) { s.offset -= len; CHECK_GE(s.offset, 0); }
  return rb.size();
}

int Connection::WriteFlush(const char *buf, int len, int transfer_socket) {
  struct iovec iov = { Void(const_cast<char*>(buf)), size_t(len) };
  return WriteVFlush(&iov, 1, transfer_socket);
}

int Connection::WriteFlush() {
  int wrote = WriteFlush(wb.begin(), wb.size());
  if (wrote) wb.Flush(wrote);
  wb.EnsureZeroTerminated();
  return wrote;
}

/* SocketConnection */

void SocketConnection::Close() {
  app->net->ConnCloseDetached(app->net->tcp_client.get(), this);
}

int SocketConnection::Read() {
  int readlen = rb.Remaining(), len = 0;
  if (readlen <= 0) return ERRORv(-1, Name(), ": read queue full, rl=", rb.size());

  if (bio.ssl) {
    if ((len = bio.Read(rb.end(), readlen)) < 0) {
      string err_string = bio.ErrorString();
      return ERRORv(-1, Name(), ": BIO_read: ", err_string.size() ? err_string : "read() zero");
    } else if (!len) return 0;

#ifndef LFL_WINDOWS
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

  } else {
#ifdef LFL_WINDOWS
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

int SocketConnection::ReadPackets() {
  int ret=1;
  while (ret > 0) {
    ret = ReadPacket();
    if (ret < 0 && !SystemNetwork::EWouldBlock()) return ret;
  }
  return 0;
}

int SocketConnection::Write(const char *buf, int len) {
  if (!buf || len<0) return -1;
  if (!len) len = strlen(buf);
  if (wb.size() + len > wb.Capacity()) return ERRORv(-1, Name(), ": write queue full");

  if (!wb.size() && len) {
    writable = true;
    app->net->UpdateActive(this);
  }
  wb.Add(buf, len);
  wb.EnsureZeroTerminated();
  wt = Now();
  return len;
}

int SocketConnection::WriteFlush(const char *buf, int len) {
  CHECK_GE(len, 0);
  int wrote = 0;
  if (bio.ssl) {
    if ((wrote = bio.Write(StringPiece(buf, len))) < 0) {
      if (!SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": send: ", strerror(errno));
      wrote = 0;
    }
  }
  else {
    if ((wrote = send(socket, buf, len, 0)) < 0) {
      if (!SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": send: ", strerror(errno));
      wrote = 0;
    }
  }
  if (FLAGS_network_debug && wrote) INFO("write(", socket, ", ", wrote, ", '", string(buf, wrote), "')");
  return wrote;
}

int SocketConnection::WriteVFlush(const iovec *iov, int len) {
  int wrote = 0;
  if (bio.ssl) { FATAL("ssl writev unimplemented"); }
  else {
#ifdef LFL_WINDOWS
    DWORD sent = 0;
    vector<WSABUF> buf;
    for (const iovec *i = iov, *e = i + len; i != e; ++i) buf.push_back({ i->iov_len, static_cast<char*>(i->iov_base) });
    if (WSASend(socket, &buf[0], buf.size(), &sent, 0, NULL, NULL)) {
      if (!SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": WSASend: ", strerror(errno));
      sent = 0;
    }
    wrote = sent;
#else
    if ((wrote = writev(socket, iov, len)) < 0) {
      if (!SystemNetwork::EWouldBlock()) return ERRORv(-1, Name(), ": send: ", strerror(errno));
      wrote = 0;
    }
#endif
  }
  if (FLAGS_network_debug) INFO("writev(", socket, ", ", wrote, ", '", len, "')");
  return wrote;
}

int SocketConnection::WriteVFlush(const iovec *iov, int len, int transfer_socket) {
  int wrote = 0;
#if defined(LFL_WINDOWS) || defined(LFL_MOBILE)
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

int SocketConnection::SendTo(const char *buf, int len) { return SystemNetwork::SendTo(socket, addr, port, buf, len); }
void SocketConnection::RemoveFromMainWait(Window *w) { app->scheduler.DelMainWaitSocket(w, socket); }
void SocketConnection::AddToMainWait(Window *w, function<bool()> readable_cb) {
  app->scheduler.AddMainWaitSocket(w, socket, SocketSet::READABLE, move(readable_cb));
}

/* SocketService */

void SocketService::Close(SocketConnection *c) {
  app->net->active.Del(c->socket);
  if (c->detach) conn[c->socket].release();
  else SystemNetwork::CloseSocket(c->socket);
  if (connect_src_pool && (c->src_addr || c->src_port)) connect_src_pool->Close(c->src_addr, c->src_port);
}

int SocketService::OpenSocket(SocketConnection *c, int protocol, int blocking, IPV4EndpointSource* src_pool) {
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

Socket SocketService::Listen(IPV4::Addr addr, int port, SocketListener *listener) {
  Socket fd = -1;
  if (listener->ssl) {
    if ((listener->socket = listener->bio.Listen(port, true)) == InvalidSocket)
      return ERRORv(-1, "ssl_listen: ", -1);
  } else {
    if ((listener->socket = SystemNetwork::Listen(protocol, addr, port)) == InvalidSocket)
      return ERRORv(-1, "SystemNetwork::Listen(", protocol, ", ", port, "): ", SystemNetwork::LastError());
  }
  app->net->active.Add(listener->socket, SocketSet::READABLE, &listener->self_reference);
  return listener->socket;
}

SocketConnection *SocketService::Accept(int state, Socket socket, IPV4::Addr addr, int port) {
  auto c = (conn[socket] = make_unique<SocketConnection>(this, state, socket, addr, port)).get();
  app->net->active.Add(c->socket, SocketSet::READABLE, &c->self_reference);
  return c;
}

SocketConnection *SocketService::Connect(IPV4::Addr addr, int port, IPV4::Addr src_addr, int src_port, Callback *detach) {
  SingleIPV4Endpoint src_pool(src_addr, src_port);
  return Connect(addr, port, &src_pool, detach);
}

SocketConnection *SocketService::Connect(IPV4::Addr addr, int port, IPV4EndpointSource *src_pool, Callback *detach) {
  auto c = new SocketConnection(this, Connection::Connecting, addr, port, detach);
  if (SocketService::OpenSocket(c, protocol, 0, src_pool ? src_pool : connect_src_pool))
  { ERROR(c->Name(), ": connecting: ", SystemNetwork::LastError()); delete c; return 0; }

  int connected = 0;
  if (SystemNetwork::Connect(c->socket, c->addr, c->port, &connected) == -1) {
    ERROR(c->Name(), ": connecting: ", SystemNetwork::LastError());
    SystemNetwork::CloseSocket(c->socket);
    delete c;
    return 0;
  }
  INFO(c->Name(), ": connecting");
  conn[c->socket] = unique_ptr<SocketConnection>(c);

  if (connected) {
    /* connected 3 */ 
    c->SetConnected();
    c->SetSourceAddress();
    INFO(c->Name(), ": connected");
    if (this->Connected(c) < 0) c->SetError();
    if (c->handler) { if (c->handler->Connected(c) < 0) { ERROR(c->Name(), ": handler connected"); c->SetError(); } }
    if (c->detach) { Detach(c); conn.erase(c->socket); }
    else app->net->UpdateActive(c);
  } else {
    app->net->active.Add(c->socket, SocketSet::READABLE|SocketSet::WRITABLE, &c->self_reference);
  }
  return c;
}

SocketConnection *SocketService::Connect(const string &hostport, int default_port, Callback *detach) {
  int port;
  IPV4::Addr addr;
  if (!HTTP::ResolveHost(hostport.c_str(), 0, &addr, &port, 0, default_port)) return ERRORv(nullptr, "resolve ", hostport, " failed");
  return Connect(addr, port, NULL, detach);
}

SocketConnection *SocketService::SSLConnect(SSLSocket::CTXPtr sslctx, const string &hostport, int default_port, Callback *detach) {
  if (!sslctx) sslctx = app->net->ssl;
  if (!sslctx) return ERRORv(nullptr, "no ssl: ", -1);

  auto c = new SocketConnection(this, Connection::Connecting, 0, 0, detach);
  if (!HTTP::ResolveHost(hostport.c_str(), 0, &c->addr, &c->port, true, default_port)) return ERRORv(nullptr, "resolve: ", hostport);

  if ((c->socket = c->bio.Connect(sslctx, hostport)) == InvalidSocket) {
    ERROR(hostport, ": BIO_do_connect: ", c->bio.ErrorString());
    delete c;
    return 0;
  }

  INFO(c->Name(), ": connecting (fd=", c->socket, ")");
  conn[c->socket] = unique_ptr<SocketConnection>(c);
  app->net->active.Add(c->socket, SocketSet::WRITABLE, &c->self_reference);
  return c;
}

SocketConnection *SocketService::SSLConnect(SSLSocket::CTXPtr sslctx, IPV4::Addr addr, int port, Callback *detach) {
  if (!sslctx) sslctx = app->net->ssl;
  if (!sslctx) return ERRORv(nullptr, "no ssl: ", -1);

  auto c = new SocketConnection(this, Connection::Connecting, addr, port, detach);
  if ((c->socket = c->bio.Connect(sslctx, addr, port)) == InvalidSocket) {
    ERROR(c->Name(), ": BIO_do_connect: ", c->bio.ErrorString());
    delete c;
    return 0;
  }

  INFO(c->Name(), ": connecting (fd=", c->socket, ")");
  conn[c->socket] = unique_ptr<SocketConnection>(c);
  app->net->active.Add(c->socket, SocketSet::WRITABLE, &c->self_reference);
  return c;
}

SocketConnection *SocketService::AddConnectedSocket(Socket conn_socket, Connection::Handler *handler) {
  auto *conn = new SocketConnection(this, handler);
  CHECK_NE(-1, (conn->socket = conn_socket));
  conn->state = Connection::Connected;
  conn->svc->conn[conn->socket] = unique_ptr<SocketConnection>(conn);
  app->net->active.Add(conn->socket, SocketSet::READABLE, &conn->self_reference);
  return conn;
}

SocketConnection *SocketService::EndpointConnect(const string &endpoint_name) {
  auto c = (endpoint[endpoint_name] = make_unique<SocketConnection>(this, Connection::Connected, -1)).get();
  c->endpoint_name = endpoint_name;

  /* connected 4 */
  if (this->Connected(c) < 0) c->SetError();
  INFO(Protocol::Name(protocol), "(", Void(this), ") endpoint connect: ", endpoint_name);
  if (c->handler) { if (c->handler->Connected(c) < 0) { ERROR(c->Name(), ": handler connected"); c->SetError(); } }
  return c;
}

void SocketService::EndpointReadCB(string *endpoint_name, string *packet) {
  EndpointRead(*endpoint_name, packet->c_str(), packet->size());
  delete endpoint_name;
  delete packet;
}

void SocketService::EndpointRead(const string &endpoint_name, const char *buf, int len) {
  if (len) CHECK(buf);
  if (!app->MainThread()) return app->RunInMainThread(bind(&SocketService::EndpointReadCB, this, new string(endpoint_name), new string(buf, len)));

  auto ep = endpoint.find(endpoint_name);
  if (ep == endpoint.end()) { 
    if (!endpoint_read_autoconnect) return ERROR("unknown endpoint ", endpoint_name);
    if (!EndpointConnect(endpoint_name)) return ERROR("endpoint_read_autoconnect ", endpoint_name);
    ep = endpoint.find(endpoint_name);
    CHECK(ep != endpoint.end());
  }

  int ret;
  Connection *c = ep->second.get();
  if ((ret = c->AddPacket(buf, len)) != len) 
  { ERROR(c->Name(), ": addpacket(", len, ")"); c->SetError(); return; }
}

void SocketService::EndpointClose(const string &endpoint_name) {
  INFO(Protocol::Name(protocol), "(", Void(this), ") endpoint close: ", endpoint_name);
  auto ep = endpoint.find(endpoint_name);
  if (ep != endpoint.end()) ep->second->SetError();
}

void SocketService::Detach(SocketConnection *c) {
  INFO(c->Name(), ": detached from ", name);
  SocketService::Close(c);
  c->readable = c->writable = 0;
  app->RunInMainThread([=]() { (*c->detach)(); });
  app->scheduler.Wakeup(app->focused);
}

/* SocketServices */

SocketServices::SocketServices() {
  udp_client = make_unique<UDPClient>();
  tcp_client = make_unique<TCPClient>();
  unix_client = make_unique<UnixClient>();
  system_resolver = make_unique<SystemResolver>();
}

SocketServices::~SocketServices() {
  SSLSocket::Free();
}

int SocketServices::Init() {
  INFO("Network::Init()");
  ssl = SSLSocket::Init();
  Enable(unix_client.get());
  Enable(udp_client.get());
  Enable(tcp_client.get());
  system_resolver->HandleNoConnections();
  return 0;
}

int SocketServices::Shutdown(const vector<SocketService*> &st) { int ret = 0; for (auto s : st) if (Shutdown(s) < 0) ret = -1; return ret; }
int SocketServices::Disable (const vector<SocketService*> &st) { int ret = 0; for (auto s : st) if (Disable (s) < 0) ret = -1; return ret; }
int SocketServices::Enable  (const vector<SocketService*> &st) { int ret = 0; for (auto s : st) if (Enable  (s) < 0) ret = -1; return ret; }

int SocketServices::Enable(SocketService *s) {
  /* listen */
  if (s->listen.size() && !s->initialized) {
    s->initialized = true;
    vector<string> removelist;
    for (auto i = s->listen.begin(); i != s->listen.end(); ++i) {
      const IPV4Endpoint *listen_addr = IPV4Endpoint::FromString(i->first.c_str());
      if (s->Listen(listen_addr->addr, listen_addr->port, i->second.get()) == -1)
        removelist.push_back(i->first);
    }
    for (auto i = removelist.begin(); i != removelist.end(); ++i) s->listen.erase(*i);
  }

  /* insert */
  auto i = find(service_table.begin(), service_table.end(), s);
  if (i != service_table.end()) return 0;
  service_table.push_back(s);
  return 0;
}

int SocketServices::Disable(SocketService *s) {
  auto i = find(service_table.begin(), service_table.end(), s);
  if (i == service_table.end()) return -1;
  service_table.erase(i);
  return 0;
}

void SocketServices::ConnClose(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist) {
  if (c->handler) c->handler->Close(c);
  svc->Close(c);
  removelist->AddSocket(svc, c->socket);
}

void SocketServices::ConnCloseDetached(SocketService *svc, SocketConnection *c) {
  SystemNetwork::CloseSocket(c->socket);
  delete c;
}

void SocketServices::ConnCloseAll(SocketService *svc) {
  SocketServiceEndpointEraseList removelist;
  for (auto &i : svc->conn) ConnClose(svc, i.second.get(), &removelist);
  svc->conn.clear();
}

void SocketServices::EndpointRead(SocketService *svc, const char *name, const char *buf, int len) { return svc->EndpointRead(name, buf, len); }
void SocketServices::EndpointClose(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist, const string &epk) {
  if (c->handler) c->handler->Close(c);
  if (svc->listen.empty()) svc->Close(c);
  removelist->AddEndpoint(svc, epk);
}
void SocketServices::EndpointCloseAll(SocketService *svc) {
  SocketServiceEndpointEraseList unused;
  for (auto i = svc->endpoint.begin(); i != svc->endpoint.end(); ++i)
    EndpointClose(svc, i->second.get(), &unused, "");
  svc->endpoint.clear();
}

int SocketServices::Shutdown(SocketService *svc) {
  ConnCloseAll(svc);
  EndpointCloseAll(svc);
  return 0;
}

int SocketServices::Frame(unsigned clicks) {
  if (active.Select(select_time)) return ERRORv(-1, "SocketSet.select: ", SystemNetwork::LastError());
  SocketServiceEndpointEraseList removelist;

#ifndef LFL_NETWORK_MONOLITHIC_FRAME
  /* iterate events */
  for (active.cur_event = 0; active.cur_event < active.num_events; active.cur_event++) {
    typed_ptr *tp = reinterpret_cast<typed_ptr*>(active.events[active.cur_event].data.ptr);
    if      (auto c = tp->Get<Connection>()) { active.cur_fd = c->socket; TCPConnectionFrame(c->svc, c, &removelist); }
    else if (auto l = tp->Get<Listener  >()) { active.cur_fd = l->socket; AcceptFrame(l->svc, l); }
    else FATAL("unknown type", tp->type);
  }
#endif

  /* pre loop */
  for (int svc_i = 0, svc_l = service_table.size(); svc_i < svc_l; ++svc_i) {
    SocketService *svc = service_table[svc_i];

#ifdef LFL_NETWORK_MONOLITHIC_FRAME
    /* answer listening sockets & UDP server read */
    for (auto i = svc->listen.begin(), e = svc->listen.end(); i != e; ++i) 
      if (active.GetReadable(i->second->socket)) AcceptFrame(svc, i->second.get());

    /* iterate connections */
    for (auto i = svc->conn.begin(), e = svc->conn.end(); i != e; ++i)
      TCPConnectionFrame(svc, i->second.get(), &removelist);
#endif

    /* iterate endpoints */
    for (auto i = svc->endpoint.begin(), e = svc->endpoint.end(); i != e; i++) {
      UDPConnectionFrame(svc, i->second.get(), &removelist, i->first);
    }

    /* remove closed */
    removelist.Erase();

    if (svc->heartbeats) { /* connection heartbeats */
      for (auto i = svc->conn.begin(), e = svc->conn.end(); i != e; ++i) {
        auto c = i->second.get();
        int ret = c->handler ? c->handler->Heartbeat(c) : 0;
        if (c->state == Connection::Error || ret < 0) ConnClose(svc, c, &removelist);
      }

      for (auto i = svc->endpoint.begin(), e = svc->endpoint.end(); i != e; ++i) {
        auto c = i->second.get();
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

void SocketServices::AcceptFrame(SocketService *svc, SocketListener *listener) {
  for (;;) {
    bool inserted = false;
    SocketConnection *c = 0;
    if (listener->ssl) { /* SSL accept */
      Socket socket;
      SSLSocket sslsocket;
      if ((socket = listener->bio.Accept(&sslsocket)) == InvalidSocket) continue;

      struct sockaddr_in sin;
      socklen_t sinSize = sizeof(sin);
      if (::getpeername(socket, reinterpret_cast<struct sockaddr*>(&sin), &sinSize) < 0)
      { if (!SystemNetwork::EWouldBlock()) ERROR("getpeername: ", strerror(errno)); break; }

      /* insert socket */
      c = svc->Accept(Connection::Connected, socket, sin.sin_addr.s_addr, ntohs(sin.sin_port));
      if (!c) { SystemNetwork::CloseSocket(socket); continue; }
      c->bio = sslsocket;
    }
    else if (svc->protocol == Protocol::UDP) { /* UDP server read */
      struct sockaddr_in sin;
      socklen_t sinSize = sizeof(sin);
      int inserted = 0, ret;
      char buf[2048];
      int len = recvfrom(listener->socket, buf, sizeof(buf)-1, 0, reinterpret_cast<struct sockaddr*>(&sin), &sinSize);
      if (len <= 0) {
        if (SystemNetwork::EWouldBlock()) break;
        else { ERROR("recvfrom: ", SystemNetwork::LastError()); break; }
      }
      buf[len] = 0;

      IPV4Endpoint epk(sin.sin_addr.s_addr, ntohs(sin.sin_port));
      string epkstr = epk.ToString();
      auto ep = svc->endpoint.find(epkstr);
      if (ep != svc->endpoint.end()) c = ep->second.get();
      else {
        svc->fake.addr = epk.addr;
        svc->fake.port = epk.port;
        svc->fake.socket = listener->socket;
        if (svc->UDPFilter(&svc->fake, buf, len)) continue;
        c = (svc->endpoint[epkstr] = make_unique<SocketConnection>
             (svc, Connection::Connected, listener->socket, epk.addr, epk.port)).get();
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
    if (c->detach) { svc->Detach(c); svc->conn.erase(c->socket); }
  }
}

void SocketServices::TCPConnectionFrame(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist) {
  /* complete connecting sockets */
  if (c->state == Connection::Connecting && active.GetWritable(c->socket)) {
    int res=0;
    socklen_t resSize=sizeof(res);
    if (!getsockopt(c->socket, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&res), &resSize) && !res) {

      /* connected 2 */ 
      c->SetConnected();
      c->SetSourceAddress();
      INFO(c->Name(), ": connected");
      if (svc->Connected(c) < 0) c->SetError();
      if (c->handler) { if (c->handler->Connected(c) < 0) { ERROR(c->Name(), ": handler connected"); c->SetError(); } }
      if (c->detach) { svc->Detach(c); removelist->AddSocket(svc, c->socket); }
      else UpdateActive(c);
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
    else if (c->bio.ssl || active.GetReadable(c->socket)) { /* TCP Read */
      if (c->Read()<0) { c->SetError(); break; }
      if (c->rb.size()) {
        if (c->handler) { if (c->handler->Read(c) < 0) { ERROR(c->Name(), ": handler read"); c->SetError(); } }
      }
    }

    if (c->wb.size() && active.GetWritable(c->socket)) {
      if (static_cast<Connection*>(c)->WriteFlush()<0) { c->SetError(); break; }
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

void SocketServices::UDPConnectionFrame(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist, const string &epk) {
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

void SocketServices::UpdateActive(SocketConnection *c) {
  if (FLAGS_network_debug) INFO(c->Name(), " active = { ", c->readable?"READABLE":"", " , ", c->writable?"WRITABLE":"", " }");
  int flag = (c->readable?SocketSet::READABLE:0) | (c->writable?SocketSet::WRITABLE:0);
  active.Set(c->socket, flag, &c->self_reference);
}

/* SocketServicesThread */

void SocketServicesThread::ConnectionHandler::HandleMessage(Callback *cb) { 
  (*cb)();
  delete cb;
}

int SocketServicesThread::ConnectionHandler::Read(Connection *c) {
  int consumed = 0, s = sizeof(Callback*);
  for (; consumed + s <= c->rb.size(); consumed += s)
    HandleMessage(*reinterpret_cast<Callback**>(c->rb.begin() + consumed));
  if (consumed) c->ReadFlush(consumed);
  return 0;
}

SocketServicesThread::SocketServicesThread(SocketServices *N, bool Init) : net(N), init(Init),
  rd(new SocketConnection(app->net->unix_client.get(), new SocketServicesThread::ConnectionHandler())),
  wr(new SocketConnection(app->net->unix_client.get(), new SocketServicesThread::ConnectionHandler())),
  thread(make_unique<Thread>(bind(&SocketServicesThread::HandleMessagesLoop, this))) {
  Socket fd[2];
  CHECK(SystemNetwork::OpenSocketPair(fd));
  rd->state = wr->state = Connection::Connected;
  rd->socket = fd[0];
  wr->socket = fd[1];

  net->select_time = -1;
  rd->svc->conn[rd->socket] = unique_ptr<SocketConnection>(rd);
  net->active.Add(rd->socket, SocketSet::READABLE, &rd->self_reference);
}

void SocketServicesThread::Write(Callback *x) {
  CHECK_EQ(sizeof(x), wr->WriteFlush(reinterpret_cast<const char*>(&x), sizeof(x)));
}

void SocketServicesThread::HandleMessagesLoop() {
  if (init) net->Init();
  while (GetLFApp()->run) { net->Frame(0); }
}

/* UDP Client */

int UDPClient::PersistentConnectionHandler::Read(Connection *c) {
  for (int i=0; i<c->packets.size() && responseCB; i++) {
    if (c->state != Connection::Connected) break;
    responseCB(c, c->rb.begin() + c->packets[i].offset, c->packets[i].len);
  }
  return 0;
}

SocketConnection *UDPClient::PersistentConnection(const string &url, const ResponseCB &responseCB, const HeartbeatCB &heartbeatCB, int default_port) {
  int udp_port;
  IPV4::Addr ipv4_addr; 
  if (!HTTP::ResolveURL(url.c_str(), 0, &ipv4_addr, &udp_port, 0, 0, default_port))
    return ERRORv(nullptr, url, ": connect failed");

  auto c = Connect(ipv4_addr, udp_port);
  if (!c) return ERRORv(nullptr, url, ": connect failed");

  c->handler = make_unique<PersistentConnectionHandler>(responseCB, heartbeatCB);
  return c;
}

struct PacketHandler {
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

/* GPlusClient */

SocketConnection *GPlusClient::PersistentConnection(const string &name, UDPClient::ResponseCB responseCB, UDPClient::HeartbeatCB HCB) {
  auto c = EndpointConnect(name);
  c->handler = make_unique<PacketHandler::PersistentConnection>(responseCB, HCB);
  return c;
}

/* InProcessClient */

SocketConnection *InProcessClient::PersistentConnection(InProcessServer *server, UDPClient::ResponseCB responseCB,
                                                        UDPClient::HeartbeatCB HCB) {
  SocketConnection *c1 = EndpointConnect(StringPrintf("%p:%d", server, server->next_id++));
  SocketConnection *c2 = server->EndpointConnect(StringPrintf("%p:%d", this, next_id++));
  c1->handler = make_unique<PacketHandler::PersistentConnection>(responseCB, HCB);
  c1->next = c2;
  c2->next = c1;
  return c1;
}

/* Sniffer */

#ifdef LFL_PCAP
void Sniffer::Threadproc() {
  pcap_pkthdr *pkthdr; const unsigned char *packet; int ret;
  while (app->run && (ret = pcap_next_ex((pcap_t*)handle, &pkthdr, &packet)) >= 0) {
    if (!ret) continue;
    cb((const char *)packet, pkthdr->caplen, pkthdr->len);
  }
}

unique_ptr<Sniffer> Sniffer::Open(const string &dev, const string &filter, int snaplen, CB cb) {
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
  unique_ptr<Sniffer> sniffer = make_unique<Sniffer>(handle, ip, mask, cb);
  sniffer->thread.Open(bind(&Sniffer::Threadproc, sniffer.get()));
  sniffer->thread.Start();
  return sniffer;
}

void Sniffer::PrintDevices(vector<string> *out) {
  char errbuf[PCAP_ERRBUF_SIZE]; pcap_if_t *devs = 0; int ret;
  if ((ret = pcap_findalldevs(&devs, errbuf))) FATAL("pcap_findalldevs: ", ret);
  for (pcap_if_t *d = devs; d; d = d->next) {
    if (out) out->push_back(d->name);
    INFO(ret++, ": ", d->name, " : ", d->description ? d->description : "(none)");
  }
}

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
#else /* LFL_PCAP */
void Sniffer::Threadproc() {}
unique_ptr<Sniffer> Sniffer::Open(const string &dev, const string &filter, int snaplen, CB cb) { return ERRORv(nullptr, "sniffer not implemented"); }
void Sniffer::PrintDevices(vector<string> *out) {}
void Sniffer::GetDeviceAddressSet(set<IPV4::Addr> *out) {}
#endif /* LFL_PCAP */

void Sniffer::GetIPAddress(IPV4::Addr *out) {
  static IPV4::Addr localhost = IPV4::Parse("127.0.0.1");
  *out = 0;
#if defined(LFL_WINDOWS) || defined(LFL_EMSCRIPTEN)
#elif defined(LFL_ANDROID)
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getAddress", "()I"));
  jint addr = jni->env->CallIntMethod(jni->activity, mid);
  *out = ntohl(addr);
#else
  ifaddrs* ifap = NULL;
  int r = getifaddrs(&ifap);
  if (r) return ERROR("getifaddrs ", r);
  for (ifaddrs *i = ifap; i; i = i->ifa_next) {
    if (!i->ifa_dstaddr || i->ifa_dstaddr->sa_family != AF_INET) continue;
    IPV4::Addr addr = reinterpret_cast<struct sockaddr_in*>(i->ifa_addr)->sin_addr.s_addr;
    if (addr == localhost) continue;
    *out = addr;
    break;
  }
#endif
}

void Sniffer::GetBroadcastAddress(IPV4::Addr *out) {
  static IPV4::Addr localhost = IPV4::Parse("127.0.0.1");
  *out = 0;
#if defined(LFL_WINDOWS) || defined(LFL_EMSCRIPTEN)
#elif defined(LFL_ANDROID)
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getBroadcastAddress", "()I"));
  jint addr = jni->env->CallIntMethod(jni->activity, mid);
  *out = ntohl(addr);
#else
  ifaddrs* ifap = NULL;
  int r = getifaddrs(&ifap);
  if (r) return ERROR("getifaddrs ", r);
  for (ifaddrs *i = ifap; i; i = i->ifa_next) {
    if (!i->ifa_dstaddr || i->ifa_dstaddr->sa_family != AF_INET) continue;
    IPV4::Addr addr = reinterpret_cast<struct sockaddr_in*>(i->ifa_dstaddr)->sin_addr.s_addr;
    if (addr == localhost) continue;
    *out = addr;
    break;
  }
#endif
}

#ifdef LFL_GEOIP
unique_ptr<GeoResolution> GeoResolution::Open(const string &db) {
  void *impl = GeoIP_open(db.c_str(), GEOIP_INDEX_CACHE);
  if (!impl) return 0;
  return make_unique<GeoResolution>(impl);
}
bool GeoResolution::Resolve(const string &addr, string *country, string *region, string *city, float *lat, float *lng) {
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
unique_ptr<GeoResolution> GeoResolution::Open(const string &db) { return nullptr; }
bool GeoResolution::Resolve(const string &addr, string *country, string *region, string *city, float *lat, float *lng) { FATAL("not implemented"); }
#endif

bool NBReadable(Socket fd, int timeout) {
  SelectSocketSet ss;
  ss.Add(fd, SocketSet::READABLE, 0);
  ss.Select(timeout);
  return app->run && ss.GetReadable(fd);
}

int NBRead(Socket fd, char *buf, int len, int timeout) {
  if (timeout && !NBReadable(fd, timeout)) return 0;
  int o = 0, s = 0;
  do {
    if ((s = read(fd, buf+o, len-o)) <= 0) {
      if (!s || (s < 0 && !SystemNetwork::EWouldBlock())) return o ? o : -1;
    } else o += s;
  } while (s > 0 && len - o > 1024);
  return o;
}

int NBRead(Socket fd, string *buf, int timeout) {
  int l = NBRead(fd, &(*buf)[0], buf->size(), timeout);
  buf->resize(max(0,l));
  return l;
}

#if 1
int FWrite(FILE *f, const string &s) { 
  int ret = fwrite(s.data(), 1, s.size(), f); 
  if (FLAGS_network_debug && ret >= 0) INFO("FWrite(", fileno(f), ", ", ret, ", '", s.substr(0, ret), "')");
  return ret;
}
#else
int FWrite(FILE *f, const string &s) { return write(fileno(f), s.data(), s.size()); }
#endif
bool FWriteSuccess(FILE *f, const string &s) { return FWrite(f, s) == s.size(); }
bool FGets(char *buf, int len) { return NBFGets(stdin, buf, len); }
bool NBFGets(FILE *f, char *buf, int len, int timeout) {
#ifndef LFL_WINDOWS
  int fd = fileno(f);
  SelectSocketSet ss;
  ss.Add(fd, SocketSet::READABLE, 0);
  ss.Select(timeout);
  if (!app->run || !ss.GetReadable(fd)) return 0;
  fgets(buf, len, f);
  return 1;
#else
  return 0;
#endif
}

string PromptFGets(const string &p, int s) {
  printf("%s\n", p.c_str());
  fflush(stdout);
  string ret(s, 0);
  fgets(&ret[0], ret.size(), stdin);
  return ret;
}

}; // namespace LFL
