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

#ifndef LFL_CORE_APP_NETWORK_H__
#define LFL_CORE_APP_NETWORK_H__

namespace LFL {
DECLARE_bool(network_debug);
extern const Socket InvalidSocket;

struct SocketType { static const int Stream, Datagram, SeqPacket, Raw; };
struct Protocol { 
  enum { TCP=1, UDP=2, UNIX=3, GPLUS=4, InProcess=5 };
  static const char *Name(int p);
};

struct Ethernet {
  UNALIGNED_struct Header {
    static const int Size = 14, AddrSize = 6;
    unsigned char dst[AddrSize], src[AddrSize];
    unsigned short type;
  }; UNALIGNED_END(Header, Header::Size);
};

struct IPV4 {
  typedef unsigned Addr;
  static const Addr ANY;

  UNALIGNED_struct Header {
    static const int MinSize = 20;
    unsigned char vhl, tos;
    unsigned short len, id, off;
    unsigned char ttl, prot;
    unsigned short checksum;
    unsigned int src, dst;
    int version() const { return vhl >> 4; }
    int hdrlen() const { return (vhl & 0x0f); }
  }; UNALIGNED_END(Header, Header::MinSize);

  static Addr Parse(const string &ip);
  static void ParseCSV(const string &text, vector<Addr> *out);
  static void ParseCSV(const string &text, set<Addr> *out);
  static string MakeCSV(const vector<Addr> &in);
  static string MakeCSV(const set<Addr> &in);
  static string Text(Addr addr)           { return StringPrintf("%u.%u.%u.%u",    addr&0xff, (addr>>8)&0xff, (addr>>16)&0xff, (addr>>24)&0xff); }
  static string Text(Addr addr, int port) { return StringPrintf("%u.%u.%u.%u:%u", addr&0xff, (addr>>8)&0xff, (addr>>16)&0xff, (addr>>24)&0xff, port); }
};

struct TCP {
  UNALIGNED_struct Header {
    static const int MinSize = 20;
    unsigned short src, dst;
    unsigned int seqn, ackn;
#ifdef LFL_BIG_ENDIAN
    unsigned char offx2, fin:1, syn:1, rst:1, push:1, ack:1, urg:1, exe:1, cwr:1;
#else
    unsigned char offx2, cwr:1, exe:1, urg:1, ack:1, push:1, rst:1, syn:1, fin:1;
#endif
    unsigned short win, checksum, urgp;
    int offset() const { return offx2 >> 4; }
  }; UNALIGNED_END(Header, Header::MinSize);
};

struct UDP {
  UNALIGNED_struct Header {
    static const int Size = 8;
    unsigned short src, dst, len, checksum;
  }; UNALIGNED_END(Header, Header::Size);
};

struct IPV4Endpoint {
  IPV4::Addr addr=0;
  int port=0;
  IPV4Endpoint() {}
  IPV4Endpoint(int A, int P) : addr(A), port(P) {};
  string name() const { return IPV4::Text(addr, port); }
  string ToString() const { string s; s.resize(sizeof(*this)); memcpy(&s[0], this, sizeof(*this)); return s; }
  static const IPV4Endpoint *FromString(const char *s) { return reinterpret_cast<const IPV4Endpoint*>(s); }
};

struct IPV4EndpointSource { 
  virtual bool Available() const { return true; }
  virtual void Get(IPV4::Addr *addr, int *port) = 0;
  virtual void Get(IPV4::Addr  addr, int *port) = 0;
  virtual void Close(IPV4::Addr addr, int port) = 0;
  virtual void BindFailed(IPV4::Addr addr, int port) {}
};

struct SingleIPV4Endpoint : public IPV4EndpointSource {
  IPV4Endpoint val;
  SingleIPV4Endpoint(IPV4::Addr A, int P) : val(A, P) {}
  void Get(IPV4::Addr *A, int *P) { *A=val.addr; *P=val.port; }
  void Get(IPV4::Addr  A, int *P) { if (A == val.addr) *P=val.port; }
  void Close(IPV4::Addr A, int P) { CHECK_EQ(A, val.addr); CHECK_EQ(P, val.port); }
};

struct IPV4EndpointPool : public IPV4EndpointSource {
  static const int ports_per_ip = 65536-1024, bytes_per_ip = ports_per_ip/8;
  vector<IPV4::Addr> source_addrs;
  vector<string> source_ports;
  IPV4EndpointPool(const string &ip_csv);
  bool Available() const;
  void Close(IPV4::Addr addr, int port);
  void Get(IPV4::Addr *addr, int *port); 
  void Get(IPV4::Addr addr, int *port);
  bool GetPort(int ind, int *port);
};

struct IPV4EndpointPoolFilter : public IPV4EndpointSource {
  set<IPV4::Addr> filter;
  IPV4EndpointPool *wrap;
  IPV4EndpointPoolFilter(IPV4EndpointPool *W) : wrap(W) {}
  bool Available() const { return wrap->Available(); }
  void Close(IPV4::Addr addr, int port) { return wrap->Close(addr, port); }
  void Get(IPV4::Addr addr, int *port) { return wrap->Get(addr, port); }
  bool GetPort(int ind, int *port) { return wrap->GetPort(ind, port); }
  void Get(IPV4::Addr *addr, int *port);
};

struct SystemNetwork {
  static void CloseSocket(Socket);
  static Socket OpenSocket(int protocol);
  static bool OpenSocketPair(Socket *fd_out, int sock_type=SocketType::Stream, bool close_on_exec=true);
  static int SetSocketBlocking(Socket fd, int blocking);
  static int SetSocketCloseOnExec(Socket fd, int close);
  static int SetSocketBroadcastEnabled(Socket fd, int enabled);
  static int SetSocketBufferSize(Socket fd, bool send_or_recv, int size);
  static int GetSocketBufferSize(Socket fd, bool send_or_recv);

  static int Bind(int fd, IPV4::Addr addr, int port);
  static Socket Accept(Socket listener, IPV4::Addr *addr, int *port, bool blocking=false);
  static Socket Listen(int protocol, IPV4::Addr addr, int port, int backlog=32, bool blocking=false);
  static int Connect(Socket fd, IPV4::Addr addr, int port, int *connected);
  static int SendTo(Socket fd, IPV4::Addr addr, int port, const char *buf, int len);
  static int GetSockName(Socket fd, IPV4::Addr *addr_out, int *port_out);
  static int GetPeerName(Socket fd, IPV4::Addr *addr_out, int *port_out);
  static string GetHostByAddr(IPV4::Addr addr);
  static IPV4::Addr GetHostByName(const string &host);
  static int IOVLen(const iovec*, int len);
  static bool EWouldBlock();
  static string LastError();
};

struct TransferredSocket { Socket socket; int offset; };

struct SSLSocket {
  struct SSLPtr : public VoidPtr { using VoidPtr::VoidPtr; };
  struct BIOPtr : public VoidPtr { using VoidPtr::VoidPtr; };
  struct CTXPtr : public VoidPtr { using VoidPtr::VoidPtr; };
  Socket socket = InvalidSocket;
  int last_error = 0;
  bool ready = 0;
  SSLPtr ssl = 0;
  BIOPtr bio = 0;
  string buf;
  virtual ~SSLSocket();
  string ErrorString() const;
  ptrdiff_t Read(char *buf, int readlen);
  ptrdiff_t Write(const StringPiece &b);
  Socket Connect(CTXPtr sslctx, const string &hostport);
  Socket Connect(CTXPtr sslctx, IPV4::Addr addr, int port);
  Socket Listen(CTXPtr sslctx, int port, bool reuse);
  Socket Accept(SSLSocket *out);
  static CTXPtr Init();
  static void Free();
};

struct SocketSet {
  enum { READABLE=1, WRITABLE=2, EXCEPTION=4 };
  virtual ~SocketSet() {}
  virtual void Del(Socket fd) = 0;
  virtual void Add(Socket fd, int flag, void *val) = 0;
  virtual void Set(Socket fd, int flag, void *val) = 0;
  virtual int GetReadable(Socket fd) { return 0; };
  virtual int GetWritable(Socket fd) { return 0; };
  virtual int GetException(Socket fd) { return 0; };
  virtual int Select(int wait_time) { return 0; };
};

struct SelectSocketSet : public SocketSet {
  unordered_map<Socket, pair<int, void*> > socket;
  fd_set rfds, wfds, xfds;
  SocketSet *mirror=0;

  int Select(int wait_time);
  void Del(Socket fd)                    { socket.erase(fd);                if (mirror) mirror->Del(fd); }
  void Add(Socket fd, int flag, void *v) { socket[fd] = make_pair(flag, v); if (mirror) mirror->Add(fd, flag, v); }
  void Set(Socket fd, int flag, void *v) { socket[fd] = make_pair(flag, v); if (mirror) mirror->Set(fd, flag, v); }
  int Get(Socket fd, fd_set *set) { return FD_ISSET(fd, set); } 
  int GetReadable(Socket fd) { return Get(fd, &rfds); }
  int GetWritable(Socket fd) { return Get(fd, &wfds); }
  int GetException(Socket fd) { return Get(fd, &xfds); }
  string DebugString() const {
    string ret="SelectSocketSet={";
    for (auto &s : socket) StrAppend(&ret, s.first, ", ");
    return StrCat(ret.substr(0, ret.size()-2), "}");
  }
};

#if defined(LFL_WFMO) && defined(LFL_WINDOWS)
struct WFMOSocketSet : public SocketSet {
  static const int max_sockets = MAXIMUM_WAIT_OBJECTS;
  struct Event { struct Data { void *ptr=0; } data; } events[1];
  SortedVector<HANDLE> sockets;
  int num_events=0, cur_event=0;
  Socket cur_fd=-1;
  WFMOSocketSet() { sockets.reserve(max_sockets); }

  int Select(int wait_time) {
    int ret = WaitForMultipleObjects(sockets.size(), sockets.data(), FALSE, ToMilliseconds(wait_time));
    if (ret == WAIT_FAILED) return ERRORv(-1, "WFMO ", GetLastError());
    num_events = ret != WAIT_TIMEOUT;
    return 0;
  }
  void Del(Socket fd) {}
  void Add(Socket fd, int flag, void *val) {}
  void Set(Socket fd, int flag, void *val) {}
  int GetReadable(Socket fd) { return 0; }
  int GetWritable(Socket fd) { return 0; }
  int GetException(Socket fd) { return 0; }
};
typedef WFMOSocketSet LFLSocketSet;

#elif defined(LFL_EPOLL) && defined(LFL_LINUX_SERVER)
#include <sys/epoll.h>
template <int S> struct EPollSocketSet : public SocketSet {
  Socket epollfd, cur_fd;
  epoll_event events[S];
  int cur_event=-1, num_events=0;
  EPollSocketSet() : epollfd(epoll_create(S)) {}
  virtual ~EPollSocketSet() { close(epollfd); }

  int Select(int wait_time) {
    if ((num_events = epoll_wait(epollfd, events, S, ToMilliseconds(wait_time))) == -1) ERROR("epoll_wait() ", strerror(errno));
    return 0;
  }
  void Change(Socket fd, int op, int flag, void *val) {
    epoll_event ev;
    memzero(ev);
    ev.data.ptr = val;
    ev.events = ((flag & READABLE) ? EPOLLIN : 0) | ((flag & WRITABLE) ? EPOLLOUT : 0);
    if (epoll_ctl(epollfd, op, fd, &ev) == -1) ERROR("epoll_ctl(", epollfd, ", ", op, ", ", events, "): ", strerror(errno)); 
  }
  void Del(Socket fd)                      { Change(fd, EPOLL_CTL_DEL, READABLE|WRITABLE, 0);   }
  void Add(Socket fd, int flag, void *val) { Change(fd, EPOLL_CTL_ADD, flag,              val); }
  void Set(Socket fd, int flag, void *val) { Change(fd, EPOLL_CTL_MOD, flag,              val); }
  int GetReadable(Socket fd) { return Get(fd)->events & (EPOLLIN  | EPOLLERR | EPOLLHUP); }
  int GetWritable(Socket fd) { return Get(fd)->events & (EPOLLOUT | EPOLLERR | EPOLLHUP); }
  int GetException(Socket fd) { return 0; }
  epoll_event *Get(Socket fd) {
    CHECK_EQ(fd, cur_fd);
    CHECK(cur_event >= 0 && cur_event < S);
    return &events[cur_event];
  }
};
typedef EPollSocketSet<65536 * 6> LFLSocketSet;

#else
#define LFL_NETWORK_MONOLITHIC_FRAME
typedef SelectSocketSet LFLSocketSet;
#endif

/// SocketWakeupThread waits on SocketSet and calls window->focused->Wakeup() on event
struct SocketWakeupThread : public SocketSet {
  WindowHolder *window;
  ThreadDispatcher *dispatch;
  mutex *frame_mutex, *wait_mutex;
  SelectSocketSet sockets;
  mutex sockets_mutex;
  Thread thread;
  Socket pipe[2];
  ~SocketWakeupThread();
  SocketWakeupThread(WindowHolder *W, ThreadDispatcher *D, mutex *FM=0, mutex *WM=0) :
    window(W), dispatch(D), frame_mutex(FM), wait_mutex(WM),
    thread(bind(&SocketWakeupThread::ThreadProc, this)) { pipe[0] = pipe[1] = -1; }

  void Add(Socket s, int f, void *v) { { ScopedMutex m(sockets_mutex); sockets.Add(s, f, v); } Wakeup(); }
  void Set(Socket s, int f, void *v) { { ScopedMutex m(sockets_mutex); sockets.Set(s, f, v); } Wakeup(); }
  void Del(Socket s)                 { { ScopedMutex m(sockets_mutex); sockets.Del(s);       } Wakeup(); }
  void Start();
  void Wait() { Wakeup(); thread.Wait(); }
  void Wakeup();
  void ThreadProc();
};

struct Connection {
  enum { Connected=1, Connecting=2, Reconnect=3, Handshake=4, Error=5 };
  typedef function<void(Connection*)> CB;
  struct Handler {  
    virtual ~Handler() {}
    virtual int Heartbeat(Connection *c) { return 0; }
    virtual int Connected(Connection *c) { return 0; }
    virtual int Read(Connection *c) { return 0; }    
    virtual int Flushed(Connection *c) { return 0; }
    virtual void Close(Connection *c) {}
  };
  struct CallbackHandler : public Handler {
    CB heartbeat_cb, connected_cb, read_cb, flushed_cb, close_cb;
    CallbackHandler(const CB &R, const CB &C) : read_cb(R), close_cb(C) {}
    virtual int Heartbeat(Connection *c) { if (heartbeat_cb) heartbeat_cb(c); return 0; }
    virtual int Connected(Connection *c) { if (connected_cb) connected_cb(c); return 0; }
    virtual int Read(Connection *c) { if (read_cb) read_cb(c); return 0; }    
    virtual int Flushed(Connection *c) { if (flushed_cb) flushed_cb(c); return 0; }
    virtual void Close(Connection *c) { if (close_cb) close_cb(c); }
  };

  SocketService *svc;
  Time ct, rt, wt;
  string endpoint_name;
  bool readable=1, writable=0, control_messages=0, detach_delete=0;
  int state;
  StringBuffer rb, wb;
  typed_ptr self_reference;
  vector<IOVec> packets;
  deque<TransferredSocket> transferred_socket;
  unique_ptr<Handler> handler;
  Connection *next=0;
  CB *detach;
  void *data=0;

  Connection(SocketService *s=0, int t=Error, Handler *h=0, CB *Detach=0) : svc(s), ct(Now()), rt(Now()), wt(Now()), state(t), rb(65536), wb(65536), self_reference(MakeTyped(this)), handler(h), detach(Detach) {}
  virtual ~Connection();

  virtual void Close()                                              = 0;
  virtual int  Read()                                               = 0;
  virtual int  WriteFlush(const char *buf, int len)                 = 0;
  virtual void AddToMainWait(Window*, function<bool()> readable_cb) = 0;
  virtual void RemoveFromMainWait(Window*)                          = 0;

  virtual string Name() const { return endpoint_name; }
  virtual IPV4Endpoint RemoteIPV4() const { return IPV4Endpoint(0, 0); }
  virtual IPV4Endpoint LocalIPV4() const { return IPV4Endpoint(0, 0); }
  virtual Socket GetSocket() const { return InvalidSocket; }
  virtual int ReadPackets() { int ret = 1; while(ret > 0) ret = ReadPacket(); return ret; }
  virtual int Write(const char *buf, int len) { return WriteFlush(buf, len); }
  virtual int WriteVFlush(const iovec *iov, int len) { string b; for (auto i=iov, e=i+len; i!=e; ++i) b.append(static_cast<const char *>(i->iov_base), i->iov_len); return WriteFlush(b.data(), b.size()); }
  virtual int WriteVFlush(const iovec *iov, int len, int xfer_socket) { return xfer_socket == InvalidSocket ? WriteVFlush(iov, len) : -1; }
  virtual int SendTo(const char *buf, int len) { return -1; } 

  void SetError() { state = Error; ct = Now(); }
  void SetConnected() { state = Connected; ct = Now(); }
  void SetReconnect() { state = Reconnect; ct = Now(); }
  void SetConnecting() { state = Connecting; ct = Now(); }
  int Write(const string &buf) { return Write(buf.c_str(), buf.size()); }
  int WriteFlush();
  int WriteFlush(const string &buf) { return WriteFlush(buf.c_str(), buf.size()); }
  int WriteFlush(const char *buf, int len, int transfer_socket);
  int Reads();
  int ReadPacket();
  int Add(const char *buf, int len);
  int AddPacket(const char *buf, int len);
  int ReadFlush(int len);

  static const char *StateName(int n);
  static bool ConnectState(int n);
};

struct SocketConnection : public Connection {
  Socket socket;
  IPV4::Addr addr, src_addr=0;
  int port, src_port=0;
  SSLSocket bio;

  SocketConnection(SocketService *s, Handler *h,                                     CB *Detach=0) : Connection(s, Error, h, Detach), socket(-1),   addr(0),    port(0)    {}
  SocketConnection(SocketService *s, int State, int Sock,                            CB *Detach=0) : Connection(s, State, 0, Detach), socket(Sock), addr(0),    port(0)    {}
  SocketConnection(SocketService *s, int State, int Sock, IPV4::Addr Addr, int Port, CB *Detach=0) : Connection(s, State, 0, Detach), socket(Sock), addr(Addr), port(Port) {}
  SocketConnection(SocketService *s, int State,           IPV4::Addr Addr, int Port, CB *Detach=0) : Connection(s, State, 0, Detach), socket(-1),   addr(Addr), port(Port) {}

  IPV4Endpoint RemoteIPV4() const override { return IPV4Endpoint(addr, port); }
  IPV4Endpoint LocalIPV4() const override { return IPV4Endpoint(src_addr, src_port); }
  Socket GetSocket() const override { return socket; }
  string Name() const override { return !endpoint_name.empty() ? endpoint_name : IPV4::Text(addr, port); }
  void Close() override;
  int Read() override;
  int ReadPackets() override;
  int Write(const char *buf, int len) override;
  int WriteFlush(const char *buf, int len) override;
  int WriteVFlush(const iovec *iov, int len) override;
  int WriteVFlush(const iovec *iov, int len, int transfer_socket) override;
  int SendTo(const char *buf, int len) override;
  int SetSourceAddress() { return SystemNetwork::GetSockName(socket, &src_addr, &src_port); }
  void AddToMainWait(Window*, function<bool()> readable_cb) override;
  void RemoveFromMainWait(Window*) override;
};

struct SocketListener {
  bool ssl;
  Socket socket = InvalidSocket;
  SocketService *svc;
  SSLSocket bio;
  typed_ptr self_reference;
  SocketListener(SocketService *s, bool SSL=false) : ssl(SSL), svc(s), self_reference(MakeTyped(this)) {}
};

struct SocketService {
  SocketServices *net;
  string name;
  int protocol, reconnect=0;
  bool initialized=0, heartbeats=0, endpoint_read_autoconnect=0;
  void *game_network=0;
  map<string, unique_ptr<SocketListener>> listen;
  map<Socket, unique_ptr<SocketConnection>> conn;
  map<string, unique_ptr<SocketConnection>> endpoint;
  IPV4EndpointSource *connect_src_pool=0;
  SocketConnection fake;
  SocketService(SocketServices *N, const string &n, int prot=Protocol::TCP) : net(N), name(n), protocol(prot), fake(this, Connection::Connected, 0) {}

  void QueueListen(IPV4::Addr addr, int port, bool SSL=false) { QueueListen(IPV4Endpoint(addr,port).ToString(), SSL); }
  void QueueListen(const string &n, bool SSL=false) { listen[n] = make_unique<SocketListener>(this, SSL); }
  SocketListener *GetListener() { return listen.size() ? listen.begin()->second.get() : 0; }

  int OpenSocket(SocketConnection *c, int protocol, int blocking, IPV4EndpointSource*);
  Socket Listen(IPV4::Addr addr, int port, SocketListener*);
  SocketConnection *Accept(int state, Socket socket, IPV4::Addr addr, int port);
  SocketConnection *Connect(IPV4::Addr addr, int port, IPV4EndpointSource *src_addr=0, Connection::CB *detach=0);
  SocketConnection *Connect(IPV4::Addr addr, int port, IPV4::Addr src_addr, int src_port, Connection::CB *detach=0);
  SocketConnection *Connect(const string &hostport, int default_port=0, Connection::CB *detach=0);
  SocketConnection *SSLConnect(SSLSocket::CTXPtr sslctx, IPV4::Addr addr, int port, Connection::CB *detach=0);
  SocketConnection *SSLConnect(SSLSocket::CTXPtr sslctx, const string &hostport, int default_port=0, Connection::CB *detach=0);
  SocketConnection *AddConnectedSocket(Socket socket, Connection::Handler*);
  SocketConnection *EndpointConnect(const string &endpoint_name);
  void EndpointReadCB(string *endpoint_name, string *packet);
  void EndpointRead(const string &endpoint_name, const char *buf, int len);
  void EndpointClose(const string &endpoint_name);
  void Detach(SocketConnection *c);

  virtual void Close(SocketConnection *c);
  virtual int UDPFilter(SocketConnection *e, const char *buf, int len) { return 0; }
  virtual int Connected(SocketConnection *c) { return 0; }
  virtual int Frame() { return 0; }
};

struct SocketServiceEndpointEraseList {
  vector<pair<SocketService*, Socket> > sockets;
  vector<pair<SocketService*, string> > endpoints;
  void AddSocket  (SocketService *s, Socket fd)        { sockets  .emplace_back(s, fd); }
  void AddEndpoint(SocketService *s, const string &en) { endpoints.emplace_back(s, en); }
  void Erase() {
    for (auto &r : endpoints) r.first->endpoint.erase(r.second);
    for (auto &r : sockets)   r.first->conn    .erase(r.second);
    sockets  .clear();
    endpoints.clear();
  }
};

struct UnixClient : public SocketService { UnixClient(SocketServices *N)                  : SocketService(N, "UnixClient", Protocol::UNIX) {} };
struct UnixServer : public SocketService { UnixServer(SocketServices *N, const string &n) : SocketService(N, "UnixServer", Protocol::UNIX) { QueueListen(n); } };

struct UDPClient : public SocketService {
  static const int MTU = 1500;
  enum { Write=1, Sendto=2 };
  typedef function<void(Connection*, const char*, int)> ResponseCB;
  typedef function<void(Connection*)> HeartbeatCB; 
  struct PersistentConnectionHandler : public Connection::Handler {
    UDPClient::ResponseCB responseCB;
    UDPClient::HeartbeatCB heartbeatCB;
    PersistentConnectionHandler(const UDPClient::ResponseCB &RCB, const UDPClient::HeartbeatCB &HCB) : responseCB(RCB), heartbeatCB(HCB) {}
    int Heartbeat(Connection *c) { if (heartbeatCB) heartbeatCB(c); return 0; }
    void Close(Connection *c) { if (responseCB) responseCB(c, 0, 0); }
    int Read(Connection *c);
  };
  UDPClient(SocketServices *N) : SocketService(N, "UDPClient", Protocol::UDP) { heartbeats=true; }
  SocketConnection *PersistentConnection(const string &url, const ResponseCB& cb, int default_port) { return PersistentConnection(url, cb, HeartbeatCB(), default_port); }
  SocketConnection *PersistentConnection(const string &url, const ResponseCB&, const HeartbeatCB&, int default_port);
};

struct UDPServer : public SocketService {
  virtual ~UDPServer() {}
  Connection::Handler *handler=0;
  UDPServer(SocketServices *N, int port) : SocketService(N, "UDPServer", Protocol::UDP) { QueueListen(0, port); }
  int Connected(SocketConnection *c) override { c->handler = unique_ptr<Connection::Handler>(handler); return 0; }
  void Close(SocketConnection *c) override { c->handler.release(); SocketService::Close(c); }
};

struct TCPClient : public SocketService { TCPClient(SocketServices *N) : SocketService(N, "TCPClient") {} };
struct TCPServer : public SocketService {
  Connection::Handler *handler=0;
  TCPServer(SocketServices *N, int port) : SocketService(N, "TCPServer") { QueueListen(0, port); }
  int Connected(SocketConnection *c) override { c->handler = unique_ptr<Connection::Handler>(handler); return 0; }
  void Close(SocketConnection *c) override { c->handler.release(); SocketService::Close(c); }
};

struct GPlusClient : public SocketService {
  static const int MTU = 1500;
  enum { Write=1, Sendto=2 };
  GPlusClient(SocketServices *N) : SocketService(N, "GPlusClient", Protocol::GPLUS) { heartbeats=true; }
  SocketConnection *PersistentConnection(const string &name, UDPClient::ResponseCB cb, UDPClient::HeartbeatCB HCB);
};

struct GPlusServer : public SocketService {
  virtual ~GPlusServer() {}
  Connection::Handler *handler=0;
  GPlusServer(SocketServices *N) : SocketService(N, "GPlusServer", Protocol::GPLUS) { endpoint_read_autoconnect=1; }
  int Connected(SocketConnection *c) override { c->handler = unique_ptr<Connection::Handler>(handler); return 0; }
  void Close(SocketConnection *c) override { c->handler.release(); SocketService::Close(c); }
};

struct InProcessServer : public SocketService {
  Connection::Handler *handler=0;
  int next_id=1;
  virtual ~InProcessServer() {}
  InProcessServer(SocketServices *N) : SocketService(N, "InProcessServer", Protocol::InProcess) { endpoint_read_autoconnect=1; }
  int Connected(SocketConnection *c) override { c->handler = unique_ptr<Connection::Handler>(handler); return 0; }
  void Close(SocketConnection *c) override { c->handler.release(); SocketService::Close(c); }
};

struct InProcessClient : public SocketService {
  int next_id=1;
  virtual ~InProcessClient() {}
  InProcessClient(SocketServices *N) : SocketService(N, "InProcessClient", Protocol::InProcess) {}
  SocketConnection *PersistentConnection(InProcessServer*, UDPClient::ResponseCB cb, UDPClient::HeartbeatCB HCB);
};

struct SocketServices : public Module {
  int select_time=0;
  LFLSocketSet active;
  ThreadDispatcher *dispatcher;
  WakeupHandle *wakeup;
  vector<SocketService*> service_table;
  unique_ptr<UDPClient> udp_client;
  unique_ptr<TCPClient> tcp_client;
  unique_ptr<UnixClient> unix_client;
  unique_ptr<SystemResolver> system_resolver;
  LazyInitializedPtr<RecursiveResolver> recursive_resolver;
  LazyInitializedPtr<InProcessClient> inprocess_client;
  LazyInitializedPtr<GPlusClient> gplus_client;
  SSLSocket::CTXPtr ssl;
  SocketServices(ThreadDispatcher*, WakeupHandle*);
  virtual ~SocketServices();

  int Init();
  int Enable(SocketService *svc);
  int Disable(SocketService *svc);
  int Shutdown(SocketService *svc);
  int Enable(const vector<SocketService*> &svc);
  int Disable(const vector<SocketService*> &svc);
  int Shutdown(const vector<SocketService*> &svc);
  int Frame(unsigned);
  void AcceptFrame(SocketService *svc, SocketListener *listener);
  void TCPConnectionFrame(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist);
  void UDPConnectionFrame(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist, const string &epk);

  void ConnClose(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist);
  void ConnCloseDetached(SocketService *svc, SocketConnection *c);
  void ConnCloseAll(SocketService *svc);

  void EndpointRead(SocketService *svc, const char *name, const char *buf, int len);
  void EndpointClose(SocketService *svc, SocketConnection *c, SocketServiceEndpointEraseList *removelist, const string &epk);
  void EndpointCloseAll(SocketService *svc);

  void UpdateActive(SocketConnection *c);
};

/// SocketServicesThread runs the SocketServices Module in a new thread with a multiplexed Callback queue
struct SocketServicesThread {
  struct ConnectionHandler : public Connection::Handler {
    void HandleMessage(Callback *cb);
    int Read(Connection *c);
  };

  bool init=0;
  SocketServices *net=0;
  SocketConnection *rd=0;
  unique_ptr<SocketConnection> wr;
  unique_ptr<Thread> thread;
  SocketServicesThread(SocketServices *N, bool Init);

  void Write(Callback *x);
  void HandleMessagesLoop();
};

struct Sniffer {
  typedef function<void(const char*, int, int)> CB;
  ThreadDispatcher *dispatch;
  Thread thread;
  CB cb;
  int ip, mask;
  void *handle;
  Sniffer(ThreadDispatcher *D, void *H, int I, int M, CB C) : dispatch(D), cb(C), ip(I), mask(M), handle(H) {}
  ~Sniffer() { thread.Wait(); }
  void Threadproc();
  static unique_ptr<Sniffer> Open(const string &dev, const string &filter, int snaplen, CB cb);
  static void PrintDevices(vector<string> *out);
  static void GetDeviceAddressSet(set<IPV4::Addr> *out);
  static void GetIPAddress(IPV4::Addr *out);
  static void GetBroadcastAddress(IPV4::Addr *out);
};

struct GeoResolution {
  void *impl;
  GeoResolution(void *I) : impl(I) {}
  bool Resolve(const string &addr, string *country, string *region, string *city, float *lat, float *lng);
  static unique_ptr<GeoResolution> Open(const string &db);
};

int NBRead(Socket fd, char *buf, int size, ApplicationLifetime *life=0, int timeout=0);
int NBRead(Socket fd, string *buf, ApplicationLifetime *life=0, int timeout=0);
bool NBReadable(Socket fd, ApplicationLifetime *life=0, int timeout=0);
bool NBFGets(FILE*, char *buf, int size, ApplicationLifetime *life=0, int timeout=0);
bool FGets(char *buf, int size, ApplicationLifetime *life=0);
int FWrite(FILE *f, const string &s);
bool FWriteSuccess(FILE *f, const string &s);
string PromptFGets(const string &p, int s=32);
bool FGetsLine(string *out, int s=1024);

}; // namespace LFL

#include "net/http.h"

#endif // LFL_CORE_APP_NETWORK_H__
