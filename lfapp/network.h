/*
 * $Id: network.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef __LFL_LFAPP_NETWORK_H__
#define __LFL_LFAPP_NETWORK_H__

#include "lfapp/wire.h"

namespace LFL {

DECLARE_bool(dns_dump);
DECLARE_bool(network_debug);

struct IOVec { char *buf; int len; };

struct IPV4Endpoint {
    IPV4::Addr addr=0; int port=0;
    IPV4Endpoint() {}
    IPV4Endpoint(int A, int P) : addr(A), port(P) {};
    string name() const { return IPV4::Text(addr, port); }
    string ToString() const { string s; s.resize(sizeof(*this)); memcpy((char*)s.data(), this, sizeof(*this)); return s; }
    static const IPV4Endpoint *FromString(const char *s) { return (const IPV4Endpoint *)s; }
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
    unordered_map<Socket, int> socket;
    fd_set rfds, wfds, xfds;
    SocketSet *mirror=0;

    int Select(int wait_time);
    void Del(Socket fd)                    { socket.erase(fd);  if (mirror) mirror->Del(fd); }
    void Add(Socket fd, int flag, void *v) { socket[fd] = flag; if (mirror) mirror->Add(fd, flag, v); }
    void Set(Socket fd, int flag, void *v) { socket[fd] = flag; if (mirror) mirror->Set(fd, flag, v); }
    int Get(Socket fd, fd_set *set) { return FD_ISSET(fd, set); } 
    int GetReadable(Socket fd) { return Get(fd, &rfds); }
    int GetWritable(Socket fd) { return Get(fd, &wfds); }
    int GetException(Socket fd) { return Get(fd, &xfds); }
    string DebugString() const {
        string ret="SelectSocketSet={";
        for (unordered_map<Socket, int>::const_iterator i = socket.begin(); i != socket.end(); ++i) StrAppend(&ret, i->first, ", ");
        return StrCat(ret.substr(0, ret.size()-2), "}");
    }
};

struct SelectSocketThread : public SocketSet {
    mutex *frame_mutex, *wait_mutex;
    SelectSocketSet sockets;
    mutex sockets_mutex;
    Thread thread;
    int pipe[2];
    SelectSocketThread(mutex *FM=0, mutex *WM=0) : frame_mutex(FM), wait_mutex(WM),
        thread(bind(&SelectSocketThread::ThreadProc, this)) { pipe[0] = pipe[1] = -1; }
    ~SelectSocketThread() { close(pipe[0]); close(pipe[1]); }

    void Add(Socket s, int f, void *v) { { ScopedMutex m(sockets_mutex); sockets.Add(s, f, v); } Wakeup(); }
    void Set(Socket s, int f, void *v) { { ScopedMutex m(sockets_mutex); sockets.Set(s, f, v); } Wakeup(); }
    void Del(Socket s)                 { { ScopedMutex m(sockets_mutex); sockets.Del(s);       } Wakeup(); }
    void Start();
    void Wait() { Wakeup(); thread.Wait(); }
    void Wakeup() { char c=0; if (pipe[1] >= 0) CHECK_EQ((int)write(pipe[1], &c, 1), 1); }
    void ThreadProc();
};

#if defined(LFL_EPOLL) && LFL_LINUX_SERVER
#include <sys/epoll.h>
template <int S> struct EPollSocketSet : public SocketSet {
    Socket epollfd, cur_fd; int cur_event=-1, num_events=0; struct epoll_event events[S];
    EPollSocketSet() : epollfd(epoll_create(S)) {}
    virtual ~EPollSocketSet() { close(epollfd); }

    int Select(int wait_time) {
        if ((num_events = epoll_wait(epollfd, events, S, ToMilliSeconds(wait_time))) == -1) ERROR("epoll_wait() ", strerror(errno));
        return 0;
    }
    void Change(Socket fd, int op, int flag, void *val) {
        struct epoll_event ev; memzero(ev); ev.data.ptr = val;
        ev.events = ((flag & READABLE) ? EPOLLIN : 0) | ((flag & WRITABLE) ? EPOLLOUT : 0);
        if (epoll_ctl(epollfd, op, fd, &ev) == -1) ERROR("epoll_ctl(", epollfd, ", ", op, ", ", events, "): ", strerror(errno)); 
    }
    void Del(Socket fd)                      { Change(fd, EPOLL_CTL_DEL, READABLE|WRITABLE, 0);   }
    void Add(Socket fd, int flag, void *val) { Change(fd, EPOLL_CTL_ADD, flag,              val); }
    void Set(Socket fd, int flag, void *val) { Change(fd, EPOLL_CTL_MOD, flag,              val); }
    int GetReadable(Socket fd) { return Get(fd)->events & (EPOLLIN  | EPOLLERR | EPOLLHUP); }
    int GetWritable(Socket fd) { return Get(fd)->events & (EPOLLOUT | EPOLLERR | EPOLLHUP); }
    int GetException(Socket fd) { return 0; }
    struct epoll_event *Get(Socket fd) {
        CHECK_EQ(fd, cur_fd);
        CHECK(cur_event >= 0 && cur_event < S);
        return &events[cur_event];
    }
};
#define LFL_EPOLL_SOCKET_SET
#define LFLSocketSet EPollSocketSet<65536*6>
#else
#define LFLSocketSet SelectSocketSet
#endif

struct Network : public Module {
    int select_time=0;
    LFLSocketSet active;
    vector<Service*> service_table;

    int Init();
    int Enable(Service *svc);
    int Disable(Service *svc);
    int Shutdown(Service *svc);
    int Enable(const vector<Service*> &svc);
    int Disable(const vector<Service*> &svc);
    int Shutdown(const vector<Service*> &svc);
    int Frame(unsigned);
    void AcceptFrame(Service *svc, Listener *listener);
    void TCPConnectionFrame(Service *svc, Connection *c, ServiceEndpointEraseList *removelist);
    void UDPConnectionFrame(Service *svc, Connection *c, ServiceEndpointEraseList *removelist, const string &epk);

    void ConnClose(Service *svc, Connection *c, ServiceEndpointEraseList *removelist);
    void ConnCloseAll(Service *svc);

    void EndpointRead(Service *svc, const char *name, const char *buf, int len);
    void EndpointClose(Service *svc, Connection *c, ServiceEndpointEraseList *removelist, const string &epk);
    void EndpointCloseAll(Service *svc);

    void UpdateActive(Connection *c);
};

struct SystemNetwork {
    static Socket OpenSocket(int protocol);
    static int SetSocketBlocking(Socket fd, int blocking);
    static int SetSocketBroadcastEnabled(Socket fd, int enabled);
    static int SetSocketReceiveBufferSize(Socket fd, int size);
    static int GetSocketReceiveBufferSize(Socket fd);

    static int Bind(int fd, IPV4::Addr addr, int port);
    static Socket Listen(int protocol, IPV4::Addr addr, int port);
    static int Connect(Socket fd, IPV4::Addr addr, int port, int *connected);
    static int SendTo(Socket fd, IPV4::Addr addr, int port, const char *buf, int len);
    static int GetSockName(Socket fd, IPV4::Addr *addr_out, int *port_out);
    static int GetPeerName(Socket fd, IPV4::Addr *addr_out, int *port_out);
    static string GetHostByAddr(IPV4::Addr addr);
    static IPV4::Addr GetHostByName(const string &host);
    static bool EWouldBlock();
    static string LastError();
};

struct Query {  
    virtual ~Query() {}
    virtual int Heartbeat(Connection *c) { return 0; }
    virtual int Connected(Connection *c) { return 0; }
    virtual int Read(Connection *c) { return 0; }    
    virtual int Flushed(Connection *c) { return 0; }
    virtual void Close(Connection *c) {}
};

struct Listener {
    BIO *ssl;
    Service *svc;
    Socket socket;
    typed_ptr self_reference;
    Listener(Service *s, bool ssl=false) : svc(s), ssl((BIO*)ssl), socket(-1), self_reference(TypePointer(this)) {}
};

struct Connection {
    const static int BufSize = 16384;
    enum { Connected=1, Connecting=2, Reconnect=3, Error=5 };

    Service *svc;
    Socket socket;
    Time ct, rt, wt;
    string endpoint_name;
    bool readable=1, writable=0;
    IPV4::Addr addr, src_addr=0;
    int state, port, src_port=0, rl=0, wl=0;
    char rb[BufSize], wb[BufSize];
    typed_ptr self_reference;
    vector<IOVec> packets;
    SSL *ssl=0;
    BIO *bio=0;
    Query *query;

    ~Connection() { delete query; }
    Connection(Service *s, Query *q)                                       : svc(s), socket(-1),   ct(Now()), rt(Now()), wt(Now()), addr(0),    state(Error), port(0),    self_reference(TypePointer(this)), query(q) {}
    Connection(Service *s, int State, int Sock)                            : svc(s), socket(Sock), ct(Now()), rt(Now()), wt(Now()), addr(0),    state(State), port(0),    self_reference(TypePointer(this)), query(0) {}
    Connection(Service *s, int State, IPV4::Addr Addr, int Port)           : svc(s), socket(-1),   ct(Now()), rt(Now()), wt(Now()), addr(Addr), state(State), port(Port), self_reference(TypePointer(this)), query(0) {}
    Connection(Service *s, int State, int Sock, IPV4::Addr Addr, int Port) : svc(s), socket(Sock), ct(Now()), rt(Now()), wt(Now()), addr(Addr), state(State), port(Port), self_reference(TypePointer(this)), query(0) {}

    string Name() const { return !endpoint_name.empty() ? endpoint_name : IPV4::Text(addr, port); }
    void SetError() { state = Error; ct = Now(); }
    void SetConnected() { state = Connected; ct = Now(); }
    void SetReconnect() { state = Reconnect; ct = Now(); }
    void SetConnecting() { state = Connecting; ct = Now(); }
    int SetSourceAddress() { return SystemNetwork::GetSockName(socket, &src_addr, &src_port); }
    int Write(const string &buf) { return Write(buf.c_str(), buf.size()); }
    int Write(const char *buf, int len);
    int WriteFlush();
    int WriteFlush(const string &buf) { return WriteFlush(buf.c_str(), buf.size()); }
    int WriteFlush(const char *buf, int len);
    int SendTo(const char *buf, int len);
    int Read();
    int ReadPacket();
    int ReadPackets();
    int Add(const char *buf, int len);
    int AddPacket(const char *buf, int len);
    int ReadFlush(int len);
};

struct Service {
    typedef map<string, Listener*> ListenMap;
    typedef map<Socket, Connection*> ConnMap;
    typedef map<string, Connection*> EndpointMap;

    string name;
    int protocol, reconnect=0;
    bool initialized=0, heartbeats=0, endpoint_read_autoconnect=0;
    void *game_network=0;
    ListenMap listen;
    ConnMap conn;
    Connection fake;
    IPV4EndpointSource *connect_src_pool=0;
    EndpointMap endpoint;
    Service(int prot=Protocol::TCP) : protocol(prot), fake(this, Connection::Connected, 0) {}

    void QueueListen(IPV4::Addr addr, int port, bool SSL=false) { listen[IPV4Endpoint(addr,port).ToString()] = new Listener(this, SSL); }
    Listener *GetListener() { return listen.size() ? listen.begin()->second : 0; }

    int OpenSocket(Connection *c, int protocol, int blocking, IPV4EndpointSource*);
    Socket Listen(IPV4::Addr addr, int port, Listener*);
    Connection *Accept(int state, Socket socket, IPV4::Addr addr, int port);
    Connection *Connect(IPV4::Addr addr, int port, IPV4EndpointSource *src_addr=0);
    Connection *Connect(IPV4::Addr addr, int port, IPV4::Addr src_addr, int src_port);
    Connection *Connect(const char *hostport);
    Connection *SSLConnect(SSL_CTX *sslctx, IPV4::Addr addr, int port);
    Connection *SSLConnect(SSL_CTX *sslctx, const char *hostport);
    Connection *EndpointConnect(const string &endpoint_name);
    void EndpointReadCB(string *endpoint_name, string *packet);
    void EndpointRead(const string &endpoint_name, const char *buf, int len);
    void EndpointClose(const string &endpoint_name);

    virtual void Close(Connection *c);
    virtual int UDPFilter(Connection *e, const char *buf, int len) { return 0; }
    virtual int Connected(Connection *c) { return 0; }
    virtual int Frame() { return 0; }
};

struct ServiceEndpointEraseList {
    vector<pair<Service*, Socket> > sockets;
    vector<pair<Service*, string> > endpoints;
    void AddSocket  (Service *s, Socket fd)        { sockets  .emplace_back(s, fd); }
    void AddEndpoint(Service *s, const string &en) { endpoints.emplace_back(s, en); }
    void Erase() {
        for (auto &r : endpoints) r.first->endpoint.erase(r.second);
        for (auto &r : sockets)   r.first->conn    .erase(r.second);
        sockets  .clear();
        endpoints.clear();
    }
};

struct NetworkThread {
    struct Service : public LFL::Service {};
    struct Query : public LFL::Query {
        void HandleMessage(Callback *cb) { (*cb)(); delete cb; }
        int Read(Connection *c);
    };

    Network *net;
    Connection *rd, *wr;
    unique_ptr<Thread> thread;
    NetworkThread(Network *N);

    void Write(Callback *x) { CHECK_EQ(sizeof(x), wr->WriteFlush(reinterpret_cast<const char*>(&x), sizeof(x))); }
    void HandleMessagesLoop() { while (GetLFApp()->run) { net->Frame(0); } }
};

struct UDPClient : public Service {
    static const int MTU = 1500;
    enum { Write=1, Sendto=2 };
    UDPClient() : Service(Protocol::UDP) { heartbeats=true; }

    typedef function<void(Connection*, const char*, int)> ResponseCB;
    Connection *PersistentConnection(const string &url, ResponseCB cb, int default_port) { return PersistentConnection(url, cb, HeartbeatCB(), default_port); }

    typedef function<void(Connection*)> HeartbeatCB; 
    Connection *PersistentConnection(const string &url, ResponseCB, HeartbeatCB, int default_port);
};

struct UDPServer : public Service {
    virtual ~UDPServer() {}
    Query *query=0;
    UDPServer(int port) { protocol=Protocol::UDP; QueueListen(0, port); }
    virtual int Connected(Connection *c) { c->query = query; return 0; }
};

struct HTTPClient : public Service {
    typedef function<void(Connection*, const char*, const string&, const char*, int)> ResponseCB;
    static int request(Connection *c, int method, const char *host, const char *path, const char *postmime, const char *postdata, int postlen, bool persist);

    bool WGet(const string &url, File *out=0, ResponseCB responseCB=ResponseCB());
    bool WPost(const string &url, const string &mimetype, const char *postdata, int postlen, ResponseCB=ResponseCB());
    Connection *PersistentConnection(const string &url, string *hostOut, string *pathOut, ResponseCB responseCB);
};

struct HTTPServer : public Service {
    virtual ~HTTPServer() { ClearURL(); }
    HTTPServer(IPV4::Addr addr, int port, bool SSL) { protocol=Protocol::TCP; QueueListen(addr, port, SSL); }
    HTTPServer(                 int port, bool SSL) { protocol=Protocol::TCP; QueueListen(0,    port, SSL); }
    int Connected(Connection *c);

    typedef function<void(Connection*)> ConnectionClosedCB;
    static void connectionClosedCB(Connection *httpServerConnection, ConnectionClosedCB cb);

    struct Method {
        enum { GET=1, POST=2 }; int x;
        static const char *name(int n) {
            if      (n == GET)  return "GET";
            else if (n == POST) return "POST";
            return 0;
        }
    };

    struct Response {
        int code, content_length;
        const char *type, *content;
        string type_buf, content_buf;
        Query *refill; bool write_headers;

        Response(     const string *T, const string      *C) : code(200), content_length(C->size()), type_buf(*T), content_buf(*C), refill(0), write_headers(1)  { content=content_buf.c_str(); type=type_buf.c_str(); }
        Response(       const char *T, const string      *C) : code(200), content_length(C->size()), type(T),      content_buf(*C), refill(0), write_headers(1)  { content=content_buf.c_str(); }
        Response(       const char *T, const char        *C) : code(200), content_length(strlen(C)), type(T),      content(C),      refill(0), write_headers(1)  {}
        Response(       const char *T, const StringPiece &C) : code(200), content_length(C.len),     type(T),      content(C.buf),  refill(0), write_headers(1)  {}
        Response(int K, const char *T, const StringPiece &C) : code(K),   content_length(C.len),     type(T),      content(C.buf),  refill(0), write_headers(1)  {}
        Response(int K, const char *T, const char        *C) : code(K),   content_length(strlen(C)), type(T),      content(C),      refill(0), write_headers(1)  {}
        Response(const char *T, int L, Query *R, bool WH=1)  : code(200), content_length(L),         type(T),      content(0),      refill(R), write_headers(WH) {}

        static Response _400;
    };

    struct Resource {
        virtual ~Resource() {}
        virtual Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) = 0;
    };

    typedef map<string, Resource*> URLMap;
    URLMap urlmap;

    void AddURL(const string &url, Resource *resource) { urlmap[url] = resource; }
    void ClearURL() { for (URLMap::iterator i = urlmap.begin(); i != urlmap.end(); i++) delete (*i).second; }

    virtual Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
        Resource *requested = FindOrNull(urlmap, url);
        if (!requested) return Response::_400;
        return requested->Request(c, method, url, args, headers, postdata, postlen);
    }

    struct DebugResource : public Resource {
        Response Request(Connection *c, int, const char *url, const char *args, const char *hdrs, const char *postdata, int postlen) {
            INFO("url: %s", url);
            INFO("args: %s", args);
            INFO("hdrs: %s", hdrs);
            return Response::_400;
        }
    };

    struct StringResource : public Resource {
        Response val;
        StringResource(const char *T, const StringPiece &C) : val(T, C) {}
        StringResource(const char *T, const char        *C) : val(T, C) {}
        Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) { return val; }
    };

    struct FileResource : public Resource {
        string filename; int size;
        const char *type;

        FileResource(const char *fn, const char *mimetype=0) : filename(fn), size(0) {
            type = mimetype ? mimetype : "application/octet-stream";
            LocalFile f(filename, "r");
            if (!f.Opened()) return;
            size = f.Size();
        }
        Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
    };

    struct ConsoleResource : public Resource {
        HTTPServer::Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
    };

#ifdef LFL_FFMPEG
    struct StreamResource : public Resource {
        AVFormatContext *fctx;
        typedef map<void*, Connection*> SubscriberMap;
        SubscriberMap subscribers;
        bool open; int abr, vbr;

        AVStream *audio;
        AVFrame *samples;
        short *sample_data;
        AudioResampler resampler;
        int frame, channels, resamples_processed;

        AVStream *video;
        AVFrame *picture; SwsContext *conv;

        virtual ~StreamResource();
        StreamResource(const char *outputFileType, int audioBitRate, int videoBitRate);        
        Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
        void openStreams(bool audio, bool video);
        void update(int audio_samples, bool video_sample);
        void resampleAudio(int audio_samples);
        void sendAudio();
        void sendVideo();        
        void broadcast(AVPacket *, unsigned long long timestamp);
    };
#endif

    struct SessionResource : public Resource {
        virtual ~SessionResource() { clear_connmap(); }

        typedef map<Connection*, Resource*> ConnMap;
        ConnMap connmap;
        void clear_connmap() { for (ConnMap::iterator i = connmap.begin(); i != connmap.end(); i++) delete (*i).second; }

        Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
            Resource *resource;
            if (!(resource = FindOrNull(connmap, c))) {
                resource = open();
                connmap[c] = resource;
                connectionClosedCB(c, bind(&SessionResource::ConnectionClosedCB, this, _1));
            }
            return resource->Request(c, method, url, args, headers, postdata, postlen);
        }
        void ConnectionClosedCB(Connection *c) {
            ConnMap::iterator i = connmap.find(c);
            if (i == connmap.end()) return;
            Resource *resource = (*i).second;
            connmap.erase(i);
            close(resource);
        }

        virtual Resource *open() = 0;
        virtual void close(Resource *) = 0;
    };
};

struct SMTPClient : public Service {
    long long total_connected=0, total_disconnected=0, delivered=0, failed=0;
    map<IPV4::Addr, string> domains; string domain;
    string HeloDomain(IPV4::Addr addr) const { return domain.empty() ? FindOrDie(domains, addr) : domain; }

    virtual int Connected(Connection *c) { total_connected++; return 0; }

    static void DeliverDeferred(Connection *c);
    typedef function<bool(Connection*, const string&, SMTP::Message*)> DeliverableCB;
    typedef function<void(Connection*, const SMTP::Message &, int, const string&)> DeliveredCB;
    Connection *DeliverTo(IPV4::Addr mx, IPV4EndpointSource*, DeliverableCB deliverableCB, DeliveredCB deliveredCB);
};

struct SMTPServer : public Service {
    long long total_connected=0;
    map<IPV4::Addr, string> domains; string domain;
    string HeloDomain(IPV4::Addr addr) const { return domain.empty() ? FindOrDie(domains, addr) : domain; }

    SMTPServer(const string &n) : domain(n) {}
    virtual int Connected(Connection *c);
    virtual void ReceiveMail(Connection *c, const SMTP::Message &message);
};

struct GPlusClient : public Service {
    static const int MTU = 1500;
    enum { Write=1, Sendto=2 };
    GPlusClient() : Service(Protocol::GPLUS) { heartbeats=true; }
    Connection *PersistentConnection(const string &name, UDPClient::ResponseCB cb, UDPClient::HeartbeatCB HCB);
};

struct GPlusServer : public Service {
    virtual ~GPlusServer() {}
    Query *query=0;
    GPlusServer() : Service(Protocol::GPLUS) { endpoint_read_autoconnect=1; }
    virtual int Connected(Connection *c) { c->query = query; return 0; }
};

struct Sniffer {
    static void PrintDevices(vector<string> *out);
    static void GetDeviceAddressSet(set<IPV4::Addr> *out);
    static void GetBroadcastAddress(IPV4::Addr *out);
    static void GetIPAddress(IPV4::Addr *out);
    typedef function<void(const char*, int, int)> CB;
    static Sniffer *Open(const string &dev, const string &filter, int snaplen, CB cb);

    Thread thread;
    CB cb;
    int ip, mask;
    void *handle;
    Sniffer(void *H, int I, int M, CB C) : cb(C), handle(H), ip(I), mask(M) {}
    ~Sniffer() { thread.Wait(); }
    void Threadproc();
};

struct GeoResolution {
    static GeoResolution *Open(const string &db);
    bool resolve(const string &addr, string *country, string *region, string *city, float *lat, float *lng);
    GeoResolution(void *I) : impl(I) {}
    void *impl;
};

}; // namespace LFL
#endif // __LFL_LFAPP_NETWORK_H__
