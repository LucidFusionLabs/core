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

#ifdef LFL_OPENSSL
#include "openssl/bio.h"
#include "openssl/ssl.h"
#include "openssl/err.h"
#endif

#ifdef WIN32
#include <WinDNS.h>
#else
#include <sys/socket.h>
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
const IPV4::Addr IPV4::ANY = INADDR_ANY;

DEFINE_bool(dns_dump, 0, "Print DNS responses");
DEFINE_bool(network_debug, 0, "Print send()/recv() bytes");
DEFINE_int(udp_idle_sec, 15, "Timeout UDP connections idle for seconds");
#ifdef LFL_OPENSSL
DEFINE_string(ssl_certfile, "", "SSL server certificate file");
DEFINE_string(ssl_keyfile, "", "SSL server key file");
SSL_CTX *lfapp_ssl = 0;
#endif

const char *Protocol::Name(int p) {
    if      (p == TCP)   return "TCP";
    else if (p == UDP)   return "UDP";
    else if (p == GPLUS) return "GPLUS";
    else return "";
}

void IPV4::ParseCSV(const string &text, vector<IPV4::Addr> *out) {
    vector<string> addrs; IPV4::Addr addr;
    Split(text, iscomma, &addrs);
    for (int i = 0; i < addrs.size(); i++) {
        if ((addr = Network::addr(addrs[i])) == INADDR_NONE) FATAL("unknown addr ", addrs[i]);
        out->push_back(addr);
    }
}

void IPV4::ParseCSV(const string &text, set<IPV4::Addr> *out) {
    vector<string> addrs; IPV4::Addr addr;
    Split(text, iscomma, &addrs);
    for (int i = 0; i < addrs.size(); i++) {
        if ((addr = Network::addr(addrs[i])) == INADDR_NONE) FATAL("unknown addr ", addrs[i]);
        out->insert(addr);
    }
}

string IPV4::MakeCSV(const vector<IPV4::Addr> &in) {
    string ret;
    for (vector<Addr>::const_iterator i = in.begin(); i != in.end(); ++i) StrAppend(&ret, ret.size()?",":"", IPV4Endpoint::name(*i));
    return ret;
}

string IPV4::MakeCSV(const set<IPV4::Addr> &in) {
    string ret;
    for (set<Addr>::const_iterator i = in.begin(); i != in.end(); ++i) StrAppend(&ret, ret.size()?",":"", IPV4Endpoint::name(*i));
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
        if (BitField::LastClear((const unsigned char*)source_ports[i].data(), source_ports[i].size()) != -1) return true;
    return false;
}

void IPV4EndpointPool::Close(IPV4::Addr addr, int port) {
    if (!source_addrs.size()) return;
    int bit = port - 1024;
    for (int i=0; i<source_addrs.size(); i++) if (source_addrs[i] == addr) {
        if (!BitField::Get((const unsigned char*)source_ports[i].data(), bit))
            ERROR("IPV4EndpointPool: Close unopened endpoint: ", IPV4Endpoint::name(addr, port));

        BitField::Clear((unsigned char*)source_ports[i].data(), bit);
        return;
    }
    ERROR("IPV4EndpointPool: Close unknown endpoint: ", IPV4Endpoint::name(addr, port));
}

void IPV4EndpointPool::Get(IPV4::Addr addr, int *port) {
    *port=0;
    if (!source_addrs.size()) return;
    for (int i=0; i<source_addrs.size(); i++) if (source_addrs[i] == addr) { GetPort(i, port); return; }
    ERROR("IPV4EndpointPool: address full: ", IPV4Endpoint::name(addr));
}

void IPV4EndpointPool::Get(IPV4::Addr *addr, int *port) {
    *addr=0; *port=0;
    if (!source_addrs.size()) return;
    for (int i=0, max_retries=10; i<max_retries; i++) {
        int ind = ::rand() % source_addrs.size();
        *addr = source_addrs[ind];
        if (GetPort(ind, port)) return;
    }
    ERROR("IPV4EndpointPool: full");
}

bool IPV4EndpointPool::GetPort(int ind, int *port) {
    int zero_bit = BitField::FirstClear((const unsigned char *)source_ports[ind].data(), source_ports[ind].size());
    if (zero_bit == -1) return false;
    *port = 1025 + zero_bit;
    BitField::Set((unsigned char*)source_ports[ind].data(), zero_bit);
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

int Network::Init() {
#ifdef LFL_OPENSSL
    SSL_load_error_strings();
    SSL_library_init(); 

    if (bool client_only=0) lfapp_ssl = SSL_CTX_new(SSLv23_client_method());
    else                    lfapp_ssl = SSL_CTX_new(SSLv23_method());

    if (!lfapp_ssl) FATAL("no SSL_CTX: ", ERR_reason_error_string(ERR_get_error()));
    SSL_CTX_set_verify(lfapp_ssl, SSL_VERIFY_NONE, 0);

    if (FLAGS_ssl_certfile.size() && FLAGS_ssl_keyfile.size()) {
        if (!SSL_CTX_use_certificate_file(lfapp_ssl, FLAGS_ssl_certfile.c_str(), SSL_FILETYPE_PEM)) { ERROR("SSL_CTX_use_certificate_file ", ERR_reason_error_string(ERR_get_error())); return -1; }
        if (!SSL_CTX_use_PrivateKey_file(lfapp_ssl, FLAGS_ssl_keyfile.c_str(), SSL_FILETYPE_PEM)) { ERROR("SSL_CTX_use_PrivateKey_file ",  ERR_reason_error_string(ERR_get_error())); return -1; }
        if (!SSL_CTX_check_private_key(lfapp_ssl)) { ERROR("SSL_CTX_check_private_key ", ERR_reason_error_string(ERR_get_error())); return -1; }
    }
#endif
    return 0;
}

int Network::Shutdown(const vector<Service*> &s) { int ret = 0; for (int i = 0; i < s.size(); i++) if (Shutdown(s[i]) < 0) ret = -1; return ret; }
int Network::Disable (const vector<Service*> &s) { int ret = 0; for (int i = 0; i < s.size(); i++) if (Disable (s[i]) < 0) ret = -1; return ret; }
int Network::Enable  (const vector<Service*> &s) { int ret = 0; for (int i = 0; i < s.size(); i++) if (Enable  (s[i]) < 0) ret = -1; return ret; }
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
    ServiceTableIter(service_table) { if (service_table[i] == s) return 0; }
    service_table.push_back(s);
    return 0;
}

int Network::Disable(Service *s) {
    ServiceTableIter(service_table) {
        if (service_table[i] != s) continue;
        service_table.erase(service_table.begin() + i);
        return 0;
    }
    return -1;
}

void Network::ConnClose(Service *svc, Connection *c, vector<Socket> *removelist) {
    if (c->query) c->query->Close(c);
    svc->Close(c);
    if (removelist) removelist->push_back(c->socket);
    delete c;
}
void Network::ConnCloseAll(Service *svc) {
    for (Service::ConnMap::iterator i = svc->conn.begin(); i != svc->conn.end(); ++i)
        ConnClose(svc, i->second, 0);
    svc->conn.clear();
}

void Network::EndpointRead(Service *svc, const char *name, const char *buf, int len) { return svc->EndpointRead(name, buf, len); }
void Network::EndpointClose(Service *svc, Connection *c, vector<string> *removelist, const string &epk) {
    if (c->query) c->query->Close(c);
    if (svc->listen.empty()) svc->Close(c);
    if (removelist) removelist->push_back(epk);
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

int Network::Frame() {
    Service *svc; Socket fd; void **v;
    int listener_type = Typed::Id<Listener>(), connection_type = Typed::Id<Connection>();

    /* pre loop */
    ServiceTableIter(service_table) {
        if (!(svc = service_table[i])) break;
        vector<Socket> removelist;
        vector<string> removelist2;

        /* select */
        if (svc->active.Select(svc->select_time))
        { ERROR("SocketSet.select: ", Connection::lasterror()); return -1; }

#ifdef LFL_EPOLL_SOCKET_SET
        /* iterate events */
        for (svc->active.cur_event = 0; svc->active.cur_event < svc->active.num_events; svc->active.cur_event++) {
            typed_ptr *tp = (typed_ptr*)svc->active.events[svc->active.cur_event].data.ptr;
            if      (tp->type == connection_type) { Connection *c=(Connection*)tp->value; svc->active.cur_fd = c->socket; service_frame_tcp_connection(svc, c, &removelist); }
            else if (tp->type == listener_type)   { Listener   *l=(Listener*)  tp->value; svc->active.cur_fd = l->socket; service_frame_accept(svc, l); }
            else FATAL("unknown type", tp->type);
        }
#else
        /* answer listening sockets & UDP server read */
        for (Service::ListenMap::iterator iter = svc->listen.begin(); iter != svc->listen.end(); ++iter) 
            if (svc->active.GetReadable(iter->second->socket)) AcceptFrame(svc, iter->second);

        /* iterate connections */
        for (Service::ConnMap::iterator iter = svc->conn.begin(); iter != svc->conn.end(); ++iter)
            TCPConnectionFrame(svc, (*iter).second, &removelist);
#endif
        /* iterate endpoints */
        for (Service::EndpointMap::iterator iter = svc->endpoint.begin(); iter != svc->endpoint.end(); iter++) {
            UDPConnectionFrame(svc, (*iter).second, &removelist2, (*iter).first);
        }

        /* remove closed */
        for (vector<string>::iterator i = removelist2.begin(); i != removelist2.end(); i++) svc->endpoint.erase(*i);
        for (vector<Socket>::iterator i = removelist.begin(); i != removelist.end(); i++) svc->conn.erase(*i);
        removelist.clear();
        removelist2.clear();

        if (svc->heartbeats) { /* connection heartbeats */
            for (Service::ConnMap::iterator i = svc->conn.begin(); i != svc->conn.end(); i++) {
                Connection *c = (*i).second; int ret = 0;
                if (c->query) ret = c->query->Heartbeat(c);
                if (c->state == Connection::Error || ret < 0) ConnClose(svc, c, &removelist);
            }

            for (Service::EndpointMap::iterator i = svc->endpoint.begin(); i != svc->endpoint.end(); i++) {
                Connection *c = (*i).second; int ret = 0;
                if (c->query) ret = c->query->Heartbeat(c);
                if (c->state == Connection::Error || ret < 0) EndpointClose(svc, c, &removelist2, c->endpoint_name);
            }

            for (vector<string>::iterator i = removelist2.begin(); i != removelist2.end(); i++) svc->endpoint.erase(*i);
            for (vector<Socket>::iterator i = removelist.begin(); i != removelist.end(); i++) svc->conn.erase(*i);
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
            struct sockaddr_in sin; int sinSize=sizeof(sin); Socket socket;
            if (BIO_do_accept(listener->ssl) <= 0) continue;
            BIO *bio = BIO_pop(listener->ssl);
            BIO_get_fd(bio, &socket);

            if (::getpeername(socket, (struct sockaddr *)&sin, (socklen_t*)&sinSize) < 0)
            { if (!Connection::ewouldblock()) ERROR("getpeername: ", strerror(errno)); break; }

            /* insert socket */
            c = svc->Accept(Connection::Connected, socket, sin.sin_addr.s_addr, ntohs(sin.sin_port));
            if (!c) { close(socket); continue; }
            c->bio = bio;
            BIO_set_nbio(c->bio, 1);
            BIO_get_ssl(c->bio, &listener->ssl);
#endif
        }
        else if (svc->protocol == Protocol::UDP) { /* UDP server read */
            struct sockaddr_in sin; int sinSize=sizeof(sin), inserted=0, ret; char buf[2048];
            int len = recvfrom(listener->socket, buf, sizeof(buf)-1, 0, (struct sockaddr*)&sin, (socklen_t*)&sinSize);
            if (len <= 0) {
                if (Connection::ewouldblock()) break;
                else { ERROR("recvfrom: ", Connection::lasterror().c_str()); break; }
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

            if ((ret = c->addpacket(buf, len)) != len) 
            { ERROR(c->name(), ": addpacket(", len, ")"); c->_error(); continue; }
            if (!inserted) continue;
        }
        else { /* TCP accept */
            struct sockaddr_in sin; int sinSize=sizeof(sin); Socket socket;
            if ((int)(socket = ::accept(listener->socket, (struct sockaddr *)&sin, (socklen_t*)&sinSize)) < 0)
            { if (!Connection::ewouldblock()) ERROR("accept: ", Connection::lasterror()); break; }

            /* insert socket */
            c = svc->Accept(Connection::Connected, socket, sin.sin_addr.s_addr, ntohs(sin.sin_port));
            if (!c) { close(socket); continue; }
        }
        if (!c) continue;

        /* connected 1 */
        c->set_source_address();
        INFO(c->name(), ": incoming connection (socket=", c->socket, ")");
        if (svc->Connected(c) < 0) c->_error();
        if (c->query) { if (c->query->Connected(c) < 0) { ERROR(c->name(), ": query connected"); c->_error(); } }
    }
}

void Network::TCPConnectionFrame(Service *svc, Connection *c, vector<Socket> *removelist) {
    /* complete connecting sockets */
    if (c->state == Connection::Connecting && svc->active.GetWritable(c->socket)) {
        int res=0, resSize=sizeof(res);
        if (!getsockopt(c->socket, SOL_SOCKET, SO_ERROR, (char*)&res, (socklen_t*)&resSize) && !res) {

            /* connected 2 */ 
            c->connected();
            c->set_source_address();
            INFO(c->name(), ": connected");
            if (svc->Connected(c) < 0) c->_error();
            Service::UpdateActive(c);
            if (c->query) { if (c->query->Connected(c) < 0) { ERROR(c->name(), ": query connected"); c->_error(); } }
            return;
        }

        errno = res;
        INFO(c->name(), ": connect failed: ", c->lasterror());
        c->_error();
    }

    /* IO communicate */
    else if (c->state == Connection::Connected) do {
        if (svc->protocol == Protocol::UDP) {
            if (svc->listen.empty() && svc->active.GetReadable(c->socket)) { /* UDP Client Read */
                if (c->readpackets()<0) { c->_error(); break; }
            }
            if (c->packets.size()) {
                if (c->query) { if (c->query->Read(c) < 0) { ERROR(c->name(), ": query UDP read"); c->_error(); } }
                c->packets.clear();
                c->readflush(c->rl);
            }
        }
        else if (c->ssl || svc->active.GetReadable(c->socket)) { /* TCP Read */
            if (c->read()<0) { c->_error(); break; }
            if (c->rl) {
                if (c->query) { if (c->query->Read(c) < 0) { ERROR(c->name(), ": query read"); c->_error(); } }
            }
        }

        if (c->wl && svc->active.GetWritable(c->socket)) {
            if (c->writeflush()<0) { c->_error(); break; }
            if (!c->wl) {
                c->writable = 0;
                if (c->query) { if (c->query->Flushed(c) < 0) { ERROR(c->name(), ": query flushed"); c->_error(); } }
                Service::UpdateActive(c);
            }
        }
    } while(0);

    /* error */
    if (c->state == Connection::Error) ConnClose(svc, c, removelist);
}

void Network::UDPConnectionFrame(Service *svc, Connection *c, vector<string> *removelist, const string &epk) {
    if (c->state == Connection::Connected && c->packets.size()) {
        if (c->query) { if (c->query->Read(c) < 0) { ERROR(c->name(), ": query UDP read"); c->_error(); } }
        c->packets.clear();
        c->readflush(c->rl);
    }

    bool timeout; /* Timeout or error */
    if (c->state == Connection::Error || (timeout = (c->rt + Seconds(FLAGS_udp_idle_sec)) <= Now())) {
        INFO(c->name(), ": ", timeout ? "timeout" : "error");
        EndpointClose(svc, c, removelist, epk);
    }
}

Socket Network::socket_open(int protocol) {
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

int Network::socket_blocking(Socket fd, int blocking) {
#ifdef _WIN32
    u_long ioctlarg = !blocking ? 1 : 0;
    if (ioctlsocket(fd, FIONBIO, &ioctlarg) < 0) return -1;
#else
    if (fcntl(fd, F_SETFL, !blocking ? O_NONBLOCK : 0) == -1) return -1;
#endif
    return 0;
}

int Network::broadcast_enabled(Socket fd, int optval) {
    if (setsockopt(fd, SOL_SOCKET, SO_BROADCAST, (const char*)&optval, sizeof(optval)) == -1)
    { ERROR("setsockopt: ", Connection::lasterror()); return -1; }
    return 0;
}

IPV4::Addr Network::addr(const string &ip) { return inet_addr(ip.c_str()); }
IPV4::Addr Network::resolve(const string &host) {
    struct hostent *h;
    struct in_addr a;

    a.s_addr = Network::addr(host);
    if (a.s_addr != INADDR_NONE) return (int)a.s_addr;

    h = gethostbyname(host.c_str());
    if (h && h->h_length == 4) return *(int *)h->h_addr_list[0];

    ERROR("Network::resolve ", host);
    return -1;
}

int Network::bind(int fd, IPV4::Addr addr, int port) {
    sockaddr_in sin; int optval = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval, sizeof(optval)) == -1)
    { ERROR("setsockopt: ", Connection::lasterror()); return -1; }

    memset(&sin, 0, sizeof(sockaddr_in));
    sin.sin_family = PF_INET;
    sin.sin_port = htons(port);
    sin.sin_addr.s_addr = addr ? addr : INADDR_ANY;

    if (FLAGS_network_debug) INFO("bind(", fd, ", ", IPV4Endpoint::name(addr, port), ")");
    if (SystemBind(fd, (const sockaddr *)&sin, (socklen_t)sizeof(sockaddr_in)) == -1)
    { ERROR("bind: ", Connection::lasterror()); return -1; }

    return 0;
}

Socket Network::listen(int protocol, IPV4::Addr addr, int port) {
    Socket fd;
    if ((fd = Network::socket_open(protocol)) < 0) 
    { ERROR("network_socket_open: ", Connection::lasterror()); return -1; }

    if (Network::bind(fd, addr, port) == -1) { close(fd); return -1; }

    if (protocol == Protocol::TCP) {
        if (::listen(fd, 32) == -1)
        { ERROR("listen: ", Connection::lasterror()); close(fd); return -1; }
    }

    if (Network::socket_blocking(fd, 0))
    { ERROR("Network::socket_blocking: ", Connection::lasterror()); close(fd); return -1; }

    INFO("listen(port=", port, ", protocol=", (protocol == Protocol::TCP) ? "TCP" : "UDP", ")");
    return fd;
}

int Network::connect(Socket fd, IPV4::Addr addr, int port, int *connected) {
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(struct sockaddr_in));
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);
    sin.sin_addr.s_addr = addr;

    if (FLAGS_network_debug) INFO("connect(", fd, ", ", IPV4Endpoint::name(addr, port), ")");
    int ret = ::connect(fd, (struct sockaddr *)&sin, sizeof(struct sockaddr_in));
    if (ret == -1 && !Connection::ewouldblock())
    { ERROR("connect(", IPV4Endpoint::name(addr, port), "): ", Connection::lasterror()); return -1; }

    if (connected) *connected = !ret;
    return 0;
}

int Network::sendto(Socket fd, IPV4::Addr addr, int port, const char *buf, int len) {
    sockaddr_in sin; int sinSize=sizeof(sin);
    sin.sin_family = PF_INET;
    sin.sin_addr.s_addr = addr;
    sin.sin_port = htons(port);
    return ::sendto(fd, buf, len, 0, (struct sockaddr*)&sin, sinSize);
}

int Network::getpeername(Socket fd, IPV4::Addr *addr_out, int *port_out) {
    struct sockaddr_in sin; int sinSize=sizeof(sin);
    if (::getpeername(fd, (struct sockaddr *)&sin, (socklen_t*)&sinSize) < 0)
    { ERROR("getpeername: ", strerror(errno)); return -1; }
    *addr_out = sin.sin_addr.s_addr;
    *port_out = ntohs(sin.sin_port);
    return 0;
}

int Network::getsockname(Socket fd, IPV4::Addr *addr_out, int *port_out) {
    struct sockaddr_in sin; int sinSize=sizeof(sin);
    if (::getsockname(fd, (struct sockaddr *)&sin, (socklen_t*)&sinSize) < 0)
    { ERROR("getsockname: ", strerror(errno)); return -1; }
    *addr_out = sin.sin_addr.s_addr;
    *port_out = ntohs(sin.sin_port);
    return 0;
}

string Network::gethostbyaddr(IPV4::Addr addr) {
#if defined(_WIN32) || defined(LFL_ANDROID)
    struct hostent *h = ::gethostbyaddr((const char *)&addr, sizeof(addr), PF_INET);
#else
    struct hostent *h = ::gethostbyaddr((const void *)&addr, sizeof(addr), PF_INET);
#endif
    return h ? h->h_name : "";
}

void SelectSocketThread::ThreadProc() {
    while (app->run) {
        { ScopedMutex sm(app->frame_mutex); }
        if (app->run) {
            SelectSocketSet my_sockets;
            { ScopedMutex sm(sockets_mutex); my_sockets = sockets; }
            my_sockets.Select(-1);
            if (my_sockets.GetReadable(pipe[0])) NBRead(pipe[0], 4096);
        }
        if (app->run) app->Wakeup();
        { ScopedMutex sm(app->wait_mutex); }
    }
}

int SelectSocketSet::Select(int wait_time) {
    int maxfd=-1, rc=0, wc=0, xc=0;
    struct timeval tv = Time2timeval(wait_time);
    FD_ZERO(&rfds); FD_ZERO(&wfds); FD_ZERO(&xfds);
    for (unordered_map<Socket, int>::iterator i = socket.begin(); i != socket.end(); ++i) {
        bool added = 0;
        if (i->second & READABLE)  { rc++; FD_SET(i->first, &rfds); added = 1; }
        if (i->second & WRITABLE)  { wc++; FD_SET(i->first, &wfds); added = 1; }
        if (i->second & EXCEPTION) { xc++; FD_SET(i->first, &xfds); added = 1; }
        if (added && i->first > maxfd) maxfd = i->first;
    }
    if (!rc && !wc && !xc) { Msleep(ToMilliSeconds(wait_time)); return 0; }
    if ((select(maxfd+1, rc?&rfds:0, wc?&wfds:0, xc?&xfds:0, wait_time >= 0 ? &tv : 0)) == -1)
    { ERROR("select: ", Connection::lasterror()); return -1; }
    return 0;
}

int DNS::WriteRequest(unsigned short id, const string &querytext, unsigned short type, char *out, int len) {
    Serializable::MutableStream os(out, len);
    Header *hdr = (Header*)os.Get(Header::size);
    memset(hdr, 0, Header::size);
    hdr->rd = 1;
    hdr->id = id;
    hdr->qdcount = htons(1);

    StringWordIter words(querytext.c_str(), 0, isdot);
    for (string word = BlankNull(words.Next()); !word.empty(); word = BlankNull(words.Next())) {
        CHECK_LT(word.size(), 64);
        os.Write8((unsigned char)word.size());
        os.String(word);
    }
    os.Write8((char)0);

    os.Htons(type);                      // QueryTypeClass.Type
    os.Htons((unsigned short)Class::IN); // QueryTypeClass.QClass
    return os.error ? -1 : os.offset;
}

int DNS::ReadResponse(const char *buf, int bufsize, Response *res) {
    Serializable::ConstStream is(buf, bufsize);
    const Serializable::Stream *in = &is;
    const Header *hdr = (Header*)in->Get(Header::size);

    int qdcount = ntohs(hdr->qdcount);
    int ancount = ntohs(hdr->ancount);
    int nscount = ntohs(hdr->nscount);
    int arcount = ntohs(hdr->arcount);

    for (int i = 0; i < qdcount; i++) {
        Record out; int len;
        if ((len = DNS::ReadString(in->Start(), in->Get(), in->End(), &out.question)) < 0 || !in->Advance(len + 4)) return -1;
        res->Q.push_back(out);
    }

    if (DNS::ReadResourceRecord(in, ancount, &res->A)  < 0) return -1;
    if (DNS::ReadResourceRecord(in, nscount, &res->NS) < 0) return -1;
    if (DNS::ReadResourceRecord(in, arcount, &res->E)  < 0) return -1;
    return 0;
}

int DNS::ReadResourceRecord(const Serializable::Stream *in, int num, vector<Record> *out) {
    for (int i = 0; i < num; i++) {
        Record rec; int len; unsigned short rrlen;
        if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.question)) < 0 || !in->Advance(len)) return -1;

        in->Ntohs(&rec.type);
        in->Ntohs(&rec._class);
        in->Ntohs(&rec.ttl1);
        in->Ntohs(&rec.ttl2);
        in->Ntohs(&rrlen);

        if (rec._class == Class::IN && rec.type == Type::A) {
            if (rrlen != 4) return -1;
            in->Read32(&rec.addr);
        } else if (rec._class == Class::IN && (rec.type == Type::NS || rec.type == Type::CNAME)) {
            if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.answer)) != rrlen   || !in->Advance(len)) return -1;
        } else if (rec._class == Class::IN && rec.type == Type::MX) {
            in->Ntohs(&rec.pref);
            if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.answer)) != rrlen-2 || !in->Advance(len)) return -1;
        } else {
            ERROR("unhandled type=", rec.type, ", class=", rec._class);
            in->Advance(rrlen);
            continue;
        }
        out->push_back(rec);
    }
    return in->error ? -1 : 0;
}

int DNS::ReadString(const char *start, const char *cur, const char *end, string *out) {
    if (!cur) { ERROR("DNS::ReadString null input"); return -1; }
    if (out) out->clear();
    const char *cur_start = cur, *final = 0;
    for (unsigned char len = 1; len && cur < end; cur += len+1) {
        len = *cur;
        if (len >= 64) { // Pointer to elsewhere in packet
            int offset = ntohs(*(unsigned short*)cur) & ~(3<<14);
            if (!final) final = cur + 2;
            cur = start + offset - 2;
            if (cur < start || cur >= end) { ERROR("OOB cur ", (void*)start, " ", (void*)cur, " ", (void*)end); return -1; }
            len = 1;
            continue;
        }
        if (out) StrAppend(out, out->empty() ? "" : ".", string(cur+1, len));
    }
    if (out) *out = tolower(*out);
    if (final) cur = final;
    return (cur > end) ? -1 : (cur - cur_start);
}

void DNS::MakeAnswerMap(const vector<DNS::Record> &in, AnswerMap *out) {
    for (int i = 0; i < in.size(); ++i) {
        const DNS::Record &e = in[i];
        if (e.question.empty() || !e.addr) continue;
        (*out)[e.question].push_back(e.addr);
    }
    for (int i = 0; i < in.size(); ++i) {
        const DNS::Record &e = in[i];
        if (e.question.empty() || e.answer.empty() || e.type != DNS::Type::CNAME) continue;
        AnswerMap::const_iterator a = out->find(e.answer);
        if (a == out->end()) continue;
        VectorAppend((*out)[e.question], a->second.begin(), a->second.end());
    }
}

void DNS::MakeAnswerMap(const vector<DNS::Record> &in, const AnswerMap &qmap, int type, AnswerMap *out) {
    for (int i = 0; i < in.size(); ++i) {
        const DNS::Record &e = in[i];
        if (e.type != type) continue;
        AnswerMap::const_iterator q_iter = qmap.find(e.answer);
        if (e.question.empty() || e.answer.empty() || q_iter == qmap.end())
        { ERROR("DNS::MakeAnswerMap missing ", e.answer); continue; }
        VectorAppend((*out)[e.question], q_iter->second.begin(), q_iter->second.end());
    }
}

string DNS::Response::DebugString() const {
    string ret;
    StrAppend(&ret, "Question ",   Q .size(), "\n"); for (int i = 0; i < Q .size(); ++i) StrAppend(&ret, Q [i].DebugString(), "\n");
    StrAppend(&ret, "Answer ",     A .size(), "\n"); for (int i = 0; i < A .size(); ++i) StrAppend(&ret, A [i].DebugString(), "\n");
    StrAppend(&ret, "NS ",         NS.size(), "\n"); for (int i = 0; i < NS.size(); ++i) StrAppend(&ret, NS[i].DebugString(), "\n");
    StrAppend(&ret, "Additional ", E .size(), "\n"); for (int i = 0; i < E .size(); ++i) StrAppend(&ret, E [i].DebugString(), "\n");
    return ret;
}

/* HTTP */

bool HTTP::host(const char *host, const char *host_end, string *hostO, string *portO) {
    const char *colon = strstr(host, ":"), *port = 0;
    if (!host_end) host_end = host + strlen(host);
    if (colon && colon < host_end) port = colon+1;
    if (hostO) hostO->assign(host, port ? port-host-1 : host_end-host);
    if (portO) portO->assign(port ? port : "", port ? host_end-port : 0);
    return 1;
}

bool HTTP::host(const char *hostname, const char *host_end, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl) {
    string h, p;
    if (!host(hostname, host_end, &h, &p)) return 0;
    return resolve(h, p, ipv4_addr, tcp_port, ssl);
}

bool HTTP::resolve(const string &host, const string &port, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int default_port) {
    if (ipv4_addr) {
        *ipv4_addr = Network::resolve(host);
        if (*ipv4_addr == -1) { ERROR("resolve"); return 0; }
    }
    if (tcp_port) {
        *tcp_port = !port.empty() ? atoi(port.c_str()) : (default_port ? default_port : (ssl ? 443 : 80));
        if (*tcp_port < 0 || *tcp_port >= 65536) { ERROR("oob port"); return 0; }
    }
    return 1;
}

bool HTTP::URL(const char *url, string *protO, string *hostO, string *portO, string *pathO) {
    static char protHdr[] = "://";
    const char *prot = url, *prot_end = strstr(prot, protHdr), *host;
    if (prot_end) host = prot_end + strlen(protHdr);
    else { prot=0; host = url; }

    while (prot && *prot && isspace(*prot)) prot++;
    while (host && *host && isspace(*host)) host++;

    const char *host_end = strstr(host, "/");
    HTTP::host(host, host_end, hostO, portO);

    if (protO) protO->assign(prot ? prot : "", prot ? prot_end-prot : 0);
    if (pathO) pathO->assign(host_end ? host_end+1 : "");
    return 1;
}

bool HTTP::URL(const char *url, bool *ssl, IPV4::Addr *ipv4_addr, int *tcp_port, string *host, string *path, int default_port, string *prot) {
    string my_prot, port, my_host, my_path; bool my_ssl;
    if (!prot) prot = &my_prot;
    if (!host) host = &my_host;
    if (!path) path = &my_path;
    if (!ssl) ssl = &my_ssl;

    HTTP::URL(url, prot, host, &port, path);
    *ssl = !prot->empty() && !strcasecmp(prot->c_str(), "https");
    if (!prot->empty() && strcasecmp(prot->c_str(), "http") && !*ssl) return 0;
    if (host->empty()) { ERROR("no host or path"); return 0; }
    if (!HTTP::resolve(*host, port, ipv4_addr, tcp_port, *ssl, default_port)) { ERROR("HTTP::URL resolve ", host); return 0; }
    return 1;
}

string HTTP::HostURL(const char *url) {
    string my_prot, my_port, my_host, my_path;
    HTTP::URL(url, &my_prot, &my_host, &my_port, &my_path);
    string ret = !my_prot.empty() ? StrCat(my_prot, "://") : "http://";
    if (!my_host.empty()) ret += my_host;
    if (!my_port.empty()) ret += string(":") + my_port;
    return ret;
}

int HTTP::request(char *buf, char **methodO, char **urlO, char **argsO, char **verO) {
    char *url, *ver, *args;
    if (!(url = (char*)nextchar(buf, isspace)))    return -1;    *url = 0;
    if (!(url = (char*)nextchar(url+1, notspace))) return -1;
    if (!(ver = (char*)nextchar(url, isspace)))    return -1;    *ver = 0;
    if (!(ver = (char*)nextchar(ver+1, notspace))) return -1;

    if ((args = strchr(url, '?'))) *args++ = 0;

    if (methodO) *methodO = buf;
    if (urlO) *urlO = url;
    if (argsO) *argsO = args;
    if (verO) *verO = ver;
    return 0;
}

char *HTTP::headersStart(char *buf) {
    char *start = strstr(buf, "\r\n");
    if (!start) return 0;
    *start = 0;
    return start + 2;
}

char *HTTP::headerEnd(char *buf) {
    char *end = strstr(buf, "\r\n\r\n");
    if (!end) return 0;
    *(end+2) = 0;
    return end + 2;
}

const char *HTTP::headerEnd(const char *buf) {
    const char *end = strstr(buf, "\r\n\r\n");
    if (!end) return 0;
    return end + 2;
}

int HTTP::headerLen(const char *beg, const char *end) { return end - beg + 2; }

int HTTP::headerNameLen(const char *beg) {
    const char *n = beg;
    while (*n && !isspace(*n) && *n != ':') n++;
    return *n == ':' ? n - beg : 0;
}

int HTTP::argNameLen(const char *beg) {
    const char *n = beg;
    while (*n && !isspace(*n) && *n != '=' && *n != '&') n++;
    return n - beg;
}

#define DefineGrepArgs(k, kl, v) \
    va_list ap; va_start(ap, num); \
    char **k = (char **)alloca(num*sizeof(char*)); \
    int *kl = (int *)alloca(num*sizeof(int)); \
    StringPiece **v = (StringPiece **)alloca(num*sizeof(char*)); \
    for (int i=0; i<num; i++) { \
        k[i] = va_arg(ap, char*); \
        kl[i] = strlen(k[i]); \
        v[i] = va_arg(ap, StringPiece*); \
    } \
    va_end(ap);

int HTTP::argGrep(const char *args, const char *end, int num, ...) {
    DefineGrepArgs(k, kl, v);
    if (!end) end = args + strlen(args);

    int alen=end-args, anlen;
    StringWordIter words(args, alen, isand, 0, StringWordIter::Flag::InPlace);
    for (const char *a = words.Next(); a; a = words.Next()) {
        if (!(anlen = HTTP::argNameLen(a))) continue;
        for (int i=0; i<num; i++) if (anlen == kl[i] && !strncasecmp(k[i], a, anlen)) {
            if (*(a+anlen) && *(a+anlen) == '=') v[i]->assign(a+anlen+1, words.wordlen-anlen-1);
            else v[i]->assign(a, words.wordlen);
        }
    }
    return 0;
}

int HTTP::headerGrep(const char *headers, const char *end, int num, ...) {
    DefineGrepArgs(k, kl, v);
    if (!end) end = HTTP::headerEnd(headers);
    if (!end) end = headers + strlen(headers);

    int hlen=end-headers, hnlen;
    StringLineIter lines(headers, hlen, StringLineIter::Flag::InPlace);
    for (const char *h = lines.Next(); h; h = lines.Next()) {
        if (!(hnlen = HTTP::headerNameLen(h))) continue;
        for (int i=0; i<num; i++) if (hnlen == kl[i] && !strncasecmp(k[i], h, hnlen)) {
            const char *hv = nextchar(h+hnlen+1, notspace, lines.linelen-hnlen-1);
            if (!hv) v[i]->clear();
            else     v[i]->assign(hv, lines.linelen-(hv-h));
        }
    }
    return 0;
}

string HTTP::headerGrep(const char *headers, const char *end, const string &name) {
    if (!end) end = HTTP::headerEnd(headers);
    if (!end) end = headers + strlen(headers);

    int hlen=end-headers, hnlen;
    StringLineIter lines(headers, hlen, StringLineIter::Flag::InPlace);
    for (const char *line = lines.Next(); line; line = lines.Next()) {
        if (!(hnlen = HTTP::headerNameLen(line))) continue;
        if (hnlen == name.size() && !strncasecmp(name.c_str(), line, hnlen)) return string(line+hnlen+2, lines.linelen-hnlen-2);
    }
    return "";
}

bool HTTP::connectionClose(const char *connectionHeaderValue) {
    static const char close[] = "close\r\n";
    static const int closelen = strlen(close);
    return !strncmp(connectionHeaderValue, close, closelen);
}

string HTTP::encodeURL(const char *url) {
    static const char encodeURIcomponentPass[] = "~!*()'";
    static const char encodeURIPass[] = "./@#:?,;-_&";
    string ret;
    for (const unsigned char *p = (const unsigned char *)url; *p; p++) {
        if      (*p >= '0' && *p <= '9') ret += *p;
        else if (*p >= 'a' && *p <= 'z') ret += *p;
        else if (*p >= 'A' && *p <= 'Z') ret += *p;
        else if (strchr(encodeURIcomponentPass, *p)) ret += *p;
        else if (strchr(encodeURIPass, *p)) ret += *p;
        else StringAppendf(&ret, "%%%02x", *p);
    }
    return ret;
}

/* SMTP */

void SMTP::HTMLMessage(const string& from, const string& to, const string& subject, const string& content, string *out) {
    static const char seperator[] = "XzYzZy";
    *out = StrCat("From: ", from, "\nTo: ", to, "\nSubject: ", subject,
                  "\nMIME-Version: 1.0\nContent-Type: multipart/alternative; boundary=\"", seperator, "\"\n\n");
    StrAppend(out, "--", seperator, "\nContent-type: text/html\n\n", content, "\n--", seperator, "--\n");
}

void SMTP::NativeSendmail(const string &message) {
#ifdef __linux__
    Process smtp;
    const char *argv[] = { "/usr/bin/sendmail", "-i", "-t", 0 };
    if (smtp.Open(argv)) return;
    fwrite(message.c_str(), message.size(), 1, smtp.out);
#endif
}

string SMTP::EmailFrom(const string &message) {
    int lt, gt;
    string mail_from = HTTP::headerGrep(message.c_str(), 0, "From");
    if ((lt = mail_from.find("<"    )) == mail_from.npos ||
        (gt = mail_from.find(">", lt)) == mail_from.npos) FATAL("parse template from ", mail_from);
    return mail_from.substr(lt+1, gt-lt-1);
}

/* Connection */

int Connection::read() {
    int readlen = sizeof(rb)-1-rl, len = 0;
    if (readlen <= 0) { ERROR(name(), ": read queue full, rl=", rl); return -1; }

    if (ssl) {
#ifdef LFL_OPENSSL
        if ((len = BIO_read(bio, rb+rl, readlen)) <= 0) {
            if (SSL_get_error(ssl, len) != SSL_ERROR_WANT_READ) {
                const char *err_string = ERR_reason_error_string(ERR_get_error());
                ERROR(name(), ": BIO_read: ", err_string ? err_string : "read() zero");
                return -1;
            }
            return 0;
        }
#endif
    }
    else {
        if ((len = recv(socket, rb+rl, readlen, 0)) <= 0) {
            if      (!len)                      ERROR(name(), ": read() zero");
            else if (len < 0 && !ewouldblock()) ERROR(name(), ": read(): ", Connection::lasterror());
            return -1;
        }
    }

    rl += len; 
    rb[rl] = 0;
    rt = Now();
    if (FLAGS_network_debug) INFO("read(", socket, ", ", len, ", '", rb+rl-len, "')");
    return len;
}

int Connection::readpacket() {
    int ret = read();
    if (ret <= 0) return ret;

    IOVec pkt = { rb+rl-ret, ret };
    packets.push_back(pkt);
    rl++;
    return ret;
}

int Connection::readpackets() {
    int ret=1;
    while (ret > 0) {
        ret = readpacket();
        if (ret < 0 && !ewouldblock()) return ret;
    }
    return 0;
}

int Connection::add(const char *buf, int len) {
    int readlen = sizeof(rb)-1-rl;
    if (readlen < len) { ERROR(name(), ": read packet queue full"); return -1; }

    memcpy(rb+rl, buf, len);
    rl += len;
    rb[rl] = 0;
    rt = Now();
    if (FLAGS_network_debug) INFO("add(", socket, ", ", len, ", '", rb+rl-len, "')");
    return len;
}

int Connection::addpacket(const char *buf, int len) {
    int ret = add(buf, len);
    if (ret <= 0) return ret;

    IOVec pkt = { rb+rl-ret, ret };
    packets.push_back(pkt);
    rl++;
    rt = Now();
    if (FLAGS_network_debug) INFO("addpacket(", name(), ", ", len, ")");
    return ret;
}

int Connection::write(const char *buf, int len) {
    if (!buf || len<0) return -1;
    if (!len) len = strlen(buf);
    if (wl+len > sizeof(wb)-1) { ERROR(name(), ": write queue full"); return -1; }

    if (!wl && len) {
        writable = true;
        Service::UpdateActive(this);
    }
    memcpy(wb+wl, buf, len);
    wl += len;
    wb[wl] = 0;
    wt = Now();
    return len;
}

int Connection::readflush(int len) {
    if (len<0) return -1;
    if (!len) return rl;
    if (rl-len < 0) { ERROR(name(), ": read queue underflow: ", len, " > ", rl); return -1; }
 
    if (rl!=len) memmove(rb, rb+len, rl-len);
    rl -= len;
    rb[rl] = 0;
    return rl;
}

int Connection::writeflush(const char *buf, int len) {
    int wrote = 0;
    if (ssl) {
#ifdef LFL_OPENSSL
        if ((wrote = BIO_write(bio, buf, len)) < 0) {
            if (!ewouldblock()) { ERROR(name(), ": send: ", strerror(errno)); return -1; }
            wrote = 0;
        }
#endif
    }
    else {
        if ((wrote = send(socket, buf, len, 0)) < 0) {
            if (!ewouldblock()) { ERROR(name(), ": send: ", strerror(errno)); return -1; }
            wrote = 0;
        }
    }
    if (FLAGS_network_debug) INFO("write(", socket, ", ", wrote, ", '", buf, "')");
    return wrote;
}

int Connection::writeflush() {
    int wrote = writeflush(wb, wl);
    if (wrote && wrote!=wl) memmove(wb, wb+wrote, wl-wrote);
    wl -= wrote;
    wb[wl] = 0;
    return wrote;
}

int Connection::sendto(const char *buf, int len) { return Network::sendto(socket, addr, port, buf, len); }

bool Connection::ewouldblock() {
#ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == WSAEINPROGRESS;
#else
    return errno == EAGAIN || errno == EINPROGRESS;
#endif
};

string Connection::lasterror() {
#ifdef _WIN32
    return StrCat(WSAGetLastError());
#else
    return strerror(errno);
#endif
}

/* Service */

void Service::Close(Connection *c) {
    active.Del(c->socket);
    if (select_socket_thread) select_socket_thread->DelSocket(c->socket);
    close(c->socket);
    if (connect_src_pool && (c->src_addr || c->src_port)) connect_src_pool->Close(c->src_addr, c->src_port);
}

int Service::OpenSocket(Connection *c, int protocol, int blocking, IPV4EndpointSource* src_pool) {
    Socket fd = Network::socket_open(protocol);
    if (fd == -1) return -1;

    if (!blocking) {
        if (Network::socket_blocking(fd, 0))
        { close(fd); return -1; }
    }

    if (src_pool) {
        IPV4Endpoint last_src;
        for (int i=0, max_bind_attempts=10; /**/; i++) {
            src_pool->Get(&c->src_addr, &c->src_port);
            if (i >= max_bind_attempts || (i && c->src_addr == last_src.addr && c->src_port == last_src.port))
            { ERROR("connect-bind ", IPV4Endpoint::name(c->src_addr, c->src_port), ": ", strerror(errno)); close(fd); return -1; }

            if (Network::bind(fd, c->src_addr, c->src_port) != -1) break;
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
        if (BIO_do_accept(listener->ssl) <= 0) { ERROR("ssl_listen: ", -1); return -1; }
        BIO_get_fd(listener->ssl, &listener->socket);
        BIO_set_accept_bios(listener->ssl, BIO_new_ssl(lfapp_ssl, 0));
#endif
    } else {
        if ((listener->socket = Network::listen(protocol, addr, port)) == -1)
        { ERROR("Network::listen(", protocol, ", ", port, "): ", Connection::lasterror().c_str()); return -1; }
    }
    active.Add(listener->socket, SocketSet::READABLE, &listener->self_reference);
    if (select_socket_thread) select_socket_thread->AddSocket(listener->socket, SocketSet::READABLE);
    return listener->socket;
}

Connection *Service::Accept(int state, Socket socket, IPV4::Addr addr, int port) {
    Connection *c = new Connection(this, state, socket, addr, port);
    conn[c->socket] = c;
    active.Add(c->socket, SocketSet::READABLE, &c->self_reference);
    if (select_socket_thread) select_socket_thread->AddSocket(c->socket, SocketSet::READABLE);
    return c;
}

Connection *Service::Connect(IPV4::Addr addr, int port, IPV4::Addr src_addr, int src_port) {
    SingleIPV4Endpoint src_pool(src_addr, src_port);
    return Connect(addr, port, &src_pool);
}

Connection *Service::Connect(IPV4::Addr addr, int port, IPV4EndpointSource *src_pool) {
    Connection *c = new Connection(this, Connection::Connecting, addr, port);
    if (Service::OpenSocket(c, protocol, 0, src_pool ? src_pool : connect_src_pool))
    { ERROR(c->name(), ": connecting: ", c->lasterror()); delete c; return 0; }

    int connected = 0;
    if (Network::connect(c->socket, c->addr, c->port, &connected) == -1) {
        ERROR(c->name(), ": connecting: ", c->lasterror());
        close(c->socket);
        delete c;
        return 0;
    }
    INFO(c->name(), ": connecting");
    conn[c->socket] = c;

    if (connected) {
        /* connected 3 */ 
        c->connected();
        c->set_source_address();
        INFO(c->name(), ": connected");
        if (this->Connected(c) < 0) c->_error();
        if (c->query) { if (c->query->Connected(c) < 0) { ERROR(c->name(), ": query connected"); c->_error(); } }
        active.Add(c->socket, (c->readable ? SocketSet::READABLE : 0), &c->self_reference);
        if (select_socket_thread) select_socket_thread->AddSocket(c->socket, (c->readable ? SocketSet::READABLE : 0));
    } else {
        active.Add(c->socket, SocketSet::READABLE|SocketSet::WRITABLE, &c->self_reference);
        if (select_socket_thread) select_socket_thread->AddSocket(c->socket, SocketSet::READABLE|SocketSet::WRITABLE);
    }
    return c;
}

Connection *Service::Connect(const char *hostport) {
    IPV4::Addr addr; int port;
    if (!HTTP::host(hostport, 0, &addr, &port, 0)) { ERROR("resolve ", hostport, " failed"); return 0; }
    return Connect(addr, port);
}

Connection *Service::SSLConnect(SSL_CTX *sslctx, const char *hostport) {
#ifdef LFL_OPENSSL
    if (!sslctx) sslctx = lfapp_ssl;
    if (!sslctx) { ERROR("no ssl: ", -1); return 0; }

    Connection *c = new Connection(this, Connection::Connecting, 0, 0);
    if (!HTTP::host(hostport, 0, &c->addr, &c->port, true)) { ERROR("resolve: ", hostport); return 0; }

    c->bio = BIO_new_ssl_connect(sslctx);
    BIO_set_conn_hostname(c->bio, hostport);
    BIO_get_ssl(c->bio, &c->ssl);
    BIO_set_nbio(c->bio, 1);

    int ret = BIO_do_connect(c->bio);
    if (ret < 0 && !BIO_should_retry(c->bio)) {
        ERROR(hostport, ": BIO_do_connect: ", ret);
        delete c;
        return 0;
    }
   
    BIO_get_fd(c->bio, &c->socket);

    INFO(c->name(), ": connecting (fd=", c->socket, ")");
    conn[c->socket] = c;
    active.Add(c->socket, SocketSet::WRITABLE, &c->self_reference);
    if (select_socket_thread) select_socket_thread->AddSocket(c->socket, SocketSet::WRITABLE);
    return c;
#else
    return 0;
#endif
}

Connection *Service::SSLConnect(SSL_CTX *sslctx, IPV4::Addr addr, int port) {
#ifdef LFL_OPENSSL
    if (!sslctx) sslctx = lfapp_ssl;
    if (!sslctx) { ERROR("no ssl: ", -1); return 0; }

    Connection *c = new Connection(this, Connection::Connecting, addr, port);
    c->bio = BIO_new_ssl_connect(sslctx);
    BIO_set_conn_ip(c->bio, (char*)&addr);
    BIO_set_conn_int_port(c->bio, (char*)&port);
    BIO_get_ssl(c->bio, &c->ssl);
    BIO_set_nbio(c->bio, 1);

    int ret = BIO_do_connect(c->bio);
    if (ret < 0 && !BIO_should_retry(c->bio)) {
        ERROR(c->name(), ": BIO_do_connect: ", ret);
        delete c;
        return 0;
    }
   
    BIO_get_fd(c->bio, &c->socket);

    INFO(c->name(), ": connecting (fd=", c->socket, ")");
    conn[c->socket] = c;
    active.Add(c->socket, SocketSet::WRITABLE, &c->self_reference);
    if (select_socket_thread) select_socket_thread->AddSocket(c->socket, SocketSet::WRITABLE);
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
    if (this->Connected(c) < 0) c->_error();
    INFO(Protocol::Name(protocol), "(", (void*)this, ") endpoint connect: ", endpoint_name);
    if (c->query) { if (c->query->Connected(c) < 0) { ERROR(c->name(), ": query connected"); c->_error(); } }
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
        if (!endpoint_read_autoconnect) { ERROR("unknown endpoint ", endpoint_name); return; }
        if (!EndpointConnect(endpoint_name)) { ERROR("endpoint_read_autoconnect ", endpoint_name); return; }
        ep = endpoint.find(endpoint_name);
        CHECK(ep != endpoint.end());
    }

    Connection *c = ep->second; int ret;
    if ((ret = c->addpacket(buf, len)) != len) 
    { ERROR(c->name(), ": addpacket(", len, ")"); c->_error(); return; }
}

void Service::EndpointClose(const string &endpoint_name) {
    INFO(Protocol::Name(protocol), "(", (void*)this, ") endpoint close: ", endpoint_name);
    Service::EndpointMap::iterator ep = endpoint.find(endpoint_name);
    if (ep != endpoint.end()) ep->second->_error();
}

void Service::UpdateActive(Connection *c) {
    if (FLAGS_network_debug) INFO(c->name(), " active = { ", c->readable?"READABLE":"", " , ", c->writable?"WRITABLE":"", " }");
    int flag = (c->readable?SocketSet::READABLE:0) | (c->writable?SocketSet::WRITABLE:0);
    c->svc->active.Set(c->socket, flag, &c->self_reference);
    if (c->svc->select_socket_thread) c->svc->select_socket_thread->SetSocket(c->socket, flag);
}

/* UDP Client */

struct UDPClientQuery {
    struct PersistentConnection : public Query {
        UDPClient::ResponseCB responseCB; UDPClient::HeartbeatCB heartbeatCB;
        PersistentConnection(UDPClient::ResponseCB RCB, UDPClient::HeartbeatCB HCB) : responseCB(RCB), heartbeatCB(HCB) {}

        int Heartbeat(Connection *c) { if (heartbeatCB) heartbeatCB(c); return 0; }
        void Close(Connection *c) { if (responseCB) responseCB(c, 0, 0); }
        int Read(Connection *c) {
            for (int i=0; i<c->packets.size() && responseCB; i++) {
                if (c->state != Connection::Connected) break;
                responseCB(c, c->packets[i].buf, c->packets[i].len);
            }
            return 0;
        }
    };
};

Connection *UDPClient::PersistentConnection(const string &url, ResponseCB responseCB, HeartbeatCB heartbeatCB, int default_port) {
    IPV4::Addr ipv4_addr; int udp_port;
    if (!HTTP::URL(url.c_str(), (bool*)0, &ipv4_addr, &udp_port, (string*)0, (string*)0, default_port))
    { INFO(url, ": connect failed"); return 0; }

    Connection *c = Connect(ipv4_addr, udp_port);
    if (!c) { INFO(url, ": connect failed"); return 0; }

    c->query = new UDPClientQuery::PersistentConnection(responseCB, heartbeatCB);
    return c;
}

/* Resolver */

void Resolver::Reset() {
    for (NameserverMap::iterator i = conn.begin(); i != conn.end(); ++i) delete i->second;
    conn.clear();
}

bool Resolver::Connected() {
    for (NameserverMap::const_iterator i = conn.begin(); i != conn.end(); ++i)
        if (i->second->c->state == Connection::Connected) return true;
    return false;
}

Resolver::Nameserver *Resolver::Connect(IPV4::Addr addr) {
    Nameserver *ns = new Nameserver(this, addr);
    if (!ns->c) { delete ns; return 0; }
    CHECK_EQ(addr, ns->c->addr);
    conn[addr] = ns;
    return ns;
}

Resolver::Nameserver *Resolver::Connect(const vector<IPV4::Addr> &addrs) {
    static bool randomize = false;
    int rand_connect_index = randomize ? (::rand() % addrs.size()) : 0, ri=0; Nameserver *ret=0;
    for (vector<IPV4::Addr>::const_iterator i = addrs.begin(); i != addrs.end(); ++i, ++ri) {
        if (ri == rand_connect_index) ret = Connect(*i);
        else conn_available.push_back(*i);
    } return ret;
}

bool Resolver::Resolve(const Request &req) {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    IPV4::Addr ipv4_addr = Network::resolve(req.query);
    INFO("resolved ", req.query, " to ", IPV4Endpoint::name(ipv4_addr));
    req.cb(ipv4_addr, NULL);
    return true;
#else
    if (!conn.size() && !conn_available.size()) { ERROR("resolve called with no conns"); return false; } 

    Nameserver *ns = 0; // Choose a nameserver
    NameserverMap::iterator ni = conn.begin();
    if (req.retrys || ni == conn.end() || ni->second->requestMap.size() >= max_outstanding_per_ns) {
        NameserverMap::iterator i = ni;
        if (i != conn.end()) ++i;
        for (/**/; i != conn.end(); ++i) if (i->second->requestMap.size() < max_outstanding_per_ns) break;
        if (i != conn.end()) ns = i->second;
        else if (conn_available.size()) {
            ns = Connect(conn_available.back());
            conn_available.pop_back();
        }
    }
    if (!ns && ni != conn.end() && ni->second->requestMap.size() < max_outstanding_per_ns) ns = ni->second;

    // Resolve or queue
    Request outreq(ns, req.query, req.type, req.cb, req.retrys);
    if (ns) return ns->Resolve(outreq);
    queue.push_back(outreq);
    return true;
#endif
}

void Resolver::DefaultNameserver(vector<IPV4::Addr> *nameservers) {
    nameservers->clear();
#ifdef LFL_ANDROID
    return;
#endif
#ifdef _WIN32
    IP4_ARRAY IP; DWORD size=sizeof(IP4_ARRAY);
    if (DnsQueryConfig(DnsConfigDnsServerList, 0, 0, 0, &IP, &size) || !IP.AddrCount) return;
    nameservers->push_back(IP.AddrArray[0]);
#else
    LocalFile file("/etc/resolv.conf", "r");
    if (!file.Opened()) return;

    for (const char *line = file.NextLine(); line; line = file.NextLine()) {
        StringWordIter words(line);
        if (strcmp(words.Next(), "nameserver")) continue;
        nameservers->push_back(Network::resolve(words.Next()));
    }
#endif
}

/* Resolver::Nameserver */

bool Resolver::Nameserver::Resolve(const Request &req) {
    int len; unsigned short id = NextID();
    if ((len = DNS::WriteRequest(id, req.query, req.type, c->wb, sizeof(c->wb))) < 0) return false;
    if (c->writeflush(c->wb, len) != len) return false;
    requestMap[id] = req;
    return true;
}

void Resolver::Nameserver::Response(Connection *cin, DNS::Header *hdr, int len) {
    CHECK_EQ(c, cin);
    if (!hdr) {
        ERROR(c->name(), ": nameserver closed, timedout=", timedout);
        CHECK_EQ(parent->conn.erase(c->addr), 1);
        if (timedout) parent->conn_available.push_back(c->addr);
        if (requestMap.size()) {
            bool alternatives = parent->conn.size() || parent->conn_available.size();
            for (RequestMap::iterator i = requestMap.begin(); i != requestMap.end(); ++i) {
                const Resolver::Request &req = i->second;
                if (!alternatives || !parent->Resolve(req)) { if (req.cb) req.cb(-1, 0); }
            }
        }
        delete this;
        return;
    }

    RequestMap::iterator rmiter = requestMap.find(hdr->id);
    if (rmiter == requestMap.end()) { ERROR(c->name(), ": unknown DNS reply id=", hdr->id, ", len=", len); return; }
    Resolver::Request req = rmiter->second;
    requestMap.erase(rmiter);

    DNS::Response res;
    if (DNS::ReadResponse((const char *)hdr, len, &res)) { ERROR(c->name(), ": parse "); return; }
    if (FLAGS_dns_dump) INFO(c->name(), ": ", res.DebugString());

    if (req.cb) {
        vector<IPV4::Addr> results;
        for (int i=0; i<res.A.size(); i++) if (res.A[i].type == DNS::Type::A) results.push_back(res.A[i].addr);
        IPV4::Addr ipv4_addr = results.size() ? results[::rand() % results.size()] : -1;
        INFO(c->name(), ": resolved ", req.query, " to ", IPV4Endpoint::name(ipv4_addr));
        req.cb(ipv4_addr, &res);
    }
    Dequeue();
}

void Resolver::Nameserver::Heartbeat() {
    Time now = Now();
    if (parent->auto_disconnect_seconds && !requestMap.size() && !parent->queue.size() && (c->rt + Seconds(parent->auto_disconnect_seconds)) <= now)
    { timedout=true; c->_error(); INFO(c->name(), ": nameserver timeout"); return; }

    static const int retry_interval = 1000, retry_max = 5;
    for (RequestMap::iterator rmiter = requestMap.begin(); rmiter != requestMap.end(); /**/) {
        if ((*rmiter).second.stamp + retry_interval >= now) { rmiter++; continue; }
        Resolver::Request req = (*rmiter).second;
        requestMap.erase(rmiter++);

        INFO(req.ns->c->name(), ": timeout resolving ", req.query, " (retrys=", req.retrys, ")");
        if (req.retrys++ >= retry_max || !parent->Resolve(req)) { if (req.cb) req.cb(-1, 0); }
    }
    Dequeue();
}

void Resolver::Nameserver::Dequeue() {
    while (parent->queue.size() && requestMap.size() < parent->max_outstanding_per_ns) {
        Resolver::Request req = parent->queue.back();
        parent->queue.pop_back();
        if (!parent->Resolve(req)) { if (req.cb) req.cb(-1, 0); }
    }
}

/* Recursive Resolver */

RecursiveResolver::RecursiveResolver() : queries_requested(0), queries_completed(0) {
    vector<IPV4::Addr> addrs;
#   define XX(x)
#   define YY(x) addrs.push_back(Network::addr(x));
#   include "lfapp/namedroot.h"
    root.resolver.Connect(addrs);
}

bool RecursiveResolver::Resolve(Request *req) {
    AuthorityTreeNode *node = GetAuthorityTreeNode(req->query, false);
    req->seen_authority.insert((void*)node);
    req->resolver = this;

    DNS::Response *cached = 0;
    AuthorityTreeNode::Cache::iterator ci;
    if      (req->type == DNS::Type::A  && (ci = node->Acache .find(req->query)) != node->Acache .end()) cached = ci->second;
    else if (req->type == DNS::Type::MX && (ci = node->MXcache.find(req->query)) != node->MXcache.end()) cached = ci->second;
    if (cached) {
        IPV4::Addr addr = cached->A.size() ? cached->A[::rand() % cached->A.size()].addr : -1;
        INFO("RecursiveResolver found ", req->query, " = ", IPV4Endpoint::name(addr), " in cache=", node->authority_domain);
        RunInMainThread(new Callback(bind(&Request::Complete, req, addr, cached)));
        return true;
    }

    Resolver::Request nsreq(req->query, req->type, bind(&Request::ResponseCB, req, _1, _2));
    bool ret = node->resolver.Resolve(nsreq);
    if (ret) queries_requested++;
    return ret;
}

int RecursiveResolver::ResolveMissing(Request *req, const vector<DNS::Record> &R, const DNS::AnswerMap *answer) {
    int start_requests = req->child_request.size(), start_pending_requests = req->pending_child_request.size();
    for (vector<DNS::Record>::const_iterator e = R.begin(); e != R.end(); ++e) {
        if (e->answer.empty() || (answer && Contains(*answer, e->answer))) continue;
        req->ChildResolve(new Request(e->answer, DNS::Type::A, Resolver::ResponseCB(), req));
    }
    int new_requests = req->child_request.size() - start_requests, new_pending_requests = req->pending_child_request.size() - start_pending_requests;
    if (new_requests || new_pending_requests) INFO("RecursiveResolver ", req->query, " spawned ", new_requests, " subqueries, queued ", new_pending_requests);
    return new_requests + new_pending_requests;
}

void RecursiveResolver::Response(Request *req, IPV4::Addr addr, DNS::Response *res, vector<DNS::Response> *subres) {
    if (FLAGS_dns_dump) INFO("RecursiveResolver::Response ", (int)addr, " ", res, " " , subres);
    if (addr != -1) {
        if (addr == 0 && !req->parent_request && res) {
            if (!req->missing_answer) {
                req->missing_answer = 1;
                req->answer.clear();
                req->answer.push_back(*res);
                DNS::AnswerMap extra;
                DNS::MakeAnswerMap(res->E, &extra);
                int new_child_requests = ResolveMissing(req, res->A, &extra);
                if (new_child_requests) return;
            } else if (subres) {
                for (int i = 1; i < subres->size(); ++i)
                    res->E.insert(res->E.end(), (*subres)[i].A.begin(), (*subres)[i].A.end());
            }
        }
        AuthorityTreeNode *node=0;
        if (res && (req->type == DNS::Type::A || req->type == DNS::Type::MX)) {
            node = GetAuthorityTreeNode(req->query, false);
            AuthorityTreeNode::Cache *cache = (req->type == DNS::Type::A) ? &node->Acache : &node->MXcache;
            if (Contains(*cache, req->query)) { ERROR("cache collision ", (*cache)[req->query]->DebugString(), " versus ", res->DebugString()); node=0; }
            else (*cache)[req->query] = new DNS::Response(*res);
        }
        INFO("RecursiveResolver resolved ", req->query, " to ", IPV4Endpoint::name(addr), " (cached=", node?node->authority_domain:"<NULL>", ")");
        req->Complete(addr, res);
        queries_completed++;
        return;
    }

    bool ret = false;
    DNS::AnswerMap extra, authority_zone;
    if (res) {
        DNS::MakeAnswerMap(res->E, &extra);
        for (int i = 1; subres && i < subres->size(); ++i) DNS::MakeAnswerMap((*subres)[i].A, &extra);
        DNS::MakeAnswerMap(res->NS, extra, DNS::Type::NS, &authority_zone);
        if (!authority_zone.size() && req->Ancestors() < 5 && !subres) {
            int new_child_requests = ResolveMissing(req, res->NS, 0);
            if (new_child_requests) { req->answer.clear(); req->answer.push_back(*res); return; }
        }
    }
    if (authority_zone.size() != 1) ERROR("authority_zone.size() ", authority_zone.size());
    for (DNS::AnswerMap::const_iterator i = authority_zone.begin(); i != authority_zone.end(); ++i) {
        AuthorityTreeNode *node = GetAuthorityTreeNode(i->first, true);
        CHECK_EQ(i->first, node->authority_domain);
        if (!node->authority.Q.size()) {
            node->authority = *res;
            node->resolver.Connect(i->second);
        } else ERROR("AuthorityTreeNode collision ", node->authority.DebugString(), " versus ", res->DebugString());

        if (Contains(req->seen_authority, (void*)node)) { ERROR("RecursiveResolver loop?"); continue; }
        ret = node->resolver.Resolve(Resolver::Request(req->query, req->type, bind(&Request::ResponseCB, req, _1, _2)));
        req->seen_authority.insert(node);
        break;
    }
    if (!ret) {
        INFO("RecursiveResolver failed to resolve ", req->query);
        req->Complete(-1, res);
        queries_completed++;
    }
}

RecursiveResolver::AuthorityTreeNode *RecursiveResolver::GetAuthorityTreeNode(const string &query, bool create) {
    AuthorityTreeNode *node = &root;
    vector<string> q;
    Split(query, isdot, &q);
    for (int i = q.size()-1; i >= 0; --i) {
        AuthorityTreeNode::Children::iterator it = node->child.find(q[i]);
        if (it != node->child.end()) { node = it->second; continue; }
        if (!create) break;

        AuthorityTreeNode *ret = new AuthorityTreeNode();
        ret->authority_domain = Join(q, ".", i, q.size()) + ".";
        ret->depth = node->depth + 1;
        node->child[q[i]] = ret;
        node = ret;
    }
    if (FLAGS_dns_dump) INFO("GetAuthorityTreeNode(", query, ", ", create, ") = ", node->authority_domain);
    return node;
}

void RecursiveResolver::Request::ChildResolve(Request *subreq) {
    if (child_request.size()) { pending_child_request.insert(subreq); return; }
    child_request.insert(subreq);
    resolver->Resolve(subreq);
}

void RecursiveResolver::Request::ChildResponse(Request *subreq, DNS::Response *res) {
    if (res) answer.push_back(*res);
    child_request.erase(subreq);
    if (child_request.size()) return;
    if (!pending_child_request.size()) {
        INFO(query, ": subrequests finished, ma=", missing_answer, ", as=", answer.size());
        return resolver->Response(this, missing_answer ? 0 : -1, &answer[0], &answer);
    }
    subreq = *pending_child_request.begin();
    pending_child_request.erase(pending_child_request.begin());
    ChildResolve(subreq);
}

void RecursiveResolver::Request::Complete(IPV4::Addr addr, DNS::Response *res) {
    if (parent_request) parent_request->ChildResponse(this, res);
    if (cb) cb(addr, res);
    delete this;
}

/* HTTPClient */

struct HTTPClientQuery {
    struct Protocol : public Query {
        int readHeaderLength, readContentLength, currentChunkLength, currentChunkRead;
        bool chunkedEncoding, fullChunkCBs;
        string content_type;

        Protocol(bool fullChunks) : fullChunkCBs(fullChunks) { reset(); }
        void reset() { readHeaderLength=0; readContentLength=0; currentChunkLength=0; currentChunkRead=0; chunkedEncoding=0; content_type.clear();  }

        int Read(Connection *c) {
            char *cur = c->rb;
            if (!readHeaderLength) {
                StringPiece ct, cl, te;
                char *headers = cur, *headersEnd = HTTP::headerEnd(headers);
                if (!headersEnd) return 1;

                readHeaderLength = HTTP::headerLen(headers, headersEnd);
                HTTP::headerGrep(headers, headersEnd, 3, "Content-Type", &ct, "Content-Length", &cl, "Transfer-Encoding", &te);
                currentChunkLength = readContentLength = atoi(BlankNull(cl.data()));
                chunkedEncoding = te.str() == "chunked";
                content_type = ct.str();

                Headers(c, headers, readHeaderLength);
                cur += readHeaderLength;
            }
            for (;;) {
                if (chunkedEncoding && !currentChunkLength) {
                    char *cur_in = cur;
                    cur += isnl(cur);
                    char *chunkHeader = cur;
                    if (!(cur = (char*)nextline(cur))) { cur=cur_in; break; }
                    currentChunkLength = strtoul(chunkHeader, 0, 16);
                }

                int rb_left = c->rl - (cur - c->rb);
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
            if (cur != c->rb) c->readflush(cur - c->rb);
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
            else c->query = this;
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

    int ret = c->write(hdr.data(), hdr.size());
    if (posthdr.empty()) return ret;
    if (ret != hdr.size()) return -1;
    return c->write(postdata, postlen);
}

bool HTTPClient::WGet(const string &url, File *out, ResponseCB cb) {
    bool ssl; int tcp_port; string host, path, prot;
    if (!HTTP::URL(url.c_str(), &ssl, 0, &tcp_port, &host, &path, 0, &prot)) {
        if (prot != "file") return 0;
        string fn = StrCat(!host.empty() ? "/" : "", host , "/", path), content = LocalFile::FileContents(fn);
        if (!content.empty() && cb) cb(0, 0, string(), content.data(), content.size());
        if (cb)                     cb(0, 0, string(), 0,              0);
        return true;
    }

    if (!out && !cb) {
        string fn = basename(path.c_str(),0,0);
        if (fn.empty()) fn = "index.html";
        out = new LocalFile(StrCat(dldir(), fn), "w");
        if (!out->Opened()) { ERROR("open file"); delete out; return 0; }
    }

    IPV4::Addr addr;
    HTTPClientQuery::WGet *query = new HTTPClientQuery::WGet(this, ssl, host, tcp_port, path, out, cb);
    if ((addr = Network::addr(host)) != INADDR_NONE) query->ResolverResponseCB(addr, 0);
    else if (!Singleton<Resolver>::Get()->Resolve(Resolver::Request(host, DNS::Type::A, bind(&HTTPClientQuery::WGet::ResolverResponseCB, query, _1, _2))))
    { ERROR("resolver: ", url); delete query; return 0; }
    return true;
}

bool HTTPClient::WPost(const string &url, const string &mimetype, const char *postdata, int postlen, ResponseCB cb) {
    bool ssl; int tcp_port; string host, path;
    if (!HTTP::URL(url.c_str(), &ssl, 0, &tcp_port, &host, &path)) return 0;

    HTTPClientQuery::WPost *query = new HTTPClientQuery::WPost(this, ssl, host, tcp_port, path, mimetype, postdata, postlen, cb);
    if (!Singleton<Resolver>::Get()->Resolve(Resolver::Request(host, DNS::Type::A, bind(&HTTPClientQuery::WGet::ResolverResponseCB, query, _1, _2))))
    { ERROR("resolver: ", url); delete query; return 0; }

    return true;
}

Connection *HTTPClient::PersistentConnection(const string &url, string *host, string *path, ResponseCB responseCB) {
    bool ssl; IPV4::Addr ipv4_addr; int tcp_port;
    if (!HTTP::URL(url.c_str(), &ssl, &ipv4_addr, &tcp_port, host, path)) return 0;

    Connection *c = 
#ifdef LFL_OPENSSL
        ssl ? SSLConnect(lfapp_ssl, ipv4_addr, tcp_port) : 
#endif
        Connect(ipv4_addr, tcp_port);

    if (!c) return 0;

    c->query = new HTTPClientQuery::PersistentConnection(responseCB);
    return c;
}

/* HTTPServer */

struct HTTPServerConnection : public Query {
    HTTPServer *server;
    bool persistent;
    Query *refill;

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
            if (c->rl < reqlen) return 0;
            int ret = httpserv->Dispatch(c, type, url, args, headers, postdata, postlen);
            c->readflush(reqlen);
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

            char *end = HTTP::headerEnd(c->rb);
            if (!end) return 0;

            char *start = HTTP::headersStart(c->rb);
            if (!start) return -1;

            char *headers = start;
            int headersLen = HTTP::headerLen(headers, end);
            int cmdLen = start - c->rb;

            char *method, *url, *args, *ver;
            if (HTTP::request(c->rb, &method, &url, &args, &ver) == -1) return -1;

            int type;
            if      (!strcasecmp(method, "GET"))  type = HTTPServer::Method::GET;
            else if (!strcasecmp(method, "POST")) type = HTTPServer::Method::POST;
            else return -1;

            dispatcher = Dispatcher(type, url, args, headers, cmdLen+headersLen);

            StringPiece cnhv;
            if (type == HTTPServer::Method::POST) {
                StringPiece ct, cl;
                HTTP::headerGrep(headers, end, 3, "Connection", &cnhv, "Content-Type", &ct, "Content-Length", &cl);
                dispatcher.postlen = atoi(BlankNull(cl.data()));
                dispatcher.reqlen += dispatcher.postlen;
                if (dispatcher.postlen) dispatcher.postdata = headers + headersLen;
            }
            else {
                HTTP::headerGrep(headers, end, 1, "Connection", &cnhv);
            }
            persistent = !HTTP::connectionClose(BlankNull(cnhv.data()));

            int ret = dispatcher.Thunk(this, c);
            if (ret < 0) return ret;
        }
    }

    int Dispatch(Connection *c, int type, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
        /* process request */
        Timer timer;
        HTTPServer::Response response = server->Request(c, type, url, args, headers, postdata, postlen);
        INFOf("%s %s %s %d cl=%d %f ms", c->name().c_str(), HTTPServer::Method::name(type), url, response.code, response.content_length, timer.GetTime()); 
        if (response.refill) c->readable = 0;

        /* write response/headers */
        if (response.write_headers) {
            if (WriteHeaders(c, &response) < 0) return -1;
        }

        /* prepare/deliver content */
        if (response.content) {
            if (c->write(response.content, response.content_length) < 0) return -1;
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
            Typed::Replace<Query>(&refill, 0);
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

        return c->write(h, hl);
    }
};

int HTTPServer::Connected(Connection *c) { c->query = new HTTPServerConnection(this); return 0; }

void HTTPServer::connectionClosedCB(Connection *c, ConnectionClosedCB cb) {
    ((HTTPServerConnection*)c->query)->closedCB.push_back(HTTPServerConnection::ClosedCallback(cb));
} 

HTTPServer::Response HTTPServer::Response::_400(400, "text/html; charset=iso-8859-1", StringPiece::FromString(
        "<!DOCTYPE HTML PUBLIC \"-//IETF//DTD HTML 2.0//EN\">\r\n"
        "<html><head>\r\n"
        "<title>400 Bad Request</title>\r\n"
        "</head><body>\r\n"
        "<h1>Bad Request</h1>\r\n"
        "<p>Your browser sent a request that this server could not understand.<br />\r\n"
        "</p>\r\n"
        "<hr>\r\n"
        "</body></html>\r\n"));

/* FileResource */

struct FileResourceQuery : public Query {
    LocalFile f;
    FileResourceQuery(const string &fn) : f(fn, "r") {}
    int Flushed(Connection *c) {
        if (!f.Opened()) return 0;
        c->writable = 1;
        c->wl = f.Read(c->wb, sizeof(c->wb));
        if (c->wl < sizeof(c->wb)) return 0;
        return 1;
    }
};

HTTPServer::Response HTTPServer::FileResource::Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
    if (!size) return HTTPServer::Response::_400;
    return Response(type, size, new FileResourceQuery(filename));
}

/* ConsoleResource */

HTTPServer::Response HTTPServer::ConsoleResource::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
    StringPiece v;
    if (args) HTTP::argGrep(args, 0, 1, "v", &v);
    app->shell.Run(v.str());
    string response = StrCat("<html>Shell::run('", v.str(), "')<br/></html>\n");
    return HTTPServer::Response("text/html; charset=UTF-8", &response);
}

/* StreamResource */

#ifdef LFL_FFMPEG
struct StreamResourceClient : public Query {
    Connection *conn;
    HTTPServer::StreamResource *resource;
    AVFormatContext *fctx;
    unsigned long long start;

    StreamResourceClient(Connection *c, HTTPServer::StreamResource *r) : conn(c), resource(r), start(0) {
        resource->subscribers[this] = conn;
        fctx = avformat_alloc_context();
        fctx_copy_streams(fctx, resource->fctx);
        fctx->max_delay = (int)(0.7*AV_TIME_BASE);
    }
    virtual ~StreamResourceClient() {
        resource->subscribers.erase(this);
        fctx_free(fctx);
    }

    int Flushed(Connection *c) { return 1; }
    void Open() { if (avio_open_dyn_buf(&fctx->pb)) ERROR("avio_open_dyn_buf"); }

    void Write(AVPacket *pkt, unsigned long long timestamp) {        
        Open();
        if (!start) start = timestamp;
        if (timestamp) {
            AVStream *st = fctx->streams[pkt->stream_index];
            AVRational r = {1, 1000000};
            unsigned t = timestamp - start;
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
        if (len < 0) { ERROR("avio_close_dyn_buf"); return; }
        if (conn->write(buf, len) < 0) conn->_error();
        av_free(buf);
    }

    static void fctx_free(AVFormatContext *fctx) {
        for (int i=0; i<fctx->nb_streams; i++) av_freep(&fctx->streams[i]);
        av_free(fctx);
    }

    static void fctx_copy_streams(AVFormatContext *dst, AVFormatContext *src) {
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

    static AVFrame *picture_alloc(enum PixelFormat pix_fmt, int width, int height) {
        AVFrame *picture; uint8_t *picture_buf; int size;
        if (!(picture = avcodec_alloc_frame())) return 0;
        size = avpicture_get_size(pix_fmt, width, height);
        if (!(picture_buf = (uint8_t*)av_malloc(size))) { av_free(picture); return 0; }
        avpicture_fill((AVPicture *)picture, picture_buf, pix_fmt, width, height);
        return picture;
    } 
    static void picture_free(AVFrame *picture) {
        av_free(picture->data[0]);
        av_free(picture);
    }

    static AVFrame *samples_alloc(int num_samples, int num_channels, short **samples_out) {
        AVFrame *samples; uint8_t *samples_buf; int size = 2 * num_samples * num_channels;
        if (!(samples = avcodec_alloc_frame())) return 0;
        samples->nb_samples = num_samples;
        if (!(samples_buf = (uint8_t*)av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE))) { av_free(samples); return 0; }
        avcodec_fill_audio_frame(samples, num_channels, AV_SAMPLE_FMT_S16, samples_buf, size, 1);
        memset(samples_buf+size, 0, FF_INPUT_BUFFER_PADDING_SIZE);
        if (samples_out) *samples_out = (short*)samples_buf;
        return samples;
    }
    static void samples_free(AVFrame *picture) {
        av_free(picture->data[0]);
        av_free(picture);
    }
};

HTTPServer::StreamResource::~StreamResource() {
    delete resampler.out;
    if (audio && audio->codec) avcodec_close(audio->codec);
    if (video && video->codec) avcodec_close(video->codec);
    if (picture) StreamResourceClient::picture_free(picture);
    if (samples) StreamResourceClient::samples_free(picture);
    StreamResourceClient::fctx_free(fctx);
}

HTTPServer::StreamResource::StreamResource(const char *oft, int Abr, int Vbr) : fctx(0), open(0), abr(Abr), vbr(Vbr), 
audio(0), samples(0), sample_data(0), frame(0), channels(0), resamples_processed(0), 
video(0), picture(0), conv(0)
{
    fctx = avformat_alloc_context();
    fctx->oformat = av_guess_format(oft, 0, 0);
    if (!fctx->oformat) { ERROR("guess_format '", oft, "' failed"); return; }
    INFO("StreamResource: format ", fctx->oformat->mime_type);
    openStreams(FLAGS_lfapp_audio, FLAGS_lfapp_camera);
}

HTTPServer::Response HTTPServer::StreamResource::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
    if (!open) return HTTPServer::Response::_400;
    Response response(fctx->oformat->mime_type, -1, new StreamResourceClient(c, this), false);
    if (HTTPServerConnection::WriteHeaders(c, &response) < 0) { c->_error(); return response; }
    ((StreamResourceClient*)response.refill)->WriteHeader();
    return response;
}

void HTTPServer::StreamResource::openStreams(bool A, bool V) {
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
        if (avcodec_open2(vc, codec, 0) < 0) { ERROR("avcodec_open2"); return; }
        if (!vc->codec) { ERROR("no video codec"); return; }

        if (vc->pix_fmt != PIX_FMT_YUV420P) { ERROR("pix_fmt ", vc->pix_fmt, " != ", PIX_FMT_YUV420P); return; }
        if (!(picture = StreamResourceClient::picture_alloc(vc->pix_fmt, vc->width, vc->height))) { ERROR("picture_alloc"); return; }
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
        if (avcodec_open2(ac, codec, 0) < 0) { ERROR("avcodec_open2"); return; }
        if (!ac->codec) { ERROR("no audio codec"); return; }

        if (!(frame = ac->frame_size)) { ERROR("empty frame size"); return; }
        channels = ac->channels;

        if (!(samples = StreamResourceClient::samples_alloc(frame, channels, &sample_data))) { ERROR("picture_alloc"); return; }
    }

    open = 1;
}

void HTTPServer::StreamResource::update(int audio_samples, bool video_sample) {
    if (!open || !subscribers.size()) return;

    AVCodecContext *vc = video ? video->codec : 0;
    AVCodecContext *ac = audio ? audio->codec : 0;

    if (ac && audio_samples) {
        if (!resampler.out) {
            resampler.out = new RingBuf(ac->sample_rate, ac->sample_rate*channels);
            resampler.Open(resampler.out, FLAGS_chans_in, FLAGS_sample_rate, Sample::S16,
                                          channels,       ac->sample_rate,   Sample::FromFFMpegId(ac->channel_layout));
        };
        RingBuf::Handle L(app->audio.IL, app->audio.IL->ring.back-audio_samples, audio_samples);
        RingBuf::Handle R(app->audio.IR, app->audio.IR->ring.back-audio_samples, audio_samples);
        if (resampler.Update(audio_samples, &L, FLAGS_chans_in > 1 ? &R : 0)) open=0;
    }

    for (;;) {
        bool asa = ac && resampler.output_available >= resamples_processed + frame * channels;
        bool vsa = vc && video_sample;
        if (!asa && !vsa) break;
        if (vc && !vsa) break;

        if (!vsa) { sendAudio(); continue; }       
        if (!asa) { sendVideo(); video_sample=0; continue; }

        int audio_behind = resampler.output_available - resamples_processed;
        unsigned long long audio_timestamp = resampler.out->ReadTimestamp(0, resampler.out->ring.back - audio_behind);

        if (audio_timestamp < app->camera.image_timestamp) sendAudio();
        else { sendVideo(); video_sample=0; }
    }
}

void HTTPServer::StreamResource::sendAudio() {
    int behind = resampler.output_available - resamples_processed;
    resamples_processed += frame * channels;

    AVCodecContext *ac = audio->codec;
    RingBuf::Handle H(resampler.out, resampler.out->ring.back - behind, frame * channels);

    /* linearize */
    for (int i=0; i<frame; i++) 
        for (int c=0; c<channels; c++)
            sample_data[i*channels + c] = H.Read(i*channels + c) * 32768.0;

    /* broadcast */
    AVPacket pkt; int got=0;
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    avcodec_encode_audio2(ac, &pkt, samples, &got);
    if (got) broadcast(&pkt, H.ReadTimestamp(0));

    av_free_packet(&pkt);
}

void HTTPServer::StreamResource::sendVideo() {
    AVCodecContext *vc = video->codec;

    /* convert video */
    if (!conv)
        conv = sws_getContext(FLAGS_camera_image_width, FLAGS_camera_image_height, (PixelFormat)Pixel::ToFFMpegId(app->camera.image_format),
                              vc->width, vc->height, vc->pix_fmt, SWS_BICUBIC, 0, 0, 0);

    int camera_linesize[4] = { app->camera.image_linesize, 0, 0, 0 };
    sws_scale(conv, (uint8_t**)&app->camera.image, camera_linesize, 0, FLAGS_camera_image_height, picture->data, picture->linesize);

    /* broadcast */
    AVPacket pkt; int got=0;
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    avcodec_encode_video2(vc, &pkt, picture, &got);
    if (got) broadcast(&pkt, app->camera.image_timestamp);

    av_free_packet(&pkt);
}

void HTTPServer::StreamResource::broadcast(AVPacket *pkt, unsigned long long timestamp) {
    for (SubscriberMap::iterator i = subscribers.begin(); i != subscribers.end(); i++) {
        StreamResourceClient *client = (StreamResourceClient*)(*i).first;
        client->Write(pkt, timestamp);
    }
}
#endif /* LFL_FFMPEG */

struct SMTPClientConnection : public Query {
    enum { INIT=0, SENT_HELO=1, READY=2, MAIL_FROM=3, RCPT_TO=4, SENT_DATA=5, SENDING=6, RESETING=7, QUITING=8 };
    static const char *StateString(int n) {
        static const char *s[] = { "INIT", "SENT_HELO", "READY", "MAIL_FROM", "RCPT_TO", "SENT_DATA", "SENDING", "RESETING", "QUITTING" };
        return (n >= 0 && n < sizeofarray(s)) ? s[n] : "";
    }
    static bool DeliveringState(int state) { return (state >= MAIL_FROM && state <= SENDING); }

    SMTPClient *server; SMTPClient::DeliverableCB deliverable_cb; SMTPClient::DeliveredCB delivered_cb; void *arg;
    int state, rcpt_index; string greeting, helo_domain, ehlo_response, response_lines; SMTP::Message mail;
    string RcptTo(int index) const { return StrCat("RCPT TO: <", mail.rcpt_to[index], ">\r\n"); }

    SMTPClientConnection(SMTPClient *S, SMTPClient::DeliverableCB CB1, SMTPClient::DeliveredCB CB2)
        : server(S), deliverable_cb(CB1), delivered_cb(CB2), state(0) {}

    int Connected(Connection *c) { helo_domain = server->HeloDomain(c->src_addr); return 0; }

    void Close(Connection *c) {
        server->total_disconnected++;
        if (DeliveringState(state)) delivered_cb(0, mail, 0, "");
        deliverable_cb(c, helo_domain, 0);
    }

    int Read(Connection *c) {
        int processed = 0;
        StringLineIter lines(c->rb, c->rl, StringLineIter::Flag::BlankLines);
        for (const char *line = lines.Next(); line; line = lines.Next()) {
            processed = lines.offset;
            if (!response_lines.empty()) response_lines.append("\r\n");
            response_lines.append(line, lines.linelen);

            const char *dash = nextchar(line, notnum);
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
            if (!response.empty()) if (c->writeflush(response) != response.size()) return -1;

            response_lines.clear();
            state++;
        }
        c->readflush(processed);

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
        if (c->writeflush(response) != response.size()) c->_error();
    }
};

Connection *SMTPClient::DeliverTo(IPV4::Addr ipv4_addr, IPV4EndpointSource *src_pool,
                                  DeliverableCB deliverable_cb, DeliveredCB delivered_cb) {
    static const int tcp_port = 25;
    Connection *c = Connect(ipv4_addr, tcp_port, src_pool);
    if (!c) return 0;

    c->query = new SMTPClientConnection(this, deliverable_cb, delivered_cb);
    return c;
}

void SMTPClient::DeliverDeferred(Connection *c) { ((SMTPClientConnection*)c->query)->Deliver(c); }

struct SMTPServerConnection : public Query {
    SMTPServer *server;
    string my_domain, client_domain;
    SMTP::Message message;
    bool in_data;

    SMTPServerConnection(SMTPServer *s) : server(s), in_data(0) {}
    void ClearStateTable() { message.mail_from.clear(); message.rcpt_to.clear(); message.content.clear(); in_data=0; }

    int Connected(Connection *c) {
        my_domain = server->HeloDomain(c->src_addr);
        string greeting = StrCat("220 ", my_domain, " Simple Mail Transfer Service Ready\r\n");
        return (c->write(greeting) == greeting.size()) ? 0 : -1;
    }
    int Read(Connection *c) {
        int offset = 0, processed;
        while (c->state == Connection::Connected) {
            bool last_in_data = in_data;
            if (in_data) { if ((processed = ReadData    (c, c->rb+offset, c->rl-offset)) < 0) return -1; }
            else         { if ((processed = ReadCommands(c, c->rb+offset, c->rl-offset)) < 0) return -1; }
            offset += processed;
            if (last_in_data == in_data) break;
        }
        if (offset) c->readflush(offset);
        return 0;
    }

    int ReadCommands(Connection *c, const char *in, int len) {
        int processed = 0;
        StringLineIter lines(in, len, StringLineIter::Flag::BlankLines | StringLineIter::Flag::InPlace);
        for (const char *line = lines.Next(); line && lines.offset>=0 && !in_data; line = lines.Next()) {
            processed = lines.offset;
            StringWordIter words(line, lines.linelen, isint3<' ', '\t', ':'>);
            string cmd = toupper(BlankNull(words.Next()));
            string a1_orig = BlankNull(words.Next());
            string a1 = toupper(a1_orig), response="500 unrecognized command\r\n";

            if (cmd == "MAIL" && a1 == "FROM") {
                ClearStateTable();
                message.mail_from = words.Remaining();
                response = "250 OK\r\n";
            }
            else if (cmd == "RCPT" && a1 == "TO") { message.rcpt_to.push_back(words.Remaining()); response="250 OK\r\n"; }
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
            else if (cmd == "QUIT") { c->writeflush(StrCat("221 ", server->domain, " closing connection\r\n")); c->_error(); }
            if (!response.empty()) if (c->write(response) != response.size()) return -1;
        }
        return processed;
    }

    int ReadData(Connection *c, const char *in, int len) {
        int processed = 0;
        StringLineIter lines(in, len, StringLineIter::Flag::BlankLines | StringLineIter::Flag::InPlace);
        for (const char *line = lines.Next(); line && lines.offset>=0; line = lines.Next()) {
            processed = lines.offset;
            if (lines.linelen == 1 && *line == '.') { in_data=0; break; }
            message.content.append(line, lines.linelen);
            message.content.append("\r\n");
        }
        if (!in_data) {
            c->write("250 OK\r\n");
            server->ReceiveMail(c, message); 
            ClearStateTable();
        }
        return processed;
    }
};

int SMTPServer::Connected(Connection *c) { total_connected++; c->query = new SMTPServerConnection(this); return 0; }

void SMTPServer::ReceiveMail(Connection *c, const SMTP::Message &mail) {
    INFO("SMTPServer::ReceiveMail FROM=", mail.mail_from, ", TO=", mail.rcpt_to, ", content=", mail.content);
}

/* GPlusClient */

struct GPlusClientQuery {
    struct PersistentConnection : public Query {
        UDPClient::ResponseCB responseCB; UDPClient::HeartbeatCB heartbeatCB; void *arg;
        PersistentConnection(UDPClient::ResponseCB RCB, UDPClient::HeartbeatCB HCB, void *Arg) : responseCB(RCB), heartbeatCB(HCB), arg(Arg) {}

        int Heartbeat(Connection *c) { if (heartbeatCB) heartbeatCB(c); return 0; }
        void Close(Connection *c) { if (responseCB) responseCB(c, 0, 0); }
        int Read(Connection *c) {
            for (int i=0; i<c->packets.size() && responseCB; i++) {
                if (c->state != Connection::Connected) break;
                responseCB(c, c->packets[i].buf, c->packets[i].len);
            }
            return 0;
        }
    };
};

Connection *GPlusClient::PersistentConnection(const string &name, UDPClient::ResponseCB responseCB, UDPClient::HeartbeatCB HCB, void *arg) {
    Connection *c = EndpointConnect(name);
    c->query = new GPlusClientQuery::PersistentConnection(responseCB, HCB, arg);
    return c;
}

/* Sniffer */

#ifdef LFL_PCAP
void Sniffer::GetDeviceAddressSet(set<IPV4::Addr> *out) {
    static IPV4::Addr localhost = Network::addr("127.0.0.1");
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

int Sniffer_threadproc(void *opaque) {
    Sniffer *sniffer = (Sniffer*)opaque;
    pcap_pkthdr *pkthdr; const unsigned char *packet; int ret;
    while (Running() && (ret = pcap_next_ex((pcap_t*)sniffer->handle, &pkthdr, &packet)) >= 0) {
        if (!ret) continue;
        sniffer->cb((const char *)packet, pkthdr->caplen, pkthdr->len);
    }
    return ret;
}

Sniffer *Sniffer::Open(const string &dev, const string &filter, int snaplen, CB cb) {
    char errbuf[PCAP_ERRBUF_SIZE];
    bpf_u_int32 ip, mask, ret;
    pcap_t *handle;
    if (pcap_lookupnet(dev.c_str(), &ip, &mask, errbuf)) { ERROR("no netmask for ", dev); return 0; }
    if (!(handle = pcap_open_live(dev.c_str(), snaplen, 1, 1000, errbuf))) { ERROR("open failed: ", dev, ": ", errbuf); return 0; }
    if (filter.size()) {
        bpf_program fp;
        if (pcap_compile(handle, &fp, filter.c_str(), 0, ip)) { ERROR("parse filter: ", filter, ": ", pcap_geterr(handle)); return 0; }
        if (pcap_setfilter(handle, &fp)) { ERROR("install filter: ", filter, ": ", pcap_geterr(handle)); return 0; }
    }
    Sniffer *sniffer = new Sniffer(handle, ip, mask, cb);
    sniffer->thread.Open(Sniffer_threadproc, sniffer);
    sniffer->thread.Start();
    return sniffer;
}
#else /* LFL_PCAP */
void Sniffer::GetDeviceAddressSet(set<IPV4::Addr> *out) {}
void Sniffer::PrintDevices(vector<string> *out) {}
int Sniffer_threadproc(void *opaque) { return 0; }
Sniffer *Sniffer::Open(const string &dev, const string &filter, int snaplen, CB cb) { ERROR("sniffer not implemented"); return 0; }
#endif /* LFL_PCAP */
void Sniffer::GetIPAddress(IPV4::Addr *out) {
    static IPV4::Addr localhost = Network::addr("127.0.0.1");
    *out = 0;
#if defined(_WIN32)
#elif defined(LFL_ANDROID)
    *out = ntohl(android_ipv4_address());
#else
    ifaddrs* ifap = NULL;
    int r = getifaddrs(&ifap);
    if (r) { ERROR("getifaddrs ", r); return; }
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
    static IPV4::Addr localhost = Network::addr("127.0.0.1");
    *out = 0;
#if defined(_WIN32)
#elif defined(LFL_ANDROID)
    *out = ntohl(android_ipv4_broadcast_address());
#else
    ifaddrs* ifap = NULL;
    int r = getifaddrs(&ifap);
    if (r) { ERROR("getifaddrs ", r); return; }
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
