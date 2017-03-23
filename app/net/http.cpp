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

#include "core/app/network.h"
#include "core/app/net/resolver.h"

namespace LFL {
void HTTP::SplitHostAndPort(const string &in, int default_port, string *host, int *port) {
  string port_text;
  LFL::Split(in, LFL::isint<':'>, host, &port_text);
  int port_val = atoi(port_text);
  *port = port_val ? port_val : default_port;
}

bool HTTP::ParseHost(const char *host, const char *host_end, string *hostO, string *portO) {
  const char *colon = strstr(host, ":"), *port = 0;
  if (!host_end) host_end = host + strlen(host);
  if (colon && colon < host_end) port = colon+1;
  if (hostO) hostO->assign(host, port ? port-host-1 : host_end-host);
  if (portO) portO->assign(port ? port : "", port ? host_end-port : 0);
  return 1;
}

bool HTTP::ResolveHost(const char *hostname, const char *host_end, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int default_port) {
  string h, p;
  if (!ParseHost(hostname, host_end, &h, &p)) return 0;
  return ResolveEndpoint(h, p, ipv4_addr, tcp_port, ssl, default_port);
}

bool HTTP::ResolveEndpoint(const string &host, const string &port, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int default_port) {
  if (ipv4_addr) {
    *ipv4_addr = SystemNetwork::GetHostByName(host);
    if (*ipv4_addr == -1) { ERROR("resolve"); return 0; }
  }
  if (tcp_port) {
    *tcp_port = !port.empty() ? atoi(port.c_str()) : (default_port ? default_port : (ssl ? 443 : 80));
    if (*tcp_port < 0 || *tcp_port >= 65536) { ERROR("oob port"); return 0; }
  }
  return 1;
}

bool HTTP::ParseURL(const char *url, string *protO, string *hostO, string *portO, string *pathO) {
  const char *host = ParseProtocol(url, protO);
  const char *host_end = strstr(host, "/");
  HTTP::ParseHost(host, host_end, hostO, portO);
  if (pathO) pathO->assign(host_end ? host_end+1 : "");
  return 1;
}

bool HTTP::ResolveURL(const char *url, bool *ssl, IPV4::Addr *ipv4_addr, int *tcp_port, string *host, string *path, int default_port, string *prot) {
  string my_prot, port, my_host, my_path; bool my_ssl;
  if (!prot) prot = &my_prot;
  if (!host) host = &my_host;
  if (!path) path = &my_path;
  if (!ssl) ssl = &my_ssl;

  ParseURL(url, prot, host, &port, path);
  *ssl = !prot->empty() && !strcasecmp(prot->c_str(), "https");
  if (!prot->empty() && strcasecmp(prot->c_str(), "http") && !*ssl) return 0;
  if (host->empty()) { ERROR("no host or path"); return 0; }
  if (!HTTP::ResolveEndpoint(*host, port, ipv4_addr, tcp_port, *ssl, default_port)) { ERROR("HTTP::ResolveURL ", *host); return 0; }
  return 1;
}

string HTTP::HostURL(const char *url) {
  string my_prot, my_port, my_host, my_path;
  ParseURL(url, &my_prot, &my_host, &my_port, &my_path);
  string ret = !my_prot.empty() ? StrCat(my_prot, "://") : "http://";
  if (!my_host.empty()) ret += my_host;
  if (!my_port.empty()) ret += string(":") + my_port;
  return ret;
}

int HTTP::ParseRequest(char *buf, char **methodO, char **urlO, char **argsO, char **verO) {
  char *url, *ver, *args;
  if (!(url = const_cast<char*>(FindChar(buf, isspace))))    return -1;    *url = 0;
  if (!(url = const_cast<char*>(FindChar(url+1, notspace)))) return -1;
  if (!(ver = const_cast<char*>(FindChar(url, isspace))))    return -1;    *ver = 0;
  if (!(ver = const_cast<char*>(FindChar(ver+1, notspace)))) return -1;

  if ((args = strchr(url, '?'))) *args++ = 0;

  if (methodO) *methodO = buf;
  if (urlO) *urlO = url;
  if (argsO) *argsO = args;
  if (verO) *verO = ver;
  return 0;
}

char *HTTP::FindHeadersStart(char *buf) {
  char *start = strstr(buf, "\r\n");
  if (!start) return 0;
  *start = 0;
  return start + 2;
}

char *HTTP::FindHeadersEnd(char *buf) {
  char *end = strstr(buf, "\r\n\r\n");
  if (!end) return 0;
  *(end+2) = 0;
  return end + 2;
}

const char *HTTP::FindHeadersEnd(const char *buf) {
  const char *end = strstr(buf, "\r\n\r\n");
  if (!end) return 0;
  return end + 2;
}

int HTTP::GetHeaderLen(const char *beg, const char *end) { return end - beg + 2; }

int HTTP::GetHeaderNameLen(const char *beg) {
  const char *n = beg;
  while (*n && !isspace(*n) && *n != ':') n++;
  return *n == ':' ? n - beg : 0;
}

int HTTP::GetURLArgNameLen(const char *beg) {
  const char *n = beg;
  while (*n && !isspace(*n) && *n != '=' && *n != '&') n++;
  return n - beg;
}

string HTTP::GrepHeaders(const char *headers, const char *end, const string &name) {
  if (!end) end = HTTP::FindHeadersEnd(headers);
  if (!end) end = headers + strlen(headers);

  int hlen=end-headers, hnlen;
  StringLineIter lines(StringPiece(headers, hlen));
  for (const char *line = lines.Next(); line; line = lines.Next()) {
    if (!(hnlen = HTTP::GetHeaderNameLen(line))) continue;
    if (hnlen == name.size() && !strncasecmp(name.c_str(), line, hnlen)) return string(line+hnlen+2, lines.cur_len-hnlen-2);
  }
  return "";
}

#define HTTPGrepImpl(k, kl, v) \
  va_list ap; \
  va_start(ap, num); \
  vector<char*> k(num); \
  vector<int> kl(num); \
  vector<StringPiece*> v(num); \
  for (int i=0; i<num; i++) { \
    k[i] = va_arg(ap, char*); \
    kl[i] = strlen(k[i]); \
    v[i] = va_arg(ap, StringPiece*); \
  } \
va_end(ap);

int HTTP::GrepHeaders(const char *headers, const char *end, int num, ...) {
  HTTPGrepImpl(k, kl, v);
  if (!end) end = HTTP::FindHeadersEnd(headers);
  if (!end) end = headers + strlen(headers);

  int hlen=end-headers, hnlen;
  StringLineIter lines(StringPiece(headers, hlen));
  for (const char *h = lines.Next(); h; h = lines.Next()) {
    if (!(hnlen = HTTP::GetHeaderNameLen(h))) continue;
    for (int i=0; i<num; i++) if (hnlen == kl[i] && !strncasecmp(k[i], h, hnlen)) {
      const char *hv = FindChar(h+hnlen+1, notspace, lines.cur_len-hnlen-1);
      if (!hv) v[i]->clear();
      else     v[i]->assign(hv, lines.cur_len-(hv-h));
    }
  }
  return 0;
}

int HTTP::GrepURLArgs(const char *args, const char *end, int num, ...) {
  HTTPGrepImpl(k, kl, v);
  if (!end) end = args + strlen(args);

  int alen=end-args, anlen;
  StringWordIter words(StringPiece(args, alen), isand, 0);
  for (const char *a = words.Next(); a; a = words.Next()) {
    if (!(anlen = HTTP::GetURLArgNameLen(a))) continue;
    for (int i=0; i<num; i++) if (anlen == kl[i] && !strncasecmp(k[i], a, anlen)) {
      if (*(a+anlen) && *(a+anlen) == '=') v[i]->assign(a+anlen+1, words.cur_len-anlen-1);
      else v[i]->assign(a, words.cur_len);
    }
  }
  return 0;
}

string HTTP::EncodeURL(const char *url) {
  static const char encodeURIcomponentPass[] = "~!*()'";
  static const char encodeURIPass[] = "./@#:?,;-_&";
  string ret;
  for (const unsigned char *p = MakeUnsigned(url); *p; p++) {
    if      (*p >= '0' && *p <= '9') ret += *p;
    else if (*p >= 'a' && *p <= 'z') ret += *p;
    else if (*p >= 'A' && *p <= 'Z') ret += *p;
    else if (strchr(encodeURIcomponentPass, *p)) ret += *p;
    else if (strchr(encodeURIPass, *p)) ret += *p;
    else StringAppendf(&ret, "%%%02x", *p);
  }
  return ret;
}

string HTTP::MakeHeaders(int code, int content_length, const char *content_type) {
  char date[64] = {0};
  httptime(date, sizeof(date));
  string codetext = "Unknown";
  if      (code == 200) codetext = "OK";
  else if (code == 400) codetext = "Bad Request";
  return StrCat("HTTP/1.1 ", code, " ", codetext, "\r\nDate: ", date,
                content_length >= 0 ? StrCat("\r\nContent-Length: ", content_length) : "",
                "\r\nContent-Type: ", content_type, "\r\n\r\n");
}

/* HTTPClient */

struct HTTPClientHandler {
  struct Protocol : public Connection::Handler {
    int result_code, read_header_length, read_content_length, current_chunk_length, current_chunk_read;
    bool chunked_encoding, full_chunk_cb;
    string content_type;

    Protocol(bool full_chunks) : full_chunk_cb(full_chunks) { Reset(); }
    void Reset() {
      result_code=read_header_length=read_content_length=current_chunk_length=current_chunk_read=0;
      chunked_encoding=0;
      content_type.clear();  
    }

    int Read(Connection *c) {
      char *cur = c->rb.begin();
      if (!read_header_length) {
        char *headers_end = HTTP::FindHeadersEnd(cur);
        if (!headers_end) return 1;

        StringPiece status_line=StringPiece::Unbounded(cur), h, ct, cl, te;
        h.buf = NextLine(status_line.buf, true, &status_line.len);
        h.len = read_header_length = HTTP::GetHeaderLen(cur, headers_end);
        StringWordIter status_words(status_line);
        status_words.Next();
        result_code = atoi(status_words.Next());

        HTTP::GrepHeaders(h.buf, headers_end, 3, "Content-Type", &ct, "Content-Length", &cl, "Transfer-Encoding", &te);
        current_chunk_length = read_content_length = atoi(BlankNull(cl.data()));
        chunked_encoding = te.str() == "chunked";
        content_type = ct.str();

        if (Headers(c, status_line, h)) { c->handler.reset(); return -1; }
        cur = headers_end+2;
      }
      for (;;) {
        if (chunked_encoding && !current_chunk_length) {
          char *cur_in = cur;
          cur += IsNewline(cur);
          char *chunkHeader = cur;
          if (!(cur = const_cast<char*>(NextLine(cur)))) { cur=cur_in; break; }
          current_chunk_length = strtoul(chunkHeader, 0, 16);
        }

        int rb_left = c->rb.size() - (cur - c->rb.begin());
        if (rb_left <= 0) break;
        if (chunked_encoding) {
          int chunk_left = current_chunk_length - current_chunk_read;
          if (chunk_left < rb_left) rb_left = chunk_left;
          if (rb_left < chunk_left && full_chunk_cb) break;
        }

        if (rb_left) if (Content(c, StringPiece(cur, rb_left))) { c->handler.reset(); return -1; }
        cur += rb_left;
        current_chunk_read += rb_left;
        if (current_chunk_read == current_chunk_length) current_chunk_read = current_chunk_length = 0;
      }
      if (cur != c->rb.begin()) c->ReadFlush(cur - c->rb.begin());
      return 0;
    }
    virtual int Headers(Connection *c, const StringPiece &sl, const StringPiece &h) { return 0; }
    virtual int Content(Connection *c, const StringPiece &b) { return 0; }
  };

  struct WGet : public Protocol {
    SocketService *svc=0;
    bool ssl=0;
    string host, path;
    int port=0, redirects=0;
    File *out=0;
    HTTPClient::ResponseCB cb;
    StringCB redirect_cb;

    virtual ~WGet() { if (out) INFO("close ", out->Filename()); delete out; }
    WGet(SocketService *Svc, bool SSL, const string &Host, int Port, const string &Path, File *Out,
         const HTTPClient::ResponseCB &CB=HTTPClient::ResponseCB(), const StringCB &RedirCB=StringCB()) :
      Protocol(false), svc(Svc), ssl(SSL), host(Host), path(Path), port(Port), out(Out), cb(CB), redirect_cb(RedirCB) {}

    bool LoadURL(const string &url, string *prot=0) {
      return HTTP::ResolveURL(url.c_str(), &ssl, 0, &port, &host, &path, 0, prot);
    }

    void Close(Connection *c) override { if (cb) cb(c, 0, content_type, 0, 0); }
    int Connected(Connection *c) override {
      return HTTPClient::WriteRequest(c, HTTPServer::Method::GET, host.c_str(), path.c_str(), 0, 0, 0, false);
    }

    int Headers(Connection *c, const StringPiece &sl, const StringPiece &h) override { 
      if (result_code == 301 && redirects++ < 5) {
        StringPiece loc;
        HTTP::GrepHeaders(h.begin(), h.end(), 1, "Location", &loc);
        string location = loc.str();
        if (!location.empty()) {
          if (redirect_cb) redirect_cb(location);
          else { return -1; c->handler = nullptr; ResolveHost(); return 0; }
        }
      }
      return cb ? cb(c, h.data(), content_type, 0, read_content_length) : 0;
    }

    int Content(Connection *c, const StringPiece &b) override {
      if (out) { if (out->Write(b.buf, b.len) != b.len) ERROR("write ", out->Filename()); }
      return cb ? cb(c, 0, content_type, b.buf, b.len) : 0;
    }

    void ResolveHost() {
      app->net->system_resolver->NSLookup
        (host, bind(&HTTPClientHandler::WGet::ResolverResponseCB, this, _1, _2));
    }

    void ResolverResponseCB(IPV4::Addr ipv4_addr, DNS::Response*) {
      Connection *c = 0;
      if (ipv4_addr != IPV4::Addr(-1)) {
        c = ssl ?
          svc->SSLConnect(app->net->ssl, ipv4_addr, port) :
          svc->Connect(ipv4_addr, port);
      }
      if (!c) { if (cb) cb(0, 0, string(), 0, 0); delete this; }
      else c->handler = unique_ptr<Connection::Handler>(this);
    }
  };

  struct WPost : public WGet {
    string mimetype, postdata;
    WPost(SocketService *Svc, bool SSL, const string &Host, int Port, const string &Path, const string &Mimetype, const char *Postdata, int Postlen,
          HTTPClient::ResponseCB CB=HTTPClient::ResponseCB()) : WGet(Svc, SSL, Host, Port, Path, 0, CB), mimetype(Mimetype), postdata(Postdata,Postlen) {}
    int Connected(Connection *c) {
      return HTTPClient::WriteRequest(c, HTTPServer::Method::POST, host.c_str(), path.c_str(),
                                      mimetype.data(), postdata.data(), postdata.size(), false);
    }
  };

  struct PersistentConnection : public Protocol {
    HTTPClient::ResponseCB responseCB;
    PersistentConnection(HTTPClient::ResponseCB RCB) : Protocol(true), responseCB(RCB) {}

    void Close(Connection *c) override { if (responseCB) responseCB(c, 0, content_type, 0, 0); }
    int Content(Connection *c, const StringPiece &b) override {
      int ret = 0;
      if (!read_content_length) FATAL("chunked transfer encoding not supported");
      if (responseCB) ret = responseCB(c, 0, content_type, b.buf, b.len);
      Protocol::Reset();
      return ret;
    }
  };
};

int HTTPClient::WriteRequest(Connection *c, int method, const char *host, const char *path, const char *postmime, const char *postdata, int postlen, bool persist) {
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

bool HTTPClient::WGet(const string &url, File *out, const ResponseCB &cb, const StringCB &redirect_cb) {
  unique_ptr<HTTPClientHandler::WGet> handler = make_unique<HTTPClientHandler::WGet>
    (app->net->tcp_client.get(), 0, "", 0, "", out, cb, redirect_cb);

  string prot;
  if (!handler->LoadURL(url, &prot)) {
    if (prot != "file") return false;
    string fn = StrCat(!handler->host.empty() ? "/" : "", handler->host , "/", handler->path);
    string content = LocalFile::FileContents(fn);
    if (!content.empty() && cb) cb(0, 0, string(), content.data(), content.size());
    if (cb)                     cb(0, 0, string(), 0,              0);
    return true;
  }

  if (!out && !cb) {
    string fn = BaseName(handler->path);
    if (fn.empty()) fn = "index.html";
    unique_ptr<LocalFile> f = make_unique<LocalFile>(StrCat(app->savedir, fn), "w");
    if (!f->Opened()) return ERRORv(false, "open file");
    handler->out = f.release();
  }

  handler.release()->ResolveHost();
  return true;
}

bool HTTPClient::WPost(const string &url, const string &mimetype, const char *postdata, int postlen, ResponseCB cb) {
  bool ssl;
  int tcp_port;
  string host, path;
  if (!HTTP::ResolveURL(url.c_str(), &ssl, 0, &tcp_port, &host, &path)) return 0;

  HTTPClientHandler::WPost *handler = new HTTPClientHandler::WPost
    (app->net->tcp_client.get(), ssl, host, tcp_port, path, mimetype, postdata, postlen, cb);

  if (!app->net->system_resolver->QueueResolveRequest
      (Resolver::Request(host, DNS::Type::A, bind(&HTTPClientHandler::WGet::ResolverResponseCB, handler, _1, _2))))
  { ERROR("resolver: ", url); delete handler; return 0; }
  return true;
}

Connection *HTTPClient::PersistentConnection(const string &url, string *host, string *path, ResponseCB responseCB) {
  bool ssl; IPV4::Addr ipv4_addr; int tcp_port;
  if (!HTTP::ResolveURL(url.c_str(), &ssl, &ipv4_addr, &tcp_port, host, path)) return 0;

  Connection *c = ssl ?
    app->net->tcp_client->SSLConnect(app->net->ssl, ipv4_addr, tcp_port) : 
    app->net->tcp_client->Connect(ipv4_addr, tcp_port);

  if (!c) return 0;

  c->handler = make_unique<HTTPClientHandler::PersistentConnection>(responseCB);
  return c;
}

/* HTTPServer */

struct HTTPServerConnection : public Connection::Handler {
  struct Dispatcher {
    int type=0, reqlen=0, postlen=0;
    const char *url=0, *args=0, *headers=0, *postdata=0; 
    Dispatcher() {}
    Dispatcher(int T, const char *U, const char *A, const char *H, int L) :
      type(T), reqlen(L), url(U), args(A), headers(H) {}

    void clear() { type=reqlen=postlen=0; url=args=headers=postdata=0; }
    bool empty() { return !type; }
    int Thunk(HTTPServerConnection *httpserv, Connection *c) {
      if (c->rb.size() < reqlen) return 0;
      int ret = httpserv->Dispatch(c, type, url, args, headers, postdata, postlen);
      c->ReadFlush(reqlen);
      clear();
      return ret;
    }
  };
  struct ClosedCallback {
    HTTPServer::ConnectionClosedCB cb;
    ClosedCallback(HTTPServer::ConnectionClosedCB CB) : cb(CB) {}
    void thunk(Connection *c) { cb(c); }
  };
  typedef vector<ClosedCallback> ClosedCB;

  HTTPServer *server;
  bool persistent=true;
  unique_ptr<Connection::Handler> refill;
  Dispatcher dispatcher;
  ClosedCB closedCB;
  HTTPServerConnection(HTTPServer *s) : server(s) {}

  void Closed(Connection *c) { for (auto &i : closedCB) i.thunk(c); }
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
      if (c->Write(HTTP::MakeHeaders(response.code, response.content_length, response.type)) < 0) return -1;
    }

    /* prepare/deliver content */
    if (response.content) {
      if (c->Write(response.content, response.content_length) < 0) return -1;
    }
    else if (response.refill) {
      if (refill) return -1;
      refill = unique_ptr<Connection::Handler>(response.refill);
    }
    else return -1;

    return 0;
  }

  int Flushed(Connection *c) { 
    if (refill) {
      int ret;
      if ((ret = refill->Flushed(c))) return ret;
      refill.reset();
      c->readable = 1;
      return 0;
    }
    if (!persistent) return -1;
    return 0;
  }
};

const char *HTTPServer::Method::name(int n) {
  if      (n == GET)  return "GET";
  else if (n == POST) return "POST";
  return 0;
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

int HTTPServer::Connected(Connection *c) { c->handler = make_unique<HTTPServerConnection>(this); return 0; }

HTTPServer::Response HTTPServer::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  Resource *requested = FindOrNull(urlmap, url);
  if (!requested) return Response::_400;
  return requested->Request(c, method, url, args, headers, postdata, postlen);
}

void HTTPServer::connectionClosedCB(Connection *c, ConnectionClosedCB cb) {
  dynamic_cast<HTTPServerConnection*>(c->handler.get())->closedCB.push_back(HTTPServerConnection::ClosedCallback(cb));
}

HTTPServer::Response HTTPServer::DebugResource::Request(Connection *c, int, const char *url, const char *args, const char *hdrs, const char *postdata, int postlen) {
  INFO("url: ", url);
  INFO("args: ", args);
  INFO("hdrs: ", hdrs);
  return Response::_400;
}

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

HTTPServer::FileResource::FileResource(const string &fn, const char *mimetype) :
  filename(fn), type(mimetype ? mimetype : "application/octet-stream") {
  LocalFile f(filename, "r");
  if (!f.Opened()) return;
  size = f.Size();
}

HTTPServer::Response HTTPServer::FileResource::Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  if (!size) return HTTPServer::Response::_400;
  return Response(type, size, new HTTPServerFileResourceHandler(filename));
}

HTTPServer::Response HTTPServer::SessionResource::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  Resource *resource;
  if (!(resource = FindOrNull(connmap, c))) {
    resource = Open();
    connmap[c] = resource;
    connectionClosedCB(c, bind(&SessionResource::ConnectionClosedCB, this, _1));
  }
  return resource->Request(c, method, url, args, headers, postdata, postlen);
}

void HTTPServer::SessionResource::ConnectionClosedCB(Connection *c) {
  auto i = connmap.find(c);
  if (i == connmap.end()) return;
  Resource *resource = (*i).second;
  connmap.erase(i);
  Close(resource);
}

HTTPServer::Response HTTPServer::ConsoleResource::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  StringPiece v;
  if (args) HTTP::GrepURLArgs(args, 0, 1, "v", &v);
  app->focused->shell->Run(v.str());
  string response = StrCat("<html>Shell::run('", v.str(), "')<br/></html>\n");
  return HTTPServer::Response("text/html; charset=UTF-8", &response);
}

}; // namespace LFL
