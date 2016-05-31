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

#ifndef LFL_CORE_APP_NET_HTTP_H__
#define LFL_CORE_APP_NET_HTTP_H__
namespace LFL {
  
struct HTTP {
  static bool ParseHost(const char *host, const char *host_end, string *hostO, string *portO);
  static bool ResolveHost(const char *host, const char *host_end, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int defport=0);
  static bool ResolveEndpoint(const string &host, const string &port, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int defport=0);
  static bool ParseURL(const char *url, string *protO, string *hostO, string *portO, string *pathO);
  static bool ResolveURL(const char *url, bool *ssl, IPV4::Addr *ipv4_addr, int *tcp_port, string *host, string *path, int defport=0, string *prot=0);
  static string HostURL(const char *url);

  static int ParseRequest(char *buf, char **methodO, char **urlO, char **argsO, char **verO);
  static       char *FindHeadersStart(      char *buf);
  static       char *FindHeadersEnd  (      char *buf);
  static const char *FindHeadersEnd  (const char *buf);
  static int GetHeaderLen(const char *beg, const char *end);
  static int GetHeaderNameLen(const char *beg);
  static int GetURLArgNameLen(const char *beg);
  static string GrepHeaders(const char *headers, const char *headers_end, const string &name);
  static int    GrepHeaders(const char *headers, const char *headers_end, int num, ...);
  static int    GrepURLArgs(const char *urlargs, const char *urlargs_end, int num, ...);
  static string EncodeURL(const char *url);
  static string MakeHeaders(int code, int content_length, const char *content_type);
};

struct HTTPClient {
  typedef function<void(Connection*, const char*, const string&, const char*, int)> ResponseCB;
  static bool WGet(const string &url, File *out=0, const ResponseCB &responseCB=ResponseCB(), const StringCB &redirectCB=StringCB());
  static bool WPost(const string &url, const string &mimetype, const char *postdata, int postlen, ResponseCB=ResponseCB());
  static Connection *PersistentConnection(const string &url, string *hostOut, string *pathOut, ResponseCB responseCB);
  static int WriteRequest(Connection *c, int method, const char *host, const char *path, const char *postmime, const char *postdata, int postlen, bool persist);
};

struct HTTPServer : public Service {
  typedef function<void(Connection*)> ConnectionClosedCB;
  struct Method {
    enum { GET=1, POST=2 };
    static const char *name(int n);
  };

  struct Response {
    int code, content_length;
    const char *type, *content;
    string type_buf, content_buf;
    Connection::Handler *refill;
    bool write_headers;
    Response(     const string *T, const string        *C)            : code(200), content_length(C->size()), type_buf(*T), content_buf(*C), refill(0), write_headers(1)  { content=content_buf.c_str(); type=type_buf.c_str(); }
    Response(       const char *T, const string        *C)            : code(200), content_length(C->size()), type(T),      content_buf(*C), refill(0), write_headers(1)  { content=content_buf.c_str(); }
    Response(       const char *T, const char          *C)            : code(200), content_length(strlen(C)), type(T),      content(C),      refill(0), write_headers(1)  {}
    Response(       const char *T, const StringPiece   &C)            : code(200), content_length(C.len),     type(T),      content(C.buf),  refill(0), write_headers(1)  {}
    Response(int K, const char *T, const StringPiece   &C)            : code(K),   content_length(C.len),     type(T),      content(C.buf),  refill(0), write_headers(1)  {}
    Response(int K, const char *T, const char          *C)            : code(K),   content_length(strlen(C)), type(T),      content(C),      refill(0), write_headers(1)  {}
    Response(const char *T, int L, Connection::Handler *C, bool WH=1) : code(200), content_length(L),         type(T),      content(0),      refill(C), write_headers(WH) {}
    static Response _400;
  };

  struct Resource {
    virtual ~Resource() {}
    virtual Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) = 0;
  };

  map<string, Resource*> urlmap;
  HTTPServer(IPV4::Addr addr, int port, bool SSL) : Service("HTTPServer", Protocol::TCP) { QueueListen(addr, port, SSL); }
  HTTPServer(                 int port, bool SSL) : Service("HTTPServer", Protocol::TCP) { QueueListen(0,    port, SSL); }
  virtual ~HTTPServer() { ClearURL(); }

  int Connected(Connection *c);
  void AddURL(const string &url, Resource *resource) { urlmap[url] = resource; }
  void ClearURL() { for (auto i = urlmap.begin(); i != urlmap.end(); i++) delete (*i).second; }
  virtual Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
  static void connectionClosedCB(Connection *conn, ConnectionClosedCB cb);

  struct DebugResource : public Resource {
    Response Request(Connection *c, int, const char *url, const char *args, const char *hdrs, const char *postdata, int postlen);
  };

  struct StringResource : public Resource {
    Response val;
    StringResource(const char *T, const StringPiece &C) : val(T, C) {}
    StringResource(const char *T, const char        *C) : val(T, C) {}
    Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) { return val; }
  };

  struct FileResource : public Resource {
    string filename;
    const char *type;
    int size=0;

    FileResource(const string &fn, const char *mimetype=0);
    Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
  };

  struct SessionResource : public Resource {
    map<Connection*, Resource*> connmap;
    virtual ~SessionResource() { ClearConnections(); }
    virtual Resource *Open() = 0;
    virtual void Close(Resource *) = 0;

    void ClearConnections() { for (auto i = connmap.begin(); i != connmap.end(); i++) delete (*i).second; }
    Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
    void ConnectionClosedCB(Connection *c);
  };

  struct ConsoleResource : public Resource {
    HTTPServer::Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
  };

  struct StreamResource : public Resource {
    AVFormatContext *fctx;
    map<void*, Connection*> subscribers;
    bool open=0;
    int abr=0, vbr=0;
    AVStream *audio=0;
    AVFrame *samples=0;
    short *sample_data=0;
    AudioResamplerInterface *resampler=0;
    int frame=0, channels=0, resamples_processed=0;
    AVStream *video=0;
    AVFrame *picture=0;
    SwsContext *conv=0;

    virtual ~StreamResource();
    StreamResource(const char *outputFileType, int audioBitRate, int videoBitRate);        
    Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen);
    void OpenStreams(bool audio, bool video);
    void Update(int audio_samples, bool video_sample);
    void ResampleAudio(int audio_samples);
    void SendAudio();
    void SendVideo();        
    void Broadcast(AVPacket *, microseconds timestamp);
  };
};

}; // namespace LFL
#endif // LFL_CORE_APP_NET_HTTP_H__
