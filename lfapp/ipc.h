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

#ifndef __LFL_LFAPP_IPC_H__
#define __LFL_LFAPP_IPC_H__

#ifdef LFL_FLATBUFFERS
#include "ipc_generated.h"
#endif

#define LFL_IPC_DEBUG
#ifdef LFL_IPC_DEBUG
#define IPCTrace(...) printf(__VA_ARGS__)
#else
#define IPCTrace(...)
#endif

namespace LFL {
#ifdef LFL_FLATBUFFERS
using flatbuffers::FlatBufferBuilder;
typedef pair<flatbuffers::unique_ptr_t, size_t> FlatBufferPiece;

template<typename T> FlatBufferPiece CreateFlatBuffer(const std::function<flatbuffers::Offset<T>(FlatBufferBuilder &fb)> &f) {
  FlatBufferBuilder fb;
  fb.Finish(f(fb));
  size_t size = fb.GetSize();
  return make_pair(fb.ReleaseBufferPointer(), size);
}
#define MakeFlatBufferOfType(t, x) CreateFlatBuffer(function<flatbuffers::Offset<t>(FlatBufferBuilder&)>([&](FlatBufferBuilder &fb){ return x; }))
#define MakeFlatBuffer(t, ...) MakeFlatBufferOfType(IPC::t, IPC::Create ## t(fb, __VA_ARGS__))
#define MakeResourceHandle(t, mpb) IPC::CreateResourceHandle(fb, t, (mpb).len, fb.CreateString((mpb).url))
#define SendRPC(c, seq, th, name, ...) Send ## name(c, seq, MakeFlatBuffer(name, __VA_ARGS__), th)
#define ErrorRPC(c, seq, th, name) Send ## name(c, seq, MakeFlatBufferOfType(IPC::name, IPC::Create ## name(fb)), th)
#define ExpectResponseRPC(name, ...) Expect ## name ## Response(new name ## Query(__VA_ARGS__))

#define RPC_SEND_DEFINITION(Request, Response, name) \
bool Send ## name ## Request(Connection *c, int seq, const FlatBufferPiece &q, int th=-1) { \
  bool ok = RPC::Write(c, InterProcessProtocol::name ## Request::Type, seq, \
                       StringPiece(reinterpret_cast<const char *>(q.first.get()), q.second), th); \
  IPCTrace("%s " #name " %s wrote=%d\n", rpc_table_name.c_str(), \
           InterProcessProtocol::name ## Request::DebugString \
           (flatbuffers::GetRoot<IPC::name ## Request>(q.first.get())).c_str(), ok); \
  return ok; \
}

#define RPC_TABLE_BEGIN(name) \
  typedef name Parent; \
  string rpc_table_name = #name; \
  void HandleRPC(Connection *c) { \
    while (c->rl >= RPC::Header::size) { \
      Serializable::ConstStream in(c->rb, c->rl); \
      RPC::Header hdr; \
      hdr.In(&in); \
      IPCTrace(#name " begin parse %d of %d bytes\n", hdr.len, c->rl); \
      if (c->rl < hdr.len) break; \
      switch (hdr.id) {

#define RPC_TABLE_CLIENT_QCALL(name, req, mpv) \
  IPCTrace("%s " #name " %s\n", rpc_table_name.c_str(), InterProcessProtocol::name ## Response::DebugString(req, mpv).c_str()); \
  switch (query->rpc_cb(query, req, mpv)) { \
    case RPC::Error: \
    case RPC::Done:   delete query; \
    case RPC::Accept: name ## _map.erase(hdr.seq); \
    default: break; \
  }

#define RPC_TABLE_SERVER_QCALL(name, req, mpv) \
  IPCTrace("%s " #name " %s\n", rpc_table_name.c_str(), InterProcessProtocol::name ## Request::DebugString(req, mpv).c_str()); \
  switch (Handle ## name ## Request(hdr.seq, req, mpv)) { \
    case RPC::Error: ErrorRPC(conn, hdr.seq, -1, name ## Response); \
    default: break; \
  }

#define RPC_TABLE_SERVER_CALL(name) case InterProcessProtocol::name ## Request::Type: { \
    RPC_TABLE_SERVER_QCALL(name, flatbuffers::GetRoot<IPC::name ## Request>(c->rb + RPC::Header::size), NULL); \
  } break;

#define RPC_TABLE_CLIENT_CALL(name) case InterProcessProtocol::name ## Response::Type: { \
    void *mpv = 0; \
    auto req = flatbuffers::GetRoot<IPC::name ## Response>(c->rb + RPC::Header::size); \
    auto query = FindOrNull(name ## _map, hdr.seq); \
    if (query) { RPC_TABLE_CLIENT_QCALL(name, req, mpv); } \
    else ERROR(#name " missing seq %d", hdr.seq); \
  } break;

#define RPC_TABLE_CLIENT_QXBC(name, mpv) case InterProcessProtocol::name ## Response::Type: { \
    auto req = flatbuffers::GetRoot<IPC::name ## Response>(c->rb + RPC::Header::size); \
    auto query = FindOrNull(name ## _map, hdr.seq); \
    MultiProcessBuffer mpb(conn, req->mpv()); \
    if (query && mpb.Open()) { RPC_TABLE_CLIENT_QCALL(name, req, mpb); } \
    else ERROR(#name " load MPB=%s failed", req->mpv()->url()->c_str()); \
    mpb.Close(); \
  } break;

#define RPC_TABLE_SERVER_VXBC(name, mpv) case InterProcessProtocol::name ## Request::Type: { \
    auto req = flatbuffers::GetRoot<IPC::name ## Request>(c->rb + RPC::Header::size); \
    MultiProcessBuffer mpb(conn, req->mpv()); \
    if (mpb.Open()) { RPC_TABLE_SERVER_QCALL(name, req, mpb); } \
    else ERROR(#name " load MPB=%s failed", req->mpv()->url()->c_str()); \
    mpb.Close(); \
  } break;

#define RPC_TABLE_CLIENT_QXRC(name, mpt, mpv) case InterProcessProtocol::name ## Response::Type: { \
    auto req = flatbuffers::GetRoot<IPC::name ## Response>(c->rb + RPC::Header::size); \
    auto query = FindOrNull(name ## _map, hdr.seq); \
    MultiProcessBuffer mpb(conn, req->mpv()); \
    mpt mpf; \
    if (query && req->mpv()->type() == mpt::Type && mpb.Open() && MultiProcessResource::Read(mpb, req->mpv()->type(), &mpf)) { RPC_TABLE_CLIENT_QCALL(name, req, mpf); } \
    else ERROR(#name " load " #mpt "=%s failed", req->mpv()->url()->c_str()); \
    mpb.Close(); \
  } break;

#define RPC_TABLE_SERVER_VXRC(name, mpt, mpv) case InterProcessProtocol::name ## Request::Type: { \
    auto req = flatbuffers::GetRoot<IPC::name ## Request>(c->rb + RPC::Header::size); \
    MultiProcessBuffer mpb(conn, req->mpv()); \
    mpt mpf; \
    if (req->mpv()->type() == mpt::Type && mpb.Open() && MultiProcessResource::Read(mpb, req->mpv()->type(), &mpf)) { RPC_TABLE_SERVER_QCALL(name, req, mpf); } \
    else ERROR(#name " load " #mpt "=%s failed", req->mpv()->url()->c_str()); \
    mpb.Close(); \
  } break;

#define RPC_TABLE_CLIENT_QIRC(name, mpt, mpv) case InterProcessProtocol::name ## Response::Type: { \
    auto req = flatbuffers::GetRoot<IPC::name ## Response>(c->rb + RPC::Header::size); \
    auto query = FindOrNull(name ## _map, hdr.seq); \
    MultiProcessBuffer *mpb = FindOrNull(ipc_buffer, req->mpv()); \
    mpt mpf; \
    if (mpb && mpb->buf && query && MultiProcessResource::Read(*mpb, mpt::Type, &mpf)) { RPC_TABLE_CLIENT_QCALL(name, req, mpf); }\
    else if (!query) ERROR(#name " missing seq %d", hdr.seq); \
    else ERROR(#name " load " #mpt "_id=%d failed", req->mpv()); \
  } break;

#define RPC_TABLE_SERVER_VIRC(name, mpt, mpv) case InterProcessProtocol::name ## Request::Type: { \
    auto req = flatbuffers::GetRoot<IPC::name ## Request>(c->rb + RPC::Header::size); \
    MultiProcessBuffer *mpb = FindOrNull(ipc_buffer, req->mpv()); \
    mpt mpf; \
    if (mpb && mpb->buf && MultiProcessResource::Read(*mpb, mpt::Type, &mpf)) { RPC_TABLE_SERVER_QCALL(name, req, mpf); } \
    else ERROR(#name " load " #mpt "_id=%d failed", req->mpv()); \
  } break;

#define RPC_TABLE_END(name) \
        default: FATAL("unknown id ", hdr.id); break; \
      } \
      IPCTrace(#name " flush %d bytes\n", hdr.len); \
      c->ReadFlush(hdr.len); \
    } \
  } \
  name()

#define RPC_CLIENT_CALL(name, mpt, ...) RPC_SEND_DEFINITION(Request, Response, name) \
  struct name ## Query; \
  struct name ## RPC { \
    typedef function<int(name ## Query*, const IPC::name ## Response*, mpt)> CB; \
    Parent *parent; \
    RPC::Seq seq; \
    CB rpc_cb; \
    virtual ~name ## RPC() {} \
    name ## RPC(Parent *P=0, RPC::Seq S=0, const CB &C=CB()) : parent(P), seq(S), rpc_cb(C) {} \
    int Done() { delete this; return RPC::Done; } \
  }; \
  unordered_map<RPC::Seq, name ## Query*> name ## _map; \
  int Expect ## name ## Response(name ## Query *q) { name ## _map[q->seq] = q; return RPC::Ok; } \
  void name(__VA_ARGS__); \
  struct name ## Query : public name ## RPC

#define RPC_SERVER_CALL(name, mpt) RPC_SEND_DEFINITION(Response, Request, name) \
  struct name ## Query; \
  struct name ## RPC { \
    Parent *parent; \
    RPC::Seq seq; \
    virtual ~name ## RPC() {} \
    name ## RPC(Parent *P=0, RPC::Seq S=0) : parent(P), seq(S) {} \
    int Done() { delete this; return RPC::Done; } \
  }; \
  int Handle ## name ## Request(int, const IPC::name ## Request*, mpt); \
  struct name ## Query : public name ## RPC

#define IPC_TABLE_ENTRY(id, name, mpt, ...) struct name { \
    static const int Type = id; \
    static string DebugString(const IPC::name *x, const mpt &mpv=mpt()) { return StrCat(#name " ", __VA_ARGS__); } \
  };

#else
namespace IPC {
struct Color { unsigned char r() const { return 0; } unsigned char g() const { return 0; } unsigned char b() const { return 0; } unsigned char a() const { return 0; } };
struct FontDescription { string *name() const { return 0; } string *family() const { return 0; } Color *fg() const { return 0; } Color *bg() const { return 0; } int engine() const { return 0; } int size() const { return 0; } int flag() const { return 0; } bool unicode() const { return 0; } };
struct ResourceHandle { int len() const { return 0; } int type() const { return 0; } string *url() const { return 0; } };
struct LoadAssetRequest { ResourceHandle *mpb() const { return 0; } };
struct LoadAssetResponse {};
struct AllocateBufferResponse {};
struct OpenSystemFontResponse {};
struct PaintResponse {};
struct NavigateResponse {};
struct WGetResponse {};
}; // namespace IPC
#define RPC_TABLE_BEGIN(name) typedef name Parent; void HandleRPC(Connection *c) {}
#define RPC_TABLE_CLIENT_CALL(name)
#define RPC_TABLE_SERVER_CALL(name)
#define RPC_TABLE_CLIENT_QXBC(name, mpv)
#define RPC_TABLE_SERVER_VXBC(name, mpv)
#define RPC_TABLE_CLIENT_QXRC(name, mpt, mpv)
#define RPC_TABLE_SERVER_VXRC(name, mpt, mpv)
#define RPC_TABLE_CLIENT_QIRC(name, mpt, mpv)
#define RPC_TABLE_SERVER_VIRC(name, mpt, mpv)
#define RPC_TABLE_END(name) name()
#define RPC_CLIENT_CALL(name, mpt, ...) \
  struct name ## Query; \
  struct name ## RPC { \
    typedef function<int(name ## Query*, const IPC::name ## Response*, mpt)> CB; \
    name ## RPC(Parent *P=0, RPC::Seq S=0, const CB &C=CB()) {} \
  }; \
  void name(__VA_ARGS__); \
  struct name ## Query : public name ## RPC
#define RPC_SERVER_CALL(name, mpt) \
  struct name ## RPC { name ## RPC(Parent *P, RPC::Seq S=0) {} }; \
  struct name ## Query : public name ## RPC
#define IPC_TABLE_ENTRY(id, name, mpt, ...) struct name { \
    static const int Type = id; \
    static string DebugString(const void *x, const mpt &mpv=mpt()) { return ""; } \
  };
#endif

struct NTService {
  static int Install  (const char *name, const char *path);
  static int Uninstall(const char *name);
  static int WrapMain (const char *name, MainCB main_cb, int argc, const char **argv);
};

struct ProcessPipe {
  int pid=0;
  FILE *in=0, *out=0;
  virtual ~ProcessPipe() { Close(); }
  int Open(const char **argv, const char *startdir=0);
  int OpenPTY(const char **argv, const char *startdir=0);
  int Close();
};

struct MultiProcessBuffer {
  string url;
  char *buf=0;
  int len=0, transfer_handle=-1;
#ifdef WIN32
  HANDLE impl = INVALID_HANDLE_VALUE, share_process = INVALID_HANDLE_VALUE;
#else
  int impl = -1; void *share_process = 0;
#endif
  MultiProcessBuffer(void *share_with=0) : share_process(share_with) {}
  MultiProcessBuffer(Connection *c, const IPC::ResourceHandle *h);
  virtual ~MultiProcessBuffer();
  virtual void Close();
  virtual bool Open();
  bool Create(int s) { len=s; return Open(); }
  bool Create(const Serializable &s) { bool ret; if ((ret = Create(Size(s)))) s.ToString(buf, len, 0); return ret; }
  bool Copy(const Serializable &s) { bool ret; if ((ret = len >= Size(s))) s.ToString(buf, len, 0); return ret; }
  static int Size(const Serializable &s) { return Serializable::Header::size + s.Size(); }
};

struct RPC {
  typedef unsigned short Id;
  typedef unsigned short Seq;
  enum Status { Error=0, Ok=1, Done=2, Accept=3 };
  struct Header {
    static const int size = 8;
    int len; Id id; Seq seq;
    void Out(Serializable::Stream *o) const { o->Htonl( len); o->Htons( id); o->Htons( seq); }
    void In(const Serializable::Stream *i)  { i->Ntohl(&len); i->Ntohs(&id); i->Ntohs(&seq); }
  };
  static bool Write(Connection *conn, Id id, Seq seq, const StringPiece &rpc_text, int transfer_handle=-1);
};

struct InterProcessProtocol {
  IPC_TABLE_ENTRY( 1, AllocateBufferRequest,  Void,                        "bytes=", x->bytes());
  IPC_TABLE_ENTRY( 2, AllocateBufferResponse, MultiProcessBuffer,          "mpb_id=", x->mpb_id());
  IPC_TABLE_ENTRY( 3, OpenSystemFontRequest,  Void,                        FontDesc(x->desc() ? *x->desc() : FontDesc()).DebugString());
  IPC_TABLE_ENTRY( 4, OpenSystemFontResponse, MultiProcessBuffer,          "num_glyphs=", x->num_glyphs()); 
  IPC_TABLE_ENTRY( 5, LoadAssetRequest,       MultiProcessFileResource,    "fn=", BlankNull(mpv.name.buf), ", len=", mpv.buf.len);
  IPC_TABLE_ENTRY( 6, LoadAssetResponse,      MultiProcessTextureResource, ""); 
  IPC_TABLE_ENTRY( 7, PaintRequest,           MultiProcessPaintResource,   ""); 
  IPC_TABLE_ENTRY( 8, PaintResponse,          Void,                        "");
  IPC_TABLE_ENTRY( 9, WGetRequest,            Void,                        "url=", x->url() ? x->url()->data() : ""); 
  IPC_TABLE_ENTRY(10, WGetResponse,           MultiProcessBuffer,          x->headers(), " ", mpv.buf!=0, " ", mpv.len, " ", mpv.url.c_str());
  IPC_TABLE_ENTRY(11, NavigateRequest,        Void,                        "url=", x->url() ? x->url()->data() : ""); 
  IPC_TABLE_ENTRY(12, NavigateResponse,       Void,                        "");
};

struct ProcessAPIClient {
  struct ConnectionHandler : public Connection::Handler {
    ProcessAPIClient *parent;
    ConnectionHandler(ProcessAPIClient *P) : parent(P) {}
    int Read(Connection *c) { parent->HandleRPC(c); return 0; }
  };

  RPC::Seq seq=0;
  Connection *conn=0;
  void *server_process=0;
  int pid=0, ipc_buffer_id=0;
  unordered_map<int, MultiProcessBuffer*> ipc_buffer;
  vector<Drawable*> drawable;
  void StartServer(const string &server_program);

  RPC_TABLE_BEGIN(ProcessAPIClient);
  RPC_TABLE_CLIENT_CALL(Navigate);
  RPC_TABLE_CLIENT_QIRC(LoadAsset, MultiProcessTextureResource, mpb_id);
  RPC_TABLE_SERVER_CALL(AllocateBuffer);
  RPC_TABLE_SERVER_CALL(OpenSystemFont);
  RPC_TABLE_SERVER_VIRC(Paint, MultiProcessPaintResource, mpb_id);
  RPC_TABLE_SERVER_CALL(WGet);
  RPC_TABLE_END(ProcessAPIClient) {};

  RPC_CLIENT_CALL(Navigate, Void, const string&) {};
  RPC_CLIENT_CALL(LoadAsset, const MultiProcessTextureResource&, const string&, const string&, const LoadAssetRPC::CB&) { using LoadAssetRPC::LoadAssetRPC; };
  RPC_SERVER_CALL(AllocateBuffer, Void) {};
  RPC_SERVER_CALL(OpenSystemFont, Void) {};
  RPC_SERVER_CALL(Paint, const MultiProcessPaintResource&) {};
  RPC_SERVER_CALL(WGet, Void) {
    using WGetRPC::WGetRPC;
    void WGetResponseCB(Connection *c, const char *h, const string &ct, const char *b, int l);
  };
};

struct ProcessAPIServer {
  Browser *browser=0;
  Connection *conn=0;
  RPC::Seq seq=0;

  void Start(const string &socket_name);
  void HandleMessagesLoop() { while (app->run) if (!HandleMessages()) break; }
  bool HandleMessages() {
    if (!NBReadable(conn->socket, -1)) return true;
    if (conn->Read() <= 0) return ERRORv(false, conn->Name(), ": read ");
    HandleRPC(conn);
    return true;
  }

  RPC_TABLE_BEGIN(ProcessAPIServer);
  RPC_TABLE_CLIENT_QXBC(AllocateBuffer, mpb);
  RPC_TABLE_CLIENT_CALL(OpenSystemFont);
  RPC_TABLE_CLIENT_CALL(Paint);
  RPC_TABLE_CLIENT_QXBC(WGet, mpb);
  RPC_TABLE_SERVER_CALL(Navigate);
  RPC_TABLE_SERVER_VXRC(LoadAsset, MultiProcessFileResource, mpb);
  RPC_TABLE_END(ProcessAPIServer) {};

  RPC_CLIENT_CALL(AllocateBuffer, MultiProcessBuffer&, int, int) { using AllocateBufferRPC::AllocateBufferRPC; };
  RPC_CLIENT_CALL(OpenSystemFont, Void, const FontDesc &fd, Font *out) {};
  RPC_CLIENT_CALL(Paint, Void, const point &tile, const MultiProcessPaintResourceBuilder &list) {};
  RPC_CLIENT_CALL(WGet, const MultiProcessBuffer&, const string&, const HTTPClient::ResponseCB &) {
    HTTPClient::ResponseCB cb; 
    WGetQuery(Parent *P, RPC::Seq S, const WGetRPC::CB &C, const HTTPClient::ResponseCB &R) : WGetRPC(P,S,C), cb(R) {}
    static int WGetResponse(WGetQuery*, const IPC::WGetResponse*, const MultiProcessBuffer&);
  };
  RPC_SERVER_CALL(Navigate, Void) {};
  RPC_SERVER_CALL(LoadAsset, const MultiProcessFileResource&) {
    Texture *tex; 
    LoadAssetQuery(Parent *P, RPC::Seq S, Texture *T) : LoadAssetRPC(P,S), tex(T) {}
    int AllocateBufferResponse(AllocateBufferQuery*, const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };
};

struct TilesIPC : public TilesT<MultiProcessPaintResource::PaintCmd, MultiProcessPaintResourceBuilder> {
  const Drawable::Attr *attr=0;
  void SetAttr           (const Drawable::Attr*);
  void InitDrawBox       (const point&);
  void InitDrawBackground(const point&);
  void DrawBox           (const Drawable*, const Box&, const Drawable::Attr *a=0);
  void DrawBackground    (const Box&);
  void AddScissor        (const Box&);
};
typedef LayersT<TilesIPC> LayersIPC;

}; // namespace LFL
#endif // __LFL_LFAPP_IPC_H__
