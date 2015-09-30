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
#define MakeFlatBuffer(t, ...) MakeFlatBufferOfType(t, Create ## t(fb, __VA_ARGS__))
#define MakeResourceHandle(t, mpb) CreateResourceHandle(fb, t, (mpb).len, fb.CreateString((mpb).url))
#define SendRPC(c, seq, th, name, ...) Send ## name(c, seq, MakeFlatBuffer(name, __VA_ARGS__), th)

#define RPC_TABLE_BEGIN(name) \
  string rpc_table_name = #name; \
  struct RpcQuery { \
    name *parent; \
    RPC::Seq seq; \
    RpcQuery(name *P=0, RPC::Seq S=0) : parent(P), seq(S) {} \
  }; \
  void HandleRPC(Connection *c) { \
    while (c->rl >= RPC::Header::size) { \
      Serializable::ConstStream in(c->rb, c->rl); \
      RPC::Header hdr; \
      hdr.In(&in); \
      IPCTrace(#name " begin parse %d of %d bytes\n", hdr.len, c->rl); \
      if (c->rl < hdr.len) break; \
      switch (hdr.id) {

#define RPC_TABLE_CALL(Request, Response, name) case InterProcessProtocol::name ## Response::Type: { \
  Handle ## name ## Response(c, hdr.seq, flatbuffers::GetRoot<name ## Response>(c->rb + RPC::Header::size));

#define RPC_TABLE_MPBC(Request, Response, name, mpv) case InterProcessProtocol::name ## Response::Type: { \
  auto req = flatbuffers::GetRoot<name ## Response>(c->rb + RPC::Header::size); \
  MultiProcessBuffer mpb(conn, req->mpv()); \
  if (mpb.Open()) Handle ## name ## Response(c, hdr.seq, req, mpb); \
  else ERROR(#name " load MPB=%s failed", req->mpv()->url()->c_str()); \
  mpb.Close();

#define RPC_TABLE_MPFC(Request, Response, name, mpv) case InterProcessProtocol::name ## Response::Type: { \
  auto req = flatbuffers::GetRoot<name ## Response>(c->rb + RPC::Header::size); \
  MultiProcessFileResource mpf; \
  MultiProcessBuffer mpb(conn, req->mpv()); \
  if (req->mpv()->type() == MultiProcessFileResource::Type && mpb.Open() && \
      MultiProcessResource::Read(mpb, req->mpv()->type(), &mpf)) Handle ## name ## Response(c, hdr.seq, req, mpf); \
  else ERROR(#name " load MPF=%s failed", req->mpv()->url()->c_str()); \
  mpb.Close();

#define RPC_TABLE_CLIENT_CALL(name) RPC_TABLE_CALL(Request, Response, name); } break;
#define RPC_TABLE_SERVER_CALL(name) RPC_TABLE_CALL(Response, Request, name); } break;

#define RPC_TABLE_CLIENT_MPBC(name, mpv) RPC_TABLE_MPBC(Request, Response, name, mpv); } break;
#define RPC_TABLE_SERVER_MPBC(name, mpv) RPC_TABLE_MPBC(Response, Request, name, mpv); } break;

#define RPC_TABLE_CLIENT_MPFC(name, mpv) RPC_TABLE_MPFC(Request, Response, name, mpv); } break;
#define RPC_TABLE_SERVER_MPFC(name, mpv) RPC_TABLE_MPFC(Response, Request, name, mpv); } break;

#define RPC_TABLE_END(name) \
        default: FATAL("unknown id ", hdr.id); break; \
      } \
      IPCTrace(#name " flush %d bytes\n", hdr.len); \
      c->ReadFlush(hdr.len); \
    } \
  } \
  name()

#define RPC_SEND(Request, Response, name) \
bool Send ## name ## Request(Connection *c, int seq, const FlatBufferPiece &q, int th=-1) { \
  bool ok = RPC::Write(c, InterProcessProtocol::name ## Request::Type, seq, \
                       StringPiece(reinterpret_cast<const char *>(q.first.get()), q.second), th); \
  IPCTrace("RPC Client " #name " " #Request " wrote=%d\n", ok); \
  return ok; \
}

#define RPC_CALL(Request, Response, name) void Handle ## name ## Response(Connection*, int, const name ## Response*)
#define RPC_MPBC(Request, Response, name) void Handle ## name ## Response(Connection*, int, const name ## Response*, MultiProcessBuffer&)
#define RPC_MPFC(Request, Response, name) void Handle ## name ## Response(Connection*, int, const name ## Response*, const MultiProcessFileResource&)

#define RPC_SERVER_CALL_IMPL(name) RPC_SEND(Response, Request, name) \
  struct name ## Query : public RpcQuery

#define RPC_CLIENT_CALL_IMPL(name, ...) RPC_SEND(Request, Response, name) \
  void name(__VA_ARGS__); \
  struct name ## Query; \
  unordered_map<RPC::Seq, name ## Query*> name ## query__map; \
  struct name ## Query : public RpcQuery

#define RPC_CLIENT_CALL(name, ...) RPC_CALL(Request, Response, name); RPC_CLIENT_CALL_IMPL(name, __VA_ARGS__)
#define RPC_SERVER_CALL(name)      RPC_CALL(Response, Request, name); RPC_SERVER_CALL_IMPL(name)

#define RPC_CLIENT_MPBC(name, ...) RPC_MPBC(Request, Response, name); RPC_CLIENT_CALL_IMPL(name, __VA_ARGS__)
#define RPC_SERVER_MPBC(name)      RPC_MPBC(Response, Request, name); RPC_SERVER_CALL_IMPL(name)

#define RPC_CLIENT_MPFC(name, ...) RPC_MPFC(Request, Response, name); RPC_CLIENT_CALL_IMPL(name, __VA_ARGS__)
#define RPC_SERVER_MPFC(name)      RPC_MPFC(Response, Request, name); RPC_SERVER_CALL_IMPL(name)
#else
struct ResourceHandle { int len() const { return 0; } int type() const { return 0; } string *url() const { return 0; } };
struct LoadResourceRequest { ResourceHandle *mpb() const { return 0; } };
#define RPC_TABLE_BEGIN(name) void HandleRPC(Connection *c) {} struct RpcQuery { RpcQuery(name *P=0, RPC::Seq S=0) {} }
#define RPC_TABLE_CLIENT_CALL(name)
#define RPC_TABLE_SERVER_CALL(name)
#define RPC_TABLE_CLIENT_MPBC(name, mpv)
#define RPC_TABLE_SERVER_MPBC(name, mpv)
#define RPC_TABLE_CLIENT_MPFC(name, mpv)
#define RPC_TABLE_SERVER_MPFC(name, mpv)
#define RPC_TABLE_END(name) name()
#define RPC_CLIENT_CALL(name, ...) void name(__VA_ARGS__); struct name ## Query : public RpcQuery
#define RPC_SERVER_CALL(name)                              struct name ## Query : public RpcQuery
#define RPC_CLIENT_MPBC(name, ...) void name(__VA_ARGS__); struct name ## Query : public RpcQuery
#define RPC_SERVER_MPBC(name)                              struct name ## Query : public RpcQuery
#define RPC_CLIENT_MPFC(name, ...) void name(__VA_ARGS__); struct name ## Query : public RpcQuery
#define RPC_SERVER_MPFC(name)                              struct name ## Query : public RpcQuery
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
  MultiProcessBuffer(void *share_with) : share_process(share_with) {}
  MultiProcessBuffer(Connection *c, const ResourceHandle *h);
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
  struct Header {
    static const int size = 8;
    int len; Id id; Seq seq;
    void Out(Serializable::Stream *o) const { o->Htonl( len); o->Htons( id); o->Htons( seq); }
    void In(const Serializable::Stream *i)  { i->Ntohl(&len); i->Ntohs(&id); i->Ntohs(&seq); }
  };
  static bool Write(Connection *conn, Id id, Seq seq, const StringPiece &rpc_text, int transfer_handle=-1);
};

struct InterProcessProtocol {
  struct LoadResourceRequest      { static const int Type = 1; };
  struct LoadResourceResponse     { static const int Type = 2; };
  struct AllocateBufferRequest    { static const int Type = 3; };
  struct AllocateBufferResponse   { static const int Type = 4; };
  struct WGetRequest              { static const int Type = 5; };
  struct WGetResponse             { static const int Type = 6; };
  struct NavigateRequest          { static const int Type = 7; };
  struct NavigateResponse         { static const int Type = 8; };
};

struct ProcessAPIClient {
  typedef function<void(const MultiProcessTextureResource&)> LoadResourceCompleteCB;
  struct ConnectionHandler : public Connection::Handler {
    ProcessAPIClient *parent;
    ConnectionHandler(ProcessAPIClient *P) : parent(P) {}
    int Read(Connection *c) { parent->HandleRPC(c); return 0; }
  };

  RPC_TABLE_BEGIN(ProcessAPIClient);
  RPC_TABLE_SERVER_CALL(AllocateBuffer);
  RPC_TABLE_SERVER_CALL(WGet);
  RPC_TABLE_CLIENT_CALL(Navigate);
  RPC_TABLE_CLIENT_CALL(LoadResource);
  RPC_TABLE_END(ProcessAPIClient) {};

  RPC_SERVER_CALL(AllocateBuffer) {};
  RPC_SERVER_CALL(WGet) {
    bool unused=0;
    WGetQuery(ProcessAPIClient *p=0, RPC::Seq s=0) : RpcQuery(p, s) {}
    void WGetResponseCB(Connection *c, const char *h, const string &ct, const char *b, int l);
  };
  RPC_CLIENT_CALL(Navigate, const string&) {};
  RPC_CLIENT_CALL(LoadResource, const string&, const string&, const LoadResourceCompleteCB&) {
    LoadResourceCompleteCB cb;
    LoadResourceQuery(const LoadResourceCompleteCB &c) : cb(c) {}
  };

  RPC::Seq seq=0;
  Connection *conn=0;
  void *server_process=0;
  int pid=0, ipc_buffer_id=0;
  bool reply_success_from_main_thread=true;
  unordered_map<int, MultiProcessBuffer*> ipc_buffer;
  unordered_map<RPC::Seq, LoadResourceQuery*> reqmap;

  void StartServer(const string &server_program);
  void LoadResourceSucceeded(LoadResourceQuery *query, const MultiProcessTextureResource &tex);
};

struct ProcessAPIServer {
  Browser *browser=0;
  Connection *conn=0;
  RPC::Seq seq=0;
  unordered_map<RPC::Seq, Texture*>               resmap;
  unordered_map<RPC::Seq, HTTPClient::ResponseCB> wgetmap;

  void Start(const string &socket_name);
  void HandleMessagesLoop() { while (app->run) if (!HandleMessages()) break; }
  bool HandleMessages() {
    if (!NBReadable(conn->socket, -1)) return true;
    if (conn->Read() <= 0) return ERRORv(false, conn->Name(), ": read ");
    HandleRPC(conn);
    return true;
  }

  RPC_TABLE_BEGIN(ProcessAPIServer);
  RPC_TABLE_SERVER_CALL(Navigate);
  RPC_TABLE_SERVER_MPFC(LoadResource, mpb);
  RPC_TABLE_CLIENT_MPBC(AllocateBuffer, mpb);
  RPC_TABLE_CLIENT_MPBC(WGet, mpb);
  RPC_TABLE_END(ProcessAPIServer) {};

  RPC_SERVER_CALL(Navigate) {};
  RPC_SERVER_MPFC(LoadResource) { Texture *tex=0; };
  RPC_CLIENT_MPBC(AllocateBuffer, int, int) {};
  RPC_CLIENT_MPBC(WGet, const string&, const HTTPClient::ResponseCB &) { HTTPClient::ResponseCB cb; };
};

}; // namespace LFL
#endif // __LFL_LFAPP_IPC_H__
