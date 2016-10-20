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

#ifndef LFL_CORE_APP_RPC_H__
#define LFL_CORE_APP_RPC_H__

#ifdef LFL_IPC_DEBUG
#define IPCTrace(...) DebugPrintf(stderr, __VA_ARGS__)
#else
#define IPCTrace(...)
#endif

namespace LFL {
#ifdef LFL_IPC
#define MakeResourceHandle(t, mpb) IPC::CreateResourceHandle(fb, t, (mpb).len, fb.CreateString((mpb).url))
#define MakeIPC(t, ...) MakeFlatBufferOfType(IPC::t, IPC::Create ## t(fb, __VA_ARGS__))
#define SendIPC(c, seq, th, name, ...) SendIPCRequest<Protocol::name>(c, seq, MakeIPC(name, __VA_ARGS__), th)
#define ErrorIPC(c, seq, th, name) SendIPCRequest<Protocol::name>(c, seq, MakeFlatBufferOfType(IPC::name, IPC::Create ## name(fb)), th)
#define ExpectResponseIPC(name, ...) Expect ## name ## Response(new name ## Query(__VA_ARGS__))

#define IPC_BUILDER_GET_REQ(name) auto req = flatbuffers::GetRoot<IPC::name>(b.data())
#define IPC_BUILDER_GET_QUERY(name) auto query = FindOrNull(name ## _map, hdr.seq); \
  if (!query) { ERROR(#name " missing seq ", hdr.seq); break; }

#define IPC_BUILDER_CLOSE_MPB if (mpb.len) mpb.Close()
#define IPC_BUILDER_OPEN_MPB_BY_XFD(name, mpv) \
  MultiProcessBuffer mpb(req->mpv(), xfer_fd); \
  if (mpb.len) mpb.Open();

#define IPC_BUILDER_CLOSE_MPT mpb.Close()
#define IPC_BUILDER_OPEN_MPT_BY_XFD(name, mpt, mpv) \
  MultiProcessBuffer mpb(req->mpv(), xfer_fd); \
  mpt mpf; \
  if (req->mpv()->type() != mpt::Type || !mpb.Open() || !MultiProcessResource::Read(mpb, req->mpv()->type(), &mpf)) \
  { ERROR(#name " load " #mpt "=", req->mpv()->url()->c_str(), " failed"); break; }

#define IPC_BUILDER_OPEN_MPT_BY_ID(name, mpt, mpv) \
  MultiProcessBuffer *mpb = FindOrNull(ipc_buffer, req->mpv()); \
  mpt mpf; \
  if (!mpb || !mpb->buf || !MultiProcessResource::Read(*mpb, mpt::Type, &mpf)) \
  { ERROR(#name " load " #mpt "_id=", req->mpv(), " failed"); break; }

#define IPC_BUILDER_CLIENT_RUN(name, req, mpv) if (!query->Run(req, mpv)) name ## _map.erase(hdr.seq);

#define IPC_BUILDER_SERVER_RUN(name, req, mpv) \
  IPCTrace("%s Receive " #name "Request seq=%d %s\n", ipc_name.c_str(), hdr.seq, Protocol::name ## Request::DebugString(req, mpv).c_str()); \
  switch (Handle ## name ## Request(hdr.seq, req, mpv)) { \
    case IPC::Error: ErrorIPC(conn, hdr.seq, -1, name ## Response); \
    default: break; \
  }

#define IPC_TABLE_BEGIN(name) \
  typedef name Parent; \
  void HandleIPCMessage(const IPC::Header &hdr, const StringPiece &b, int xfer_fd) { \
    switch (hdr.id) {

#define IPC_TABLE_SERVER_CALL(name) case Protocol::name ## Request::Id: { \
    IPC_BUILDER_GET_REQ(name ## Request); \
    IPC_BUILDER_SERVER_RUN(name, req, 0); \
  } break;

#define IPC_TABLE_CLIENT_CALL(name) case Protocol::name ## Response::Id: { \
    IPC_BUILDER_GET_REQ(name ## Response); \
    IPC_BUILDER_GET_QUERY(name); \
    IPC_BUILDER_CLIENT_RUN(name, req, 0); \
  } break;

#define IPC_TABLE_CLIENT_QXBC(name, mpv) case Protocol::name ## Response::Id: { \
    IPC_BUILDER_GET_REQ(name ## Response); \
    IPC_BUILDER_GET_QUERY(name); \
    IPC_BUILDER_OPEN_MPB_BY_XFD(name, mpv); \
    IPC_BUILDER_CLIENT_RUN(name, req, mpb); \
    IPC_BUILDER_CLOSE_MPB; \
  } break;

#define IPC_TABLE_SERVER_VXBC(name, mpv) case Protocol::name ## Request::Id: { \
    IPC_BUILDER_GET_REQ(name ## Request); \
    IPC_BUILDER_OPEN_MPB_BY_XFD(name, mpv); \
    IPC_BUILDER_SERVER_RUN(name, req, mpb); \
    IPC_BUILDER_CLOSE_MPB; \
  } break;

#define IPC_TABLE_CLIENT_QXRC(name, mpt, mpv) case Protocol::name ## Response::Id: { \
    IPC_BUILDER_GET_REQ(name ## Response); \
    IPC_BUILDER_GET_QUERY(name); \
    IPC_BUILDER_OPEN_MPT_BY_XFD(name, mpt, mpv); \
    IPC_BUILDER_CLIENT_RUN(name, req, mpf); \
    IPC_BUILDER_CLOSE_MPT; \
  } break;

#define IPC_TABLE_SERVER_VXRC(name, mpt, mpv) case Protocol::name ## Request::Id: { \
    IPC_BUILDER_GET_REQ(name ## Request); \
    IPC_BUILDER_OPEN_MPT_BY_XFD(name, mpt, mpv); \
    IPC_BUILDER_SERVER_RUN(name, req, mpf); \
    IPC_BUILDER_CLOSE_MPT; \
  } break;

#define IPC_TABLE_CLIENT_QIRC(name, mpt, mpv) case Protocol::name ## Response::Id: { \
    IPC_BUILDER_GET_REQ(name ## Response); \
    IPC_BUILDER_GET_QUERY(name); \
    IPC_BUILDER_OPEN_MPT_BY_ID(name, mpt, mpv); \
    IPC_BUILDER_CLIENT_RUN(name, req, mpf); \
  } break;

#define IPC_TABLE_SERVER_VIRC(name, mpt, mpv) case Protocol::name ## Request::Id: { \
    IPC_BUILDER_GET_REQ(name ## Request); \
    IPC_BUILDER_OPEN_MPT_BY_ID(name, mpt, mpv); \
    IPC_BUILDER_SERVER_RUN(name, req, mpf); \
  } break;

#define IPC_TABLE_END(name) \
      default: FATAL("unknown id ", hdr.id); break; \
    } \
  } \

#define IPC_CLIENT_CALL(name, mpt, ...) \
  struct name ## Query; \
  typedef ClientQuery<Parent, Protocol::name ## Response, mpt> name ## IPC; \
  unordered_map<IPC::Seq, name ## Query*> name ## _map; \
  int Expect ## name ## Response(name ## Query *q) { name ## _map[q->seq] = q; return IPC::Ok; } \
  void name(__VA_ARGS__); \
  struct name ## Query : public name ## IPC

#define IPC_SERVER_CALL(name, mpt) \
  typedef ServerQuery<Parent> name ## IPC; \
  int Handle ## name ## Request(int, const IPC::name ## Request*, mpt); \
  struct name ## Query : public name ## IPC

#define IPC_PROTO_ENTRY(id, name, mpt, ...) struct name { \
    typedef IPC::name Type; \
    static const int Id = id; \
    static const char *Name() { return #name; } \
    static string DebugString(const IPC::name *x, const mpt &mpv=mpt()) { return StrCat("{", __VA_ARGS__, "}"); } \
  };

#else /* LFL_IPC */
#define MakeIPC(t, ...) FlatBufferPiece()
#define IPC_TABLE_BEGIN(name) typedef name Parent; void HandleIPC(Connection *c, int fm=0) {}
#define IPC_TABLE_CLIENT_CALL(name)
#define IPC_TABLE_SERVER_CALL(name)
#define IPC_TABLE_CLIENT_QXBC(name, mpv)
#define IPC_TABLE_SERVER_VXBC(name, mpv)
#define IPC_TABLE_CLIENT_QXRC(name, mpt, mpv)
#define IPC_TABLE_SERVER_VXRC(name, mpt, mpv)
#define IPC_TABLE_CLIENT_QIRC(name, mpt, mpv)
#define IPC_TABLE_SERVER_VIRC(name, mpt, mpv)
#define IPC_TABLE_END(name)
#define IPC_CLIENT_CALL(name, mpt, ...) \
  struct name ## Query; \
  struct name ## IPC { \
    typedef function<int(const IPC::name ## Response*, mpt)> CB; \
    name ## IPC(Parent *P=0, IPC::Seq S=0, const CB &C=CB()) {} \
    virtual ~name ## IPC() {} \
  }; \
  unordered_map<IPC::Seq, void*> name ## _map; \
  void name(__VA_ARGS__) {} \
  struct name ## Query : public name ## IPC

#define IPC_SERVER_CALL(name, mpt) \
  struct name ## IPC { name ## IPC(Parent *P, IPC::Seq S=0) {} }; \
  struct name ## Query : public name ## IPC

#define IPC_PROTO_ENTRY(id, name, mpt, ...) struct name { \
    typedef IPC::name Type; \
    static const int Id = id; \
    static const char *Name() { return #name; } \
    static string DebugString(const void *x, const mpt &mpv=mpt()) { return ""; } \
  };
#endif

namespace IPC {
  typedef unsigned short Id;
  typedef unsigned short Seq;
  enum { Error=0, Ok=1, Done=2, Accept=3 };
  struct Header {
    static const int size = 8;
    int len; Id id; Seq seq;
    void Out(Serializable::Stream *o) const { o->Htonl( len); o->Htons( id); o->Htons( seq); }
    void In(const Serializable::Stream *i)  { i->Ntohl(&len); i->Ntohs(&id); i->Ntohs(&seq); }
  };
}; // namespace IPC

struct InterProcessComm {
  struct ConnectionHandler : public Connection::Handler {
    InterProcessComm *parent;
    ConnectionHandler(InterProcessComm *P) : parent(P) {}
    int Read(Connection *c) { parent->HandleIPC(c); return 0; }
  };

  template <class Parent> struct Query {
    Parent *parent;
    IPC::Seq seq;
    int mpb_id;
    Query(Parent *P=0, IPC::Seq S=0, int I=0) : parent(P), seq(S), mpb_id(I) {}
    virtual ~Query() { if (mpb_id) parent->DelBuffer(mpb_id); }
    int Done()  { delete this; return IPC::Done; }
    int Error() { delete this; return IPC::Error; }
  };
  
  template <class Parent> struct ServerQuery : public Query<Parent> {
    ServerQuery(Parent *P=0, IPC::Seq S=0, int I=0) : Query<Parent>(P, S, I) {}
    virtual ~ServerQuery() {}
  };

  template <class Parent, class Res, class MPT> struct ClientQuery : public Query<Parent> {
    typedef function<int(const typename Res::Type*, MPT)> CB;
    CB ipc_cb;
    ClientQuery(Parent *P=0, IPC::Seq S=0, const CB &C=CB()) : Query<Parent>(P, S), ipc_cb(C) {}
    virtual ~ClientQuery() {}

    bool Run(const typename Res::Type *req, MPT mpv) {
      IPCTrace("%s Receive %s Response seq=%d, %s\n",
               this->parent->ipc_name.c_str(), Res::Name(), this->seq, Res::DebugString(req, mpv).c_str());
      CHECK(ipc_cb);
      bool ret = true;
      switch (ipc_cb(req, mpv)) {
        case IPC::Error:
        case IPC::Done:   delete this;
        case IPC::Accept: ret = false;
        default: break;
      }
      return ret;
    } 
  };

  string ipc_name;
  deque<int> ipc_done;
  unordered_map<int, MultiProcessBuffer*> ipc_buffer;
  int pid=0, ipc_buffer_id=0;
  void *server_process=0;
  SocketConnection *conn=0;
  IPC::Seq seq=0;
  bool in_handle_ipc=0;

  InterProcessComm(const string &n) : ipc_name(n) {}
  virtual ~InterProcessComm() {}

  bool StartServerProcess(const string &server_program, const vector<string> &arg=vector<string>());
  bool OpenSocket(const string &socket_name);

  virtual void HandleIPC(Connection *c, int filter_msg=0);
  virtual void HandleIPCMessage(const IPC::Header&, const StringPiece&, int) = 0;

  void HandleMessagesLoop() { while (app->run) if (!HandleMessages()) break; }
  bool HandleMessages(int filter_msg=0) {
    if (!NBReadable(conn->socket, -1)) return true;
    if (conn->Reads() <= 0) return ERRORv(false, conn->Name(), ": read ");
    HandleIPC(conn, filter_msg);
    return true;
  }

  template <class X> bool SendIPCRequest(Connection *c, int seq, const FlatBufferPiece &q, int th=-1) {
    bool ok = Write(c, X::Id, seq, MakeStringPiece(q), th);
    IPCTrace("%s Send %s=%d seq=%d %s\n", ipc_name.c_str(), X::Name(), ok, seq,
             X::DebugString(flatbuffers::GetRoot<typename X::Type>(q.first.get())).c_str());
    return ok;
  }
  static bool Write(Connection *conn, IPC::Id id, IPC::Seq seq, const StringPiece &ipc_text, int transfer_handle=-1);

  MultiProcessBuffer *NewBuffer() const;
  MultiProcessBuffer *AddBuffer();
  bool DelBuffer(IPC::Seq id);
};

}; // namespace LFL
#endif // LFL_CORE_APP_RPC_H__
