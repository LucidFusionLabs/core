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

#include "rpc.h"

namespace LFL {
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
  MultiProcessBuffer(const IPC::ResourceHandle *h, int socket);
  virtual ~MultiProcessBuffer();
  virtual void Close();
  virtual bool Open();
  bool Create(int s) { len=s; return Open(); }
  bool Create(const Serializable &s) { bool ret; if ((ret = Create(Size(s)))) s.ToString(buf, len, 0); return ret; }
  bool Copy(const Serializable &s) { bool ret; if ((ret = (len >= Size(s)))) s.ToString(buf, len, 0); return ret; }
  static int Size(const Serializable &s) { return Serializable::Header::size + s.Size(); }
};

struct ProcessAPI : public InterProcessComm {
  ProcessAPI(const string &n) : InterProcessComm(n) {}
  struct Protocol {
    IPC_PROTO_ENTRY( 1, AllocateBufferRequest,  Void,                        "bytes=", x->bytes());
    IPC_PROTO_ENTRY( 2, AllocateBufferResponse, MultiProcessBuffer,          "mpb_id=", x->mpb_id(), ", mpb_len=", mpv.len, ", b=", mpv.buf!=0);
    IPC_PROTO_ENTRY( 3, OpenSystemFontRequest,  Void,                        FontDesc(x->desc() ? *x->desc() : FontDesc()).DebugString());
    IPC_PROTO_ENTRY( 4, OpenSystemFontResponse, MultiProcessBuffer,          "font_id=", x->font_id(), ", num_glyphs=", x->num_glyphs()); 
    IPC_PROTO_ENTRY( 5, SetClearColorRequest,   Void,                        "c=", x->c() ? Color(x->c()->r(), x->c()->g(), x->c()->b(), x->c()->a()).DebugString() : "" ); 
    IPC_PROTO_ENTRY( 6, SetClearColorResponse,  Void,                        "success=", x->success()); 
    IPC_PROTO_ENTRY( 7, LoadTextureRequest,     MultiProcessTextureResource, "w=", mpv.width, ", h=", mpv.height, ", pf=", BlankNull(Pixel::Name(mpv.pf)));
    IPC_PROTO_ENTRY( 8, LoadTextureResponse,    Void,                        "tex_id=", x->tex_id()); 
    IPC_PROTO_ENTRY( 9, LoadAssetRequest,       MultiProcessFileResource,    "fn=", BlankNull(mpv.name.buf), ", l=", mpv.buf.len);
    IPC_PROTO_ENTRY(10, LoadAssetResponse,      MultiProcessTextureResource, "w=", mpv.width, ", h=", mpv.height, ", pf=", BlankNull(Pixel::Name(mpv.pf)));
    IPC_PROTO_ENTRY(11, PaintRequest,           MultiProcessPaintResource,   "tile=(", x->x(), ",", x->y(), ",", x->z(), ") len=", mpv.data.size());
    IPC_PROTO_ENTRY(12, PaintResponse,          Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(13, WGetRequest,            Void,                        "url=", x->url() ? x->url()->data() : ""); 
    IPC_PROTO_ENTRY(14, WGetResponse,           MultiProcessBuffer,          "h=", (int)x->headers(), ", hl=", x->mpb()?x->mpb()->len():0, ", hu=", x->mpb()?x->mpb()->url()->data():"", " b=", mpv.buf!=0, ", l=", mpv.len);
    IPC_PROTO_ENTRY(15, NavigateRequest,        Void,                        "url=", x->url() ? x->url()->data() : ""); 
    IPC_PROTO_ENTRY(16, NavigateResponse,       Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(17, SetViewportRequest,     Void,                        "w=", x->w(), ", h=", x->h()); 
    IPC_PROTO_ENTRY(18, SetViewportResponse,    Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(19, SetDocsizeRequest,      Void,                        "w=", x->w(), ", h=", x->h()); 
    IPC_PROTO_ENTRY(20, SetDocsizeResponse,     Void,                        "success=", x->success());
  };
};

struct ProcessAPIClient : public ProcessAPI {
  int ipc_buffer_id=0;
  unordered_map<int, MultiProcessBuffer*> ipc_buffer;
  vector<Drawable*> drawable;
  vector<Font*> font_table;
  Browser *browser=0;
  ProcessAPIClient() : ProcessAPI("ProcessAPIClient") {}

  IPC_TABLE_BEGIN(ProcessAPIClient);
  IPC_TABLE_CLIENT_CALL(Navigate);
  IPC_TABLE_CLIENT_CALL(SetViewport);
  IPC_TABLE_CLIENT_QIRC(LoadAsset, MultiProcessTextureResource, mpb_id);
  IPC_TABLE_SERVER_CALL(AllocateBuffer);
  IPC_TABLE_SERVER_CALL(OpenSystemFont);
  IPC_TABLE_SERVER_CALL(SetClearColor);
  IPC_TABLE_SERVER_CALL(SetDocsize);
  IPC_TABLE_SERVER_VIRC(LoadTexture, MultiProcessTextureResource, mpb_id);
  IPC_TABLE_SERVER_VIRC(Paint, MultiProcessPaintResource, mpb_id);
  IPC_TABLE_SERVER_CALL(WGet);
  IPC_TABLE_END(ProcessAPIClient);

  IPC_CLIENT_CALL(Navigate, Void, const string&) {};
  IPC_CLIENT_CALL(SetViewport, Void, int w, int h) {};
  IPC_CLIENT_CALL(LoadAsset, const MultiProcessTextureResource&, const string&, const string&, const LoadAssetIPC::CB&) { using LoadAssetIPC::LoadAssetIPC; };
  IPC_SERVER_CALL(AllocateBuffer, Void) {};
  IPC_SERVER_CALL(OpenSystemFont, Void) {};
  IPC_SERVER_CALL(SetClearColor,  Void) {};
  IPC_SERVER_CALL(SetDocsize,     Void) {};
  IPC_SERVER_CALL(LoadTexture, const MultiProcessTextureResource&) {
    using LoadTextureIPC::LoadTextureIPC;
    void LoadTexture(const MultiProcessTextureResource&);
    void SendResponse(Texture*);
  };
  IPC_SERVER_CALL(Paint, const MultiProcessPaintResource&) {};
  IPC_SERVER_CALL(WGet, Void) {
    using WGetIPC::WGetIPC;
    void WGetResponseCB(Connection *c, const char *h, const string &ct, const char *b, int l);
  };
};

struct ProcessAPIServer : public ProcessAPI {
  Browser *browser=0;
  ProcessAPIServer() : ProcessAPI("ProcessAPIServer") {}

  IPC_TABLE_BEGIN(ProcessAPIServer);
  IPC_TABLE_CLIENT_QXBC(AllocateBuffer, mpb);
  IPC_TABLE_CLIENT_QXBC(OpenSystemFont, mpb);
  IPC_TABLE_CLIENT_CALL(SetClearColor);
  IPC_TABLE_CLIENT_CALL(SetDocsize);
  IPC_TABLE_CLIENT_CALL(LoadTexture);
  IPC_TABLE_CLIENT_CALL(Paint);
  IPC_TABLE_CLIENT_QXBC(WGet, mpb);
  IPC_TABLE_SERVER_CALL(Navigate);
  IPC_TABLE_SERVER_CALL(SetViewport);
  IPC_TABLE_SERVER_VXRC(LoadAsset, MultiProcessFileResource, mpb);
  IPC_TABLE_END(ProcessAPIServer);

  IPC_CLIENT_CALL(AllocateBuffer, MultiProcessBuffer&, int, int) { using AllocateBufferIPC::AllocateBufferIPC; };
  IPC_CLIENT_CALL(OpenSystemFont, const MultiProcessBuffer&, const FontDesc &fd, const OpenSystemFontIPC::CB &) { using OpenSystemFontIPC::OpenSystemFontIPC; };
  IPC_CLIENT_CALL(SetClearColor, Void, const Color &c) {};
  IPC_CLIENT_CALL(SetDocsize, Void, int w, int h) {};
  IPC_CLIENT_CALL(LoadTexture, Void, Texture*, const LoadTextureIPC::CB &cb) {
    Texture *tex;
    LoadTextureQuery(Parent *P, Texture *T, const LoadTextureIPC::CB &cb) : LoadTextureIPC(P,0,cb), tex(T) {}
    int AllocateBufferResponse(const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };
  IPC_CLIENT_CALL(Paint, Void, int layer, const point &tile, MultiProcessPaintResourceBuilder &list) {
    int layer;
    point tile;
    MultiProcessPaintResourceBuilder paint_list;
    PaintQuery(Parent *P, int L, const point &X, MultiProcessPaintResourceBuilder &list) :
      PaintIPC(P,0), layer(L), tile(X) { swap(paint_list, list); }
    int AllocateBufferResponse(const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };
  IPC_CLIENT_CALL(WGet, const MultiProcessBuffer&, const string&, const HTTPClient::ResponseCB &) {
    HTTPClient::ResponseCB cb; 
    WGetQuery(Parent *P, IPC::Seq S, const WGetIPC::CB &C, const HTTPClient::ResponseCB &R) : WGetIPC(P,S,C), cb(R) {}
    int WGetResponse(const IPC::WGetResponse*, const MultiProcessBuffer&);
  };
  IPC_SERVER_CALL(Navigate, Void) {};
  IPC_SERVER_CALL(SetViewport, Void) {};
  IPC_SERVER_CALL(LoadAsset, const MultiProcessFileResource&) {
    Texture *tex; 
    LoadAssetQuery(Parent *P, IPC::Seq S, Texture *T) : LoadAssetIPC(P,S), tex(T) {}
    int AllocateBufferResponse(const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };

  void WaitAllOpenSystemFontResponse() {
    while (OpenSystemFont_map.size()) HandleMessages(Protocol::OpenSystemFontResponse::Id);
  }
};

struct TilesIPC : public TilesT<MultiProcessPaintResource::Cmd, MultiProcessPaintResourceBuilder, MultiProcessPaintResource> {
  const Drawable::Attr *attr=0;
  TilesIPC(int l, int w=256, int h=256) : TilesT(l, w, h) {}
  void SetAttr           (const Drawable::Attr*);
  void InitDrawBox       (const point&);
  void InitDrawBackground(const point&);
  void DrawBox           (const Drawable*, const Box&, const Drawable::Attr *a=0);
  void DrawBackground    (const Box&);
  void AddScissor        (const Box&);
};

struct TilesIPCServer : public TilesIPC { using TilesIPC::TilesIPC; };
struct TilesIPCClient : public TilesIPC { using TilesIPC::TilesIPC; void Run(); };

typedef LayersT<TilesIPCServer> LayersIPCServer;
typedef LayersT<TilesIPCClient> LayersIPCClient;

}; // namespace LFL
#endif // __LFL_LFAPP_IPC_H__
