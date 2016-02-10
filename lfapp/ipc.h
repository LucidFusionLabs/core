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

#ifndef LFL_LFAPP_IPC_H__
#define LFL_LFAPP_IPC_H__

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

#ifndef LFL_FLATBUFFERS
namespace IPC {
struct Color {
  unsigned char r() const { return 0; }
  unsigned char g() const { return 0; }
  unsigned char b() const { return 0; }
  unsigned char a() const { return 0; }
};
struct FontDescription {
  string *name() const { return 0; }
  string *family() const { return 0; }
  Color *fg() const { return 0; }
  Color *bg() const { return 0; }
  int engine() const { return 0; }
  int size() const { return 0; }
  int flag() const { return 0; }
  bool unicode() const { return 0; }
};
struct ResourceHandle {
  int len() const { return 0; }
  int type() const { return 0; }
  string *url() const { return 0; }
};
struct AllocateBufferRequest {};
struct AllocateBufferResponse {};
struct CloseBufferRequest {};
struct CloseBufferResponse {};
struct LoadAssetRequest { ResourceHandle *mpb() const { return 0; } };
struct LoadAssetResponse {};
struct LoadTextureRequest {};
struct LoadTextureResponse { int tex_id() const { return 0; } };
struct SetClearColorRequest {};
struct SetClearColorResponse {};
struct SetViewportRequest {};
struct SetViewportResponse {};
struct SetDocsizeRequest {};
struct SetDocsizeResponse {};
struct OpenSystemFontRequest {};
struct OpenSystemFontResponse {
  int font_id() const { return 0; }
  int start_glyph_id() const { return 0; }
  int num_glyphs() const { return 0; }
  short ascender() const { return 0; }
  short descender() const { return 0; }
  short max_width() const { return 0; }
  short fixed_width() const { return 0; }
  short missing_glyph() const { return 0; }
  bool mix_fg() const { return 0; }
  bool has_bg() const { return 0; }
  bool fix_metrics() const { return 0; }
  float scale() const { return 0; }
};
struct PaintRequest {};
struct PaintResponse {};
struct SwapTreeRequest {};
struct SwapTreeResponse {};
struct NavigateRequest {};
struct NavigateResponse {};
struct WGetRequest {};
struct WGetResponse {};
struct SetTitleRequest {};
struct SetTitleResponse {};
struct SetURLRequest {};
struct SetURLResponse {};
struct KeyPressRequest {};
struct KeyPressResponse {};
struct MouseClickRequest {};
struct MouseClickResponse {};
struct MouseMoveRequest {};
struct MouseMoveResponse {};
struct ExecuteScriptRequest {};
struct ExecuteScriptResponse { struct String { string str() { return ""; } }; String *text() const { return 0; } };
}; // namespace IPC
#endif // LFL_FLATBUFFERS

struct ProcessAPI : public InterProcessComm {
  ProcessAPI(const string &n) : InterProcessComm(n) {}
  struct Protocol {
    IPC_PROTO_ENTRY( 1, AllocateBufferRequest,  Void,                        "bytes=", x->bytes());
    IPC_PROTO_ENTRY( 2, AllocateBufferResponse, MultiProcessBuffer,          "mpb_id=", x->mpb_id(), ", mpb_len=", mpv.len, ", b=", mpv.buf!=0);
    IPC_PROTO_ENTRY( 3, CloseBufferRequest,     Void,                        "mpb_id=", x->mpb_id());
    IPC_PROTO_ENTRY( 4, CloseBufferResponse,    MultiProcessBuffer,          "success=", x->success());
    IPC_PROTO_ENTRY( 5, OpenSystemFontRequest,  Void,                        FontDesc(x->desc() ? *x->desc() : FontDesc()).DebugString());
    IPC_PROTO_ENTRY( 6, OpenSystemFontResponse, MultiProcessBuffer,          "font_id=", x->font_id(), ", num_glyphs=", x->num_glyphs()); 
    IPC_PROTO_ENTRY( 7, SetClearColorRequest,   Void,                        "c=", x->c() ? Color(x->c()->r(), x->c()->g(), x->c()->b(), x->c()->a()).DebugString() : "" ); 
    IPC_PROTO_ENTRY( 8, SetClearColorResponse,  Void,                        "success=", x->success()); 
    IPC_PROTO_ENTRY( 9, LoadTextureRequest,     MultiProcessTextureResource, "w=", mpv.width, ", h=", mpv.height, ", pf=", BlankNull(Pixel::Name(mpv.pf)));
    IPC_PROTO_ENTRY(10, LoadTextureResponse,    Void,                        "tex_id=", x->tex_id()); 
    IPC_PROTO_ENTRY(11, LoadAssetRequest,       MultiProcessFileResource,    "fn=", BlankNull(mpv.name.buf), ", l=", mpv.buf.len);
    IPC_PROTO_ENTRY(12, LoadAssetResponse,      MultiProcessTextureResource, "w=", mpv.width, ", h=", mpv.height, ", pf=", BlankNull(Pixel::Name(mpv.pf)));
    IPC_PROTO_ENTRY(13, PaintRequest,           MultiProcessPaintResource,   "tile=(", x->x(), ",", x->y(), ",", x->z(), ") flag=", x->flag(), " len=", mpv.data.size());
    IPC_PROTO_ENTRY(14, PaintResponse,          Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(15, SwapTreeRequest,        MultiProcessLayerTree,       "id=(", x->id(), ") node_len=", mpv.node_data.size(), ", child_len=", mpv.child_data.size());
    IPC_PROTO_ENTRY(16, SwapTreeResponse,       Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(17, WGetRequest,            Void,                        "url=", x->url() ? x->url()->data() : ""); 
    IPC_PROTO_ENTRY(18, WGetResponse,           MultiProcessBuffer,          "h=", (int)x->headers(), ", hl=", x->mpb()?x->mpb()->len():0, ", hu=", x->mpb()?x->mpb()->url()->data():"", " b=", mpv.buf!=0, ", l=", mpv.len);
    IPC_PROTO_ENTRY(19, SetTitleRequest,        Void,                        "title=", x->title() ? x->title()->data() : ""); 
    IPC_PROTO_ENTRY(20, SetTitleResponse,       Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(21, SetURLRequest,          Void,                        "url=", x->url() ? x->url()->data() : ""); 
    IPC_PROTO_ENTRY(22, SetURLResponse,         Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(23, NavigateRequest,        Void,                        "url=", x->url() ? x->url()->data() : ""); 
    IPC_PROTO_ENTRY(24, NavigateResponse,       Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(25, SetViewportRequest,     Void,                        "w=", x->w(), ", h=", x->h()); 
    IPC_PROTO_ENTRY(26, SetViewportResponse,    Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(27, SetDocsizeRequest,      Void,                        "w=", x->w(), ", h=", x->h()); 
    IPC_PROTO_ENTRY(28, SetDocsizeResponse,     Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(29, KeyPressRequest,        Void,                        "button=", x->button(), ", down=", x->down()); 
    IPC_PROTO_ENTRY(30, KeyPressResponse,       Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(31, MouseClickRequest,      Void,                        "button=", x->button(), ", down=", x->down(), ", x=", x->x(), ", y=", x->y()); 
    IPC_PROTO_ENTRY(32, MouseClickResponse,     Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(33, MouseMoveRequest,       Void,                        "x=", x->x(), ", y=", x->y(), ", dx=", x->dx(), ", dy=", x->dy()); 
    IPC_PROTO_ENTRY(34, MouseMoveResponse,      Void,                        "success=", x->success());
    IPC_PROTO_ENTRY(35, ExecuteScriptRequest,   Void,                        "text=", x->text() ? x->text()->data() : ""); 
    IPC_PROTO_ENTRY(36, ExecuteScriptResponse,  Void,                        "text=", x->text() ? x->text()->data() : "");
  };
};

struct ProcessAPIClient : public ProcessAPI {
  typedef function<void(const MultiProcessTextureResource&)> TextureCB;
  vector<Drawable*> drawable;
  vector<Font*> font_table;
  Browser *browser=0;
  ProcessAPIClient() : ProcessAPI("ProcessAPIClient") {}

  IPC_TABLE_BEGIN(ProcessAPIClient);
  IPC_TABLE_CLIENT_QIRC(LoadAsset, MultiProcessTextureResource, mpb_id);
  IPC_TABLE_CLIENT_CALL(SetViewport);
  IPC_TABLE_CLIENT_CALL(Navigate);
  IPC_TABLE_CLIENT_CALL(KeyPress);
  IPC_TABLE_CLIENT_CALL(MouseClick);
  IPC_TABLE_CLIENT_CALL(MouseMove);
  IPC_TABLE_CLIENT_CALL(ExecuteScript);
  IPC_TABLE_SERVER_CALL(AllocateBuffer);
  IPC_TABLE_SERVER_CALL(CloseBuffer);
  IPC_TABLE_SERVER_CALL(OpenSystemFont);
  IPC_TABLE_SERVER_CALL(SetClearColor);
  IPC_TABLE_SERVER_CALL(SetDocsize);
  IPC_TABLE_SERVER_VIRC(LoadTexture, MultiProcessTextureResource, mpb_id);
  IPC_TABLE_SERVER_VIRC(Paint, MultiProcessPaintResource, mpb_id);
  IPC_TABLE_SERVER_VIRC(SwapTree, MultiProcessLayerTree, mpb_id);
  IPC_TABLE_SERVER_CALL(WGet);
  IPC_TABLE_SERVER_CALL(SetTitle);
  IPC_TABLE_SERVER_CALL(SetURL);
  IPC_TABLE_END(ProcessAPIClient);

  IPC_CLIENT_CALL(LoadAsset, const MultiProcessTextureResource&, const string&, const string&, const TextureCB&) {
    TextureCB cb;
    LoadAssetQuery(Parent *P, IPC::Seq S, const TextureCB &c) : LoadAssetIPC(P,S,bind(&LoadAssetQuery::Response, this, _1, _2)), cb(c) {}
    int Response(const IPC::LoadAssetResponse*, const MultiProcessTextureResource&);
    void RunCB(const MultiProcessTextureResource&);
  };
  IPC_CLIENT_CALL(Navigate,       Void, const string&) {};
  IPC_CLIENT_CALL(SetViewport,    Void, int w, int h) {};
  IPC_CLIENT_CALL(KeyPress,       Void, int button, bool down) {};
  IPC_CLIENT_CALL(MouseClick,     Void, int button, bool down, int x, int y) {};
  IPC_CLIENT_CALL(MouseMove,      Void, int x, int y, int dx, int dy) {};
  IPC_CLIENT_CALL(ExecuteScript,  Void, const string&, const StringCB&) {
    StringCB cb;
    ExecuteScriptQuery(Parent *P, IPC::Seq S, const StringCB &c) : ExecuteScriptIPC(P,S,bind(&ExecuteScriptQuery::Response, this, _1, _2)), cb(c) {}
    int Response(const IPC::ExecuteScriptResponse *res, Void) { app->RunInMainThread(bind(&ExecuteScriptQuery::RunCB, cb, (res && res->text()) ? res->text()->str() : string())); return IPC::Done; }
    static void RunCB(const StringCB &cb, const string &s) { cb(s); }
  };
  IPC_SERVER_CALL(AllocateBuffer, Void) {};
  IPC_SERVER_CALL(CloseBuffer,    Void) {};
  IPC_SERVER_CALL(OpenSystemFont, Void) {};
  IPC_SERVER_CALL(SetClearColor,  Void) {};
  IPC_SERVER_CALL(SetDocsize,     Void) {};
  IPC_SERVER_CALL(LoadTexture, const MultiProcessTextureResource&) {
    using LoadTextureIPC::LoadTextureIPC;
    void LoadTexture(const MultiProcessTextureResource&);
    void Complete(Texture*);
  };
  IPC_SERVER_CALL(Paint, const MultiProcessPaintResource&) {
    using PaintIPC::PaintIPC;
    void PaintTile(int x, int y, int z, int flag, const MultiProcessPaintResource&);
  };
  IPC_SERVER_CALL(SwapTree, const MultiProcessLayerTree&) {
    using SwapTreeIPC::SwapTreeIPC;
    void SwapLayerTree(int id, const MultiProcessLayerTree&);
  };
  IPC_SERVER_CALL(WGet, Void) {
    using WGetIPC::WGetIPC;
    void WGetRedirectCB(const string &to) { return SendResponse(to.c_str(), 0, 0, true); }
    void WGetResponseCB(Connection *c, const char *h, const string &ct, const char *b, int l) { return SendResponse(h,b,l); }
    void SendResponse(const char *h, const char *b, int l, bool redir=false);
  };
  IPC_SERVER_CALL(SetTitle, Void) {};
  IPC_SERVER_CALL(SetURL, Void) {};
};

struct ProcessAPIServer : public ProcessAPI {
  Browser *browser=0;
  ProcessAPIServer() : ProcessAPI("ProcessAPIServer") {}

  IPC_TABLE_BEGIN(ProcessAPIServer);
  IPC_TABLE_CLIENT_QXBC(AllocateBuffer, mpb);
  IPC_TABLE_CLIENT_CALL(CloseBuffer);
  IPC_TABLE_CLIENT_QXBC(OpenSystemFont, mpb);
  IPC_TABLE_CLIENT_CALL(SetClearColor);
  IPC_TABLE_CLIENT_CALL(SetDocsize);
  IPC_TABLE_CLIENT_CALL(LoadTexture);
  IPC_TABLE_CLIENT_CALL(Paint);
  IPC_TABLE_CLIENT_QXBC(WGet, mpb);
  IPC_TABLE_CLIENT_CALL(SetTitle);
  IPC_TABLE_CLIENT_CALL(SetURL);
  IPC_TABLE_SERVER_VXRC(LoadAsset, MultiProcessFileResource, mpb);
  IPC_TABLE_SERVER_CALL(Navigate);
  IPC_TABLE_SERVER_CALL(SetViewport);
  IPC_TABLE_SERVER_CALL(KeyPress);
  IPC_TABLE_SERVER_CALL(MouseClick);
  IPC_TABLE_SERVER_CALL(MouseMove);
  IPC_TABLE_SERVER_CALL(ExecuteScript);
  IPC_TABLE_END(ProcessAPIServer);

  IPC_CLIENT_CALL(CloseBuffer, Void, int) {};
  IPC_CLIENT_CALL(AllocateBuffer, MultiProcessBuffer&, int, int) { using AllocateBufferIPC::AllocateBufferIPC; };
  IPC_CLIENT_CALL(OpenSystemFont, const MultiProcessBuffer&, const FontDesc &fd, const OpenSystemFontIPC::CB &) { using OpenSystemFontIPC::OpenSystemFontIPC; };
  IPC_CLIENT_CALL(SetClearColor, Void, const Color &c) {};
  IPC_CLIENT_CALL(SetDocsize, Void, int w, int h) {};
  IPC_CLIENT_CALL(LoadTexture, Void, Texture*, const LoadTextureIPC::CB &cb) {
    Texture *tex;
    LoadTextureQuery(Parent *P, Texture *T, const LoadTextureIPC::CB &cb) : LoadTextureIPC(P,0,cb), tex(T) {}
    int AllocateBufferResponse(const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };
  IPC_CLIENT_CALL(Paint, Void, int layer, const point &tile, int flag, MultiProcessPaintResourceBuilder &list) {
    int layer, flag;
    point tile;
    MultiProcessPaintResourceBuilder paint_list;
    PaintQuery(Parent *P, int L, const point &X, int F, MultiProcessPaintResourceBuilder &list) :
      PaintIPC(P,0), layer(L), tile(X), flag(F) { swap(paint_list, list); }
    int AllocateBufferResponse(const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };
  IPC_CLIENT_CALL(SwapTree, Void, int id, const LayersInterface *layers) {
    int id;
    MultiProcessLayerTree tree;
    SwapTreeQuery(Parent *P, int I, const LayersInterface *L) : SwapTreeIPC(P,0), id(I), tree(L->node, L->child) {}
    int AllocateBufferResponse(const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };
  IPC_CLIENT_CALL(WGet, const MultiProcessBuffer&, const string&, const HTTPClient::ResponseCB&, const StringCB&) {
    HTTPClient::ResponseCB cb;
    StringCB redirect_cb;
    WGetQuery(Parent *P, IPC::Seq S, const WGetIPC::CB &C, const HTTPClient::ResponseCB &R, const StringCB &D) : WGetIPC(P,S,C), cb(R), redirect_cb(D) {}
    int WGetResponse(const IPC::WGetResponse*, const MultiProcessBuffer&);
  };
  IPC_CLIENT_CALL(SetTitle, Void, const string &) {};
  IPC_CLIENT_CALL(SetURL,   Void, const string &) {};
  IPC_SERVER_CALL(LoadAsset, const MultiProcessFileResource&) {
    unique_ptr<Texture> tex; 
    LoadAssetQuery(Parent *P, IPC::Seq S, unique_ptr<Texture> T) : LoadAssetIPC(P,S), tex(move(T)) {}
    int AllocateBufferResponse(const IPC::AllocateBufferResponse*, MultiProcessBuffer&);
  };
  IPC_SERVER_CALL(Navigate,      Void) {};
  IPC_SERVER_CALL(SetViewport,   Void) {};
  IPC_SERVER_CALL(KeyPress,      Void) {};
  IPC_SERVER_CALL(MouseClick,    Void) {};
  IPC_SERVER_CALL(MouseMove,     Void) {};
  IPC_SERVER_CALL(ExecuteScript, Void) {};

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
struct TilesIPCClient : public TilesIPC { using TilesIPC::TilesIPC; void Run(int flag); };

typedef LayersT<TilesIPCServer> LayersIPCServer;
typedef LayersT<TilesIPCClient> LayersIPCClient;

}; // namespace LFL
#endif // LFL_LFAPP_IPC_H__
