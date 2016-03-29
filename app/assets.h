/*
 * $Id: assets.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef LFL_CORE_APP_ASSETS_H__
#define LFL_CORE_APP_ASSETS_H__
namespace LFL {

DECLARE_int(soundasset_seconds);
DECLARE_float(shadertoy_blend);

struct Geometry {
  int vd, td, cd, primtype, count, material, color;
  Material mat;
  Color col;
  vector<float> vert, last_position;
  int width, vert_ind, norm_offset, tex_offset, color_offset;
  Geometry(int N=0, int V=0, int PT=0, int VD=0, int TD=0, int CD=0) : vd(VD), td(TD), cd(CD), primtype(PT), count(N),
  material(0), color(0), vert(V), last_position(VD), width(VD), vert_ind(-1), norm_offset(-1), tex_offset(-1), color_offset(-1) {}

  static const int TD=2, CD=4;
  template <class X> Geometry(int VD, int primtype, int num, X *v, v3 *norm, v2 *tex, const Color *vcol) :
    Geometry(num, num*VD*(1+(norm!=0)) + num*TD*(tex!=0) + num*CD*(vcol!=0) + 256, primtype, VD, TD, CD)
  {
    if (norm) { norm_offset  = width*sizeof(float); width += VD; }
    if (tex)  { tex_offset   = width*sizeof(float); width += TD; }
    if (vcol) { color_offset = width*sizeof(float); width += CD; }
    for (int i = 0; i<count; i++) {
      int k = 0;
      for (int j = 0; j<VD && v;    j++, k++) vert[i*width + k] = v[i][j];
      for (int j = 0; j<VD && norm; j++, k++) vert[i*width + k] = norm[i][j];
      for (int j = 0; j<TD && tex;  j++, k++) vert[i*width + k] = tex[i][j];
      for (int j = 0; j<CD && vcol; j++, k++) vert[i*width + k] = vcol[i].x[j];
    }
  }
  template <class X> Geometry(int VD, int primtype, int num, X *v, v3 *norm, v2 *tex, const Color &vcol) :
    Geometry(VD, primtype, num, v, norm, tex, nullptr) { color=1; col=vcol; }

  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex            ) : Geometry(2, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex            ) : Geometry(3, pt, n, v, norm, tex, 0  ) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color  col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color  col) : Geometry(3, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v2 *v, v3 *norm, v2 *tex, Color *col) : Geometry(2, pt, n, v, norm, tex, col) {}
  Geometry (int pt, int n, v3 *v, v3 *norm, v2 *tex, Color *col) : Geometry(3, pt, n, v, norm, tex, col) {}

  void SetPosition(const float *v);
  void SetPosition(const point &p) { v2 v=p; SetPosition(&v[0]); }
  void ScrollTexCoord(float dx, float dx_extra, int *subtract_max_int);

  static unique_ptr<Geometry> LoadOBJ(const string &filename, const float *map_tex_coord=0);
  static string ExportOBJ(const Geometry *geometry, const set<int> *prim_filter=0, bool prim_filter_invert=0);
};

template <class X> struct AssetMapT {
  bool loaded=0;
  vector<X> vec;
  unordered_map<string, X*> amap;
  template<typename ...Args>
  void Add(Args&& ...args) { CHECK(!loaded); vec.emplace_back(forward<Args>(args)...); }
  void Unloaded(X *a) { if (!a->name.empty()) amap.erase(a->name); }
  void Load(X *a) { a->parent = this; if (!a->name.empty()) amap[a->name] = a; a->Load(); }
  void Load() { CHECK(!loaded); for (int i=0; i<vec.size(); i++) Load(&vec[i]); loaded=1; }
  X *operator()(const string &an) { return FindOrNull(amap, an); }
};
typedef AssetMapT<     Asset>      AssetMap;
typedef AssetMapT<SoundAsset> SoundAssetMap;
typedef AssetMapT<MovieAsset> MovieAssetMap;

struct Asset {
  typedef function<void(Asset*, Entity*)> DrawCB;

  AssetMap *parent=0;
  string name, texture, geom_fn;
  DrawCB cb;
  float scale=0;
  int translate=0, rotate=0;
  Geometry *geometry=0, *hull=0;
  Texture tex;
  unsigned texgen=0, typeID=0, particleTexID=0, blends=GraphicsDevice::SrcAlpha, blendt=GraphicsDevice::OneMinusSrcAlpha;
  Color col;
  bool color=0, zsort=0;

  Asset() {}
  Asset(const string &N, const string &Tex, float S, int T, int R, const char *G, Geometry *H, unsigned CM, const DrawCB &CB=DrawCB())
    : name(N), texture(Tex), geom_fn(BlankNull(G)), cb(CB), scale(S), translate(T), rotate(R), hull(H) { tex.cubemap=CM; }
  Asset(const string &N, const string &Tex, float S, int T, int R, Geometry *G, Geometry *H, unsigned CM, unsigned TG, const DrawCB &CB=DrawCB())
    : name(N), texture(Tex), cb(CB), scale(S), translate(T), rotate(R), geometry(G), hull(H), texgen(TG) { tex.cubemap=CM; }

  void Load(void *handle=0, VideoAssetLoader *l=0);
  void Unload();

  static void Load(vector<Asset> *assets) { for (int i=0; i<assets->size(); ++i) (*assets)[i].Load(); }
  static void LoadTexture(         const string &asset_fn, Texture *out, VideoAssetLoader *l=0) { LoadTexture(0, asset_fn, out, l); }
  static void LoadTexture(void *h, const string &asset_fn, Texture *out, VideoAssetLoader *l=0);
  static void LoadTexture(const void *from_buf, const char *fn, int size, Texture *out, int flag=VideoAssetLoader::Flag::Default);
  static Texture *LoadTexture(const MultiProcessFileResource &file, int max_image_size = 1000000);
  static string FileName(const string &asset_fn);
  static string FileContents(const string &asset_fn);
  static File *OpenFile(const string &asset_fn);
  static void Copy(const Asset *in, Asset *out) {
    string name = out->name;
    unsigned typeID = out->typeID;
    *out = *in;
    out->name = name;
    out->typeID = typeID;
  }
};

struct SoundAsset {
  static const int FlagNoRefill, FromBufPad;
  typedef function<int(SoundAsset*, int)> RefillCB;

  SoundAssetMap *parent;
  string name, filename;
  unique_ptr<RingSampler> wav;
  int channels=FLAGS_chans_out, sample_rate=FLAGS_sample_rate, seconds=FLAGS_soundasset_seconds;
  RefillCB refill;
  void *handle=0;
  int handle_arg1=-1;
  unique_ptr<AudioResamplerInterface> resampler;

  SoundAsset() {}
  SoundAsset(const string &N, const string &FN, RingSampler *W, int C, int SR, int S) :
    name(N), filename(FN), wav(W), channels(C), sample_rate(SR), seconds(S) {}

  void Load(void *handle, const char *FN, int Secs, int flag=0);
  void Load(const void *FromBuf, int size, const char *FileName, int Seconds=FLAGS_soundasset_seconds);
  void Load(int seconds=FLAGS_soundasset_seconds, bool unload=true);
  void Unload();
  int Refill(int reset);

  static void Load(vector<SoundAsset> *assets) { for (auto &a : *assets) a.Load(); }
  static size_t Size(const SoundAsset *sa) { return sa->seconds * sa->sample_rate * sa->channels; }
};

struct MovieAsset {
  MovieAssetMap *parent;
  string name, filename;
  SoundAsset audio;
  Asset video;
  void *handle;

  MovieAsset() : parent(0), handle(0) {}
  void Load(const char *fn=0);
  int Play(int seek);
};

struct MapAsset {
  virtual void Draw(const Entity &camera) = 0;
};

void glLine(const point &p1, const point &p2, const Color *color);
void glAxis(Asset*, Entity*);
void glRoom(Asset*, Entity*);
void glIntersect(int x, int y, Color *c);
void glShadertoyShader(Shader *shader, const Texture *tex=0);
void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const Box                &win, const Texture *tex=0);
void glShadertoyShaderWindows(Shader *shader, const Color &backup_color, const vector<const Box*> &win, const Texture *tex=0);
void glSpectogram(Matrix *m, unsigned char *data, int pf, int width, int height, int hjump, float max, float clip, bool interpolate, int pd=PowerDomain::dB);
void glSpectogram(Matrix *m, Texture *t, float *max=0, float clip=-INFINITY, int pd=PowerDomain::dB);
void glSpectogram(const RingSampler::Handle *in, Texture *t, Matrix *transform=0, float *max=0, float clip=-INFINITY);

struct BoxFilled             : public Drawable { void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const; };
struct BoxOutline            : public Drawable { void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const; };
struct BoxTopLeftOutline     : public Drawable { void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const; };
struct BoxBottomRightOutline : public Drawable { void Draw(const LFL::Box &b, const Drawable::Attr *a=0) const; };

struct Waveform : public Drawable {
  int width=0, height=0;
  unique_ptr<Geometry> geom;
  Waveform() {}
  Waveform(point dim, const Color *c, const Vec<float> *);
  void Draw(const LFL::Box &w, const Drawable::Attr *a=0) const;
  static Waveform Decimated(point dim, const Color *c, const RingSampler::Handle *, int decimateBy);
};

struct Cube {
  static unique_ptr<Geometry> Create(v3 v);
  static unique_ptr<Geometry> Create(float rx, float ry, float rz, bool normals=false);
  static unique_ptr<Geometry> CreateFrontFace(float r);
  static unique_ptr<Geometry> CreateBackFace(float r);
  static unique_ptr<Geometry> CreateLeftFace(float r);
  static unique_ptr<Geometry> CreateRightFace(float r);
  static unique_ptr<Geometry> CreateTopFace(float r);
  static unique_ptr<Geometry> CreateBottomFace(float r);
};

struct Grid {
  static unique_ptr<Geometry> Grid3D();
  static unique_ptr<Geometry> Grid2D(float x, float y, float range, float step);
};

struct Skybox {
  Asset               a_left, a_right, a_top, a_bottom, a_front, a_back;
  Entity              e_left, e_right, e_top, e_bottom, e_front, e_back;
  Scene::EntityVector v_left, v_right, v_top, v_bottom, v_front, v_back;
  Skybox();

  Asset *asset() { return &a_left; }
  void Load(const string &filename_prefix);
  void Draw();
};

template <class Line> struct RingFrameBuffer {
  typedef function<point(Line*, point, const Box&)> PaintCB;
  FrameBuffer fb;
  v2 scroll;
  point p;
  bool wrap=0;
  int w=0, h=0, font_size=0, font_height=0;
  RingFrameBuffer(GraphicsDevice *d) : fb(d) {}

  void ResetGL() { w=h=0; fb.ResetGL(); }
  virtual int Width()  const { return w; }
  virtual int Height() const { return h; }
  virtual void SizeChangedDone() { fb.Release(); scroll=v2(); p=point(); }
  virtual int SizeChanged(int W, int H, Font *font, const Color *bgc) {
    if (W == w && H == h && font->size == font_size) return 0;
    int orig_font_size = font_size;
    SetDimensions(W, H, font);
    fb.Resize(w, Height(), FrameBuffer::Flag::CreateGL | FrameBuffer::Flag::CreateTexture);
    ScopedClearColor scc(fb.gd, bgc);
    fb.gd->Clear();
    fb.gd->DrawMode(DrawMode::_2D, false);
    return 1 + (orig_font_size && font_size != orig_font_size);
  }

  virtual void Draw(point pos, point adjust, bool scissor=true) {
    Box box(pos.x, pos.y, w, Height());
    if (scissor) fb.gd->PushScissor(box);
    fb.tex.Bind();
    (box + adjust).DrawCrimped(fb.tex.coord, 0, 0, scroll.y);
    if (scissor) fb.gd->PopScissor();
  }

  void Clear(Line *l, const Box &b, bool vwrap=true) {
    if (1)                         { Scissor s(fb.gd, 0, l->p.y - b.h,      b.w, b.h); fb.gd->Clear(); }
    if (l->p.y - b.h < 0 && vwrap) { Scissor s(fb.gd, 0, l->p.y + Height(), b.w, b.h); fb.gd->Clear(); }
  }

  void Update(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    point lp = paint(l, l->p, b);
    if (lp.y < 0 && vwrap) paint(l, point(0, lp.y + Height() + b.h), b);
  }

  int PushFrontAndUpdate(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    int ht = Height();
    if (b.h >= ht)     p = paint(l,      point(0, ht),         b);
    else                   paint(l, (p = point(0, p.y + b.h)), b);
    if (p.y > ht && vwrap) paint(l, (p = point(0, p.y - ht)),  b);
    ScrollPercent(float(-b.h) / ht);
    return b.h;
  }

  int PushBackAndUpdate(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    int ht = Height();
    if (p.y == 0)         p =          point(0, ht);
    if (b.h >= ht)        p = paint(l, point(0, b.h),            b);
    else                  p = paint(l, point(0, p.y),            b);
    if (p.y < 0 && vwrap) p = paint(l, point(0, p.y + b.h + ht), b);
    ScrollPercent(float(b.h) / ht);
    return b.h;
  }

  void SetDimensions(int W, int H, Font *f) { w = W; h = H; font_size = f->size; font_height = f->Height(); }
  void ScrollPercent(float y) { scroll.y = fmod(scroll.y + y, 1.0); }
  void ScrollPixels(int y) { ScrollPercent(float(y) / Height()); }
  void AdvancePixels(int y) { ScrollPixels(y); p.y = RingIndex::WrapOver(p.y - y, Height()); }
  point BackPlus(const point &o, bool wrap_zero=false) const {
    int y = RingIndex::WrapOver(p.y + o.y, Height());
    return point(RingIndex::WrapOver(p.x + o.x, Width()), (wrap_zero && !y) ? Height() : y);
  }
};

struct TilesInterface {
  struct RunFlag { enum { DontClear=1, ClearEmpty=2 }; };
  virtual ~TilesInterface() {}
  virtual void AddDrawableBoxArray(const DrawableBoxArray &box, point p);
  virtual void SetAttr            (const Drawable::Attr *a)                                = 0;
  virtual void InitDrawBox        (const point&)                                           = 0;
  virtual void InitDrawBackground (const point&)                                           = 0;
  virtual void DrawBox            (const Drawable*, const Box&, const Drawable::Attr *a=0) = 0;
  virtual void DrawBackground     (const Box&)                                             = 0;
  virtual void AddScissor         (const Box&)                                             = 0;
  virtual void Draw(const Box &viewport, const point &doc_position)                        = 0;
  virtual void ContextOpen()                                                               = 0;
  virtual void ContextClose()                                                              = 0;
  virtual void Run(int flag)                                                               = 0;
};

struct LayersInterface {
  UNALIGNED_struct Node  { static const int Size=32+sizeof(void*); Box box; point scrolled; int layer_id, child_offset; }; UNALIGNED_END(Node,  Node::Size);
  UNALIGNED_struct Child { static const int Size=8;                int node_id, next_child_offset; };                      UNALIGNED_END(Child, Child::Size);
  vector<Node> node;
  vector<Child> child;
  vector<unique_ptr<TilesInterface>> layer;
  LayersInterface() { ClearLayerNodes(); }

  void ClearLayerNodes() { child.clear(); node.clear(); node.push_back({ Box(), point(), 0, 0 }); }
  int AddLayerNode(int parent_node_id, const Box &b, int layer_id) {
    node.push_back({ b, point(), layer_id, 0 });
    return AddChild(parent_node_id, node.size());
  }

  int AddChild(int node_id, int child_node_id) {
    child.push_back({ child_node_id, 0 }); 
    Node *n = &node[node_id-1];
    if (!n->child_offset) return (n->child_offset = child.size());
    Child *c = &child[n->child_offset-1];
    while (c->next_child_offset) c = &child[c->next_child_offset-1];
    return (c->next_child_offset = child.size());
  }

  virtual void Update();
  virtual void Draw(const Box &b, const point &p);
  virtual void Init(GraphicsDevice *D, int N=1) = 0;
};

#define TilesMatrixIter(m) MatrixIter(m) if (Tile *tile = static_cast<Tile*>((m)->row(i)[j]))
template<class CB, class CBL, class CBLI> struct TilesT : public TilesInterface {
  struct Tile {
    CBL cb;
    unsigned id=0, prepend_depth=0;
    bool dirty=0;
  };
  int layer, W, H, context_depth=-1;
  vector<Tile*> prepend, append;
  matrix<Tile*> mat;
  FrameBuffer fb;
  Box current_tile;
  TilesT(GraphicsDevice *d, int l, int w=256, int h=256) : layer(l), W(w), H(h), mat(1,1), fb(d) { CHECK(IsPowerOfTwo(W)); CHECK(IsPowerOfTwo(H)); }
  ~TilesT() { TilesMatrixIter(&mat) delete tile; for (auto t : prepend) delete t; for (auto t : append) delete t; }
  
  template <class... Args> void PreAdd(Args&&... args) { prepend[context_depth]->cb.Add(forward<Args>(args)...); }
  template <class... Args> void PostAdd(Args&&... args) { append[context_depth]->cb.Add(forward<Args>(args)...); }
  template <class... Args> void AddCallback(const Box *box, Args&&... args) {
    bool added = 0;
    int x1, x2, y1, y2;
    GetTileCoords(*box, &x1, &y1, &x2, &y2);
    for (int y = max(y1, 0); y <= y2; y++)
      for (int x = max(x1, 0); x <= x2; x++, added=1) GetTile(x, y)->cb.Add(args...);
    if (!added) ERROR("AddCallback zero ", box->DebugString(), " = ", x1, " ", y1, " ", x2, " ", y2);
  }

  void PushScissor(const Box &w) const { fb.gd->PushScissorOffset(current_tile, w); }
  void GetSpaceCoords(int i, int j, int *xo, int *yo) const { *xo =  j * W; *yo = (-i-1) * H; }
  void GetTileCoords (int x, int y, int *xo, int *yo) const { *xo =  x / W; *yo = -y     / H; }
  void GetTileCoords(const Box &box, int *x1, int *y1, int *x2, int *y2) const {
    GetTileCoords(box.x,         box.y + box.h, x1, y1);
    GetTileCoords(box.x + box.w, box.y,         x2, y2);
  }

  Tile *GetTile(int x, int y) {
    CHECK_GE(x, 0);
    CHECK_GE(y, 0);
    int add;
    if ((add = x - mat.N + 1) > 0) mat.AddCols(add);
    if ((add = y - mat.M + 1) > 0) mat.AddRows(add);
    Tile **ret = reinterpret_cast<Tile**>(&mat.row(y)[x]);
    if (!*ret) *ret = new Tile();
    if (!(*ret)->cb.dirty) {
      for (int i = (*ret)->prepend_depth; i <= context_depth; i++) (*ret)->cb.AddList(prepend[i]->cb);
      (*ret)->prepend_depth = context_depth + 1;
    }
    return *ret;
  }

  void ContextOpen() {
    TilesMatrixIter(&mat) { if (tile->cb.dirty) tile->dirty = 1; tile->cb.dirty = 0; }
    if (++context_depth < prepend.size()) { prepend[context_depth]->cb.Clear(); append[context_depth]->cb.Clear(); }
    else { prepend.push_back(new Tile()); append.push_back(new Tile()); CHECK_LT(context_depth, prepend.size()); }
  }

  void ContextClose() {
    CHECK_GE(context_depth, 0);
    TilesMatrixIter(&mat) {
      if (tile->cb.dirty) tile->dirty = 1;
      if (tile->dirty && tile->prepend_depth > context_depth) {
        tile->cb.AddList(append[context_depth]->cb);
        tile->prepend_depth = context_depth;
      }
      if (!context_depth) tile->dirty = 0;
    }
    context_depth--;
  }

  void Run(int flag) {
    bool clear_empty = (flag & RunFlag::ClearEmpty);
    Select();
    TilesMatrixIter(&mat) {
      if (!tile->cb.Count() && !clear_empty) continue;
      RunTile(i, j, flag, tile, tile->cb);
      tile->cb.Clear();
    }
    Release();
  }

  void Select() {
    bool init = !fb.ID;
    if (init) fb.Create(W, H);
    current_tile = Box(0, 0, W, H);
    fb.gd->DrawMode(DrawMode::_2D);
    fb.gd->ViewPort(current_tile);
    fb.gd->EnableLayering();
  }

  void RunTile(int i, int j, int flag, Tile *tile, const CBLI &tile_cb) {
    GetSpaceCoords(i, j, &current_tile.x, &current_tile.y);
    if (!tile->id) fb.AllocTexture(&tile->id);
    fb.Attach(tile->id);
    fb.gd->MatrixProjection();
    if (!(flag & RunFlag::DontClear)) fb.gd->Clear();
    fb.gd->LoadIdentity();
    fb.gd->Ortho(current_tile.x, current_tile.x + W, current_tile.y, current_tile.y + H, 0, 100);
    fb.gd->MatrixModelview();
    tile_cb.Run(current_tile);
  }

  void Release() {
    fb.Release();
    fb.gd->RestoreViewport(DrawMode::_2D);
  }

  void Draw(const Box &viewport, const point &docp) {
    int x1, x2, y1, y2, sx, sy;
    point doc_to_view = docp - viewport.Position();
    GetTileCoords(Box(docp.x, docp.y, viewport.w, viewport.h), &x1, &y1, &x2, &y2);

    Scissor scissor(fb.gd, viewport);
    fb.gd->DisableBlend();
    fb.gd->SetColor(Color::white);
    for (int y = max(y1, 0); y <= y2; y++) {
      for (int x = max(x1, 0); x <= x2; x++) {
        Tile *tile = GetTile(x, y);
        if (!tile || !tile->id) continue;
        GetSpaceCoords(y, x, &sx, &sy);
        fb.gd->BindTexture(GraphicsDevice::Texture2D, tile->id);
        Box(sx - doc_to_view.x, sy - doc_to_view.y, W, H).Draw(Texture::unit_texcoord);
      }
    }
  }
};

struct Tiles : public TilesT<Callback, CallbackList, CallbackList> {
  const Drawable::Attr *attr=0;
  Tiles(GraphicsDevice *d, int l, int w=256, int h=256) : TilesT(d, l, w, h) {}
  void SetAttr           (const Drawable::Attr *a) { attr=a; }
  void InitDrawBox       (const point&);
  void InitDrawBackground(const point&);
  void DrawBox           (const Drawable*, const Box&, const Drawable::Attr *a=0);
  void DrawBackground    (const Box&);
  void AddScissor        (const Box&);
};

template <class X> struct LayersT : public LayersInterface {
  void Init(GraphicsDevice *D, int N=1) {
    CHECK_EQ(this->layer.size(), 0);
    for (int i=0; i<N; i++) this->layer.emplace_back(make_unique<X>(D, i));
  }
};

typedef LayersT<Tiles> Layers;

}; // namespace LFL
#endif // LFL_CORE_APP_ASSETS_H__
