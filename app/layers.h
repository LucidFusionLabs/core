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

#ifndef LFL_CORE_APP_LAYERS_H__
#define LFL_CORE_APP_LAYERS_H__
namespace LFL {
  
template <class Line> struct RingFrameBuffer {
  typedef function<point(Line*, point, const Box&)> PaintCB;
  FrameBuffer fb;
  v2 scroll;
  point p;
  int w=0, h=0, font_size=0, font_height=0;
  RingFrameBuffer(GraphicsDevice *d) : fb(d) {}
  virtual ~RingFrameBuffer() {}

  void ResetGL() { w=h=0; fb.ResetGL(); }
  virtual void SizeChangedDone() { fb.gd->PopScissorStack(); fb.Release(); scroll=v2(); p=point(); }
  virtual int SizeChanged(int W, int H, Font *font, const Color *bgc) {
    if (W == w && H == h && font->size == font_size) return 0;
    int orig_font_size = font_size;
    SetDimensions(W, H, font);
    fb.Resize(w, h, FrameBuffer::Flag::CreateGL | FrameBuffer::Flag::CreateTexture);
    ScopedClearColor scc(fb.gd, bgc);
    fb.gd->PushScissorStack();
    fb.gd->Clear();
    fb.gd->DrawMode(DrawMode::_2D, false);
    return 1 + (orig_font_size && font_size != orig_font_size);
  }

  virtual void Draw(point pos, point adjust, bool scissor=true) {
    Box box(pos.x, pos.y, w, h);
    if (scissor) fb.gd->PushScissor(box);
    fb.tex.Bind();
    (box + adjust).DrawCrimped(fb.tex.coord, 0, 0, scroll.y);
    if (scissor) fb.gd->PopScissor();
  }

  void Clear(Line *l, const Box &b, bool vwrap=true) {
    if (1)                         { Scissor s(fb.gd, 0, l->p.y - b.h, b.w, b.h); fb.gd->Clear(); }
    if (l->p.y - b.h < 0 && vwrap) { Scissor s(fb.gd, 0, l->p.y + h,   b.w, b.h); fb.gd->Clear(); }
  }

  void Update(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    point lp = paint(l, l->p, b);
    if (lp.y < 0 && vwrap) paint(l, point(0, lp.y + h + b.h), b);
  }

  int PushFrontAndUpdate(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    if (b.h >= h)     p = paint(l,      point(0, h),          b);
    else                  paint(l, (p = point(0, p.y + b.h)), b);
    if (p.y > h && vwrap) paint(l, (p = point(0, p.y - h)),   b);
    ScrollPercent(float(-b.h) / h);
    return b.h;
  }

  int PushBackAndUpdate(Line *l, const Box &b, const PaintCB &paint, bool vwrap=true) {
    if (p.y == 0)         p =          point(0, h);
    if (b.h >= h)         p = paint(l, point(0, b.h),           b);
    else                  p = paint(l, point(0, p.y),           b);
    if (p.y < 0 && vwrap) p = paint(l, point(0, p.y + b.h + h), b);
    ScrollPercent(float(b.h) / h);
    return b.h;
  }

  void SetDimensions(int W, int H, Font *f) { w = W; h = H; font_size = f->size; font_height = f->Height(); }
  void ScrollPercent(float y) { scroll.y = fmod(scroll.y + y, 1.0); }
  void ScrollPixels(int y) { ScrollPercent(float(y) / h); }
  void AdvancePixels(int y) { ScrollPixels(y); p.y = RingIndex::WrapOver(p.y - y, h); }
  point BackPlus(const point &o, bool wrap_zero=false) const {
    int y = RingIndex::WrapOver(p.y + o.y, h);
    return point(RingIndex::WrapOver(p.x + o.x, w), (wrap_zero && !y) ? h : y);
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
    fb.Attach(tile->id, false);
    fb.gd->MatrixProjection();
    if (!(flag & RunFlag::DontClear)) fb.gd->Clear();
    fb.gd->LoadIdentity();
    fb.gd->Ortho(current_tile.x, current_tile.x + W, current_tile.y, current_tile.y + H, 0, 100);
    fb.gd->MatrixModelview();
    tile_cb.Run(current_tile);
  }

  void Release() { fb.Release(); }

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
#endif // LFL_CORE_APP_LAYERS_H__
