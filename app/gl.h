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

#ifndef LFL_CORE_APP_VIDEO_H__
#define LFL_CORE_APP_VIDEO_H__

namespace LFL {
DECLARE_int(dots_per_inch);
DECLARE_float(field_of_view);
DECLARE_float(near_plane);
DECLARE_float(far_plane);
DECLARE_bool(swap_axis);
DECLARE_bool(gd_debug);

struct DrawMode { enum { _2D=0, _3D=1, NullOp=2 }; int m; };
struct ResetGLFlag { enum { Forget=0, Delete=1, Reload=2 }; };
struct TexGen { enum { LINEAR=1, REFLECTION=2 }; };
struct Depth { enum { _16=1 };  };
struct CubeMap { enum { PX=1, NX=2, PY=3, NY=4, PZ=5, NZ=6 }; };
struct ColorChannel {
  enum { Red=1, Green=2, Blue=3, Alpha=4 };
  static int PixelOffset(int c);
};

struct Pixel {
  enum { RGB32=1, BGR32=2, RGBA=3, BGRA=4, ARGB=5,
    RGB24=6, BGR24=7, 
    RGB555=8, BGR555=9, RGB565=10, BGR565=11,
    YUV410P=12, YUV420P=13, YUYV422=14, YUVJ420P=15, YUVJ422P=16, YUVJ444P=17,
    ALPHA8=18, GRAY8=19, GRAYA8=20, LCD=21 };
  struct Index { enum { R=0, G=1, B=2, A=3 }; };

  static const char *Name(int id);
  static int Size(int p);
  static int GetNumComponents(int p);
  static int GetRGBAIndex(int p, int i);
};

struct Color {
  float x[4];
  Color() { r()=0; g()=0; b()=0; a()=0; }
  Color(const float *f) { r()=f[0]; g()=f[1]; b()=f[2]; a()=f[3]; }
  Color(double R, double G, double B) { r()=R; g()=G; b()=B; a()=1.0; }
  Color(double R, double G, double B, double A) { r()=R; g()=G; b()=B; a()=A; }
  Color(int R, int G, int B) { r()=R/255.0; g()=G/255.0; b()=B/255.0; a()=1.0; }
  Color(int R, int G, int B, int A) { r()=R/255.0; g()=G/255.0; b()=B/255.0; a()=A/255.0; }
  Color(ColorDesc v, bool has_alpha=true) { r()=((v>>16)&0xff)/255.0; g()=((v>>8)&0xff)/255.0; b()=(v&0xff)/255.0; a()=(has_alpha ? ((v>>24)&0xff)/255.0 : 1.0); };
  Color(const StringPiece &hs) { *this = Color(strtoul(hs.data(), 0, 16), false); }
  Color(const Color &c, double A) { *this = c; a() = A; }
  Color operator+(const Color &y) const { Color ret = *this; for (int i=0;i<4;i++) ret.x[i] += y.x[i]; return ret; }
  Color operator-(const Color &y) const { Color ret = *this; for (int i=0;i<4;i++) ret.x[i] -= y.x[i]; return ret; }
  bool operator< (const Color &y) const { SortImpl4(x[0], y.x[0], x[1], y.x[1], x[2], y.x[2], x[3], y.x[3]); }
  bool operator==(const Color &y) const { return R()==y.R() && G()==y.G() && B()==y.B() && A()==y.A(); }
  bool operator!=(const Color &y) const { return !(*this == y); }
  operator ColorDesc() const { return uint8_t(A())<<24 | uint8_t(R())<<16 | uint8_t(G())<<8 | uint8_t(B()); }
  bool Transparent() const { return a() == 0; }
  string IntString() const { return StrCat("Color(", A(), ",", R(), ",", G(), ",", B(), ")"); }
  string HexString() const { return StringPrintf("%02X%02X%02X", R(), G(), B()); }
  string DebugString() const { return StringPrintf("%02X%02X%02X%02X", A(), R(), G(), B()); }
  const float &r() const { return x[0]; }
  const float &g() const { return x[1]; }
  const float &b() const { return x[2]; }     
  const float &a() const { return x[3]; }
  float       &r()       { return x[0]; }     
  float       &g()       { return x[1]; }
  float       &b()       { return x[2]; }     
  float       &a()       { return x[3]; }
  int          R() const { return RoundF(x[0]*255); } 
  int          G() const { return RoundF(x[1]*255); }
  int          B() const { return RoundF(x[2]*255); } 
  int          A() const { return RoundF(x[3]*255); }
  Color r(float v) const { Color c=*this; c.r() = v; return c; }
  Color g(float v) const { Color c=*this; c.g() = v; return c; }
  Color b(float v) const { Color c=*this; c.b() = v; return c; }
  Color a(float v) const { Color c=*this; c.a() = v; return c; }
  void scale(float f) { r() = Clamp(r()*f, 0.0f, 1.0f); g() = Clamp(g()*f, 0.0f, 1.0f); b() = Clamp(b()*f, 0.0f, 1.0f); }
  void ToHSV(float *h, float *s, float *v) const;
  static Color FromHSV(float h, float s, float v);
  static Color Red  (float v=1.0) { return Color(  v, 0.0, 0.0, 0.0); }
  static Color Green(float v=1.0) { return Color(0.0,   v, 0.0, 0.0); }
  static Color Blue (float v=1.0) { return Color(0.0, 0.0,   v, 0.0); }
  static Color Alpha(float v=1.0) { return Color(0.0, 0.0, 0.0,   v); }
  static Color fade(float v);
  static Color Interpolate(Color l, Color r, float mix) { l.scale(mix); r.scale(1-mix); return add(l,r); }
  static Color add(const Color &l, const Color &r) { return Color(Clamp(l.r()+r.r(), 0.0f, 1.0f), Clamp(l.g()+r.g(), 0.0f, 1.0f), Clamp(l.b()+r.b(), 0.0f, 1.0f), Clamp(l.a()+r.a(), 0.0f, 1.0f)); }
  static Color white, black, red, green, blue, cyan, yellow, magenta, grey90, grey80, grey70, grey60, grey50, grey40, grey30, grey20, grey10, clear;
  static bool IsTransparent(ColorDesc x) { return !(x >> 24); }
};

struct Material {
  Color ambient, diffuse, specular, emissive;
  Material() {}
  void SetLightColor(const Color &color);
  void SetMaterialColor(const Color &color);
};

struct Light { v4 pos; Material color; };

struct Drawable {
  struct Attr { 
    Font *font=0;
    const Color *fg=0, *bg=0;
    const Texture *tex=0;
    const LFL::Box *scissor=0;
    bool underline=0, overline=0, midline=0, blink=0, blend=0;
    int line_width=1;
    constexpr Attr(Font *F=0, const Color *FG=0, const Color *BG=0, bool UL=0, bool B=0, int LW=1) : font(F), fg(FG), bg(BG), underline(UL), blend(B), line_width(LW) {}
    constexpr Attr(const Texture *T, const Color *FG=0, const Color *BG=0, bool UL=0, bool B=0, int LW=1) : fg(FG), bg(BG), tex(T), underline(UL), blend(B), line_width(LW) {}
    bool operator==(const Attr &y) const { return font==y.font && fg==y.fg && bg==y.bg && tex==y.tex && scissor==y.scissor && underline==y.underline && overline==y.overline && midline==y.midline && blink==y.blink && blend == y.blend; }
    bool operator!=(const Attr &y) const { return !(*this == y); }
    void Clear() { font=0; fg=bg=0; tex=0; scissor=0; underline=overline=midline=blink=0; }
    string DebugString() const;
  };
  struct AttrSource { virtual const Attr *GetAttr(int attr_id) const = 0; };
  struct AttrVec : public AttrSource, public vector<Attr> {
    Attr current;
    AttrSource *source=0;
    RefSet font_refs;
    AttrVec() {}
    const Attr *GetAttr(int attr_id) const override { return source ? source->GetAttr(attr_id) : &(*this)[attr_id-1]; }
    int GetAttrId(const Attr &v) { CHECK(!source); if (empty() || this->back() != v) Insert(v); return size(); }
    void Insert(const Attr &v);
  };

  virtual ~Drawable() {}
  virtual int  Id()                                            const { return 0; }
  virtual int  TexId()                                         const { return 0; }
  virtual bool Wide()                                          const { return 0; }
  virtual int  LeftBearing(                   const Attr *A=0) const { return 0; }
  virtual int  Baseline   (const LFL::Box *B, const Attr *A=0) const { return 0; }
  virtual int  Ascender   (const LFL::Box *B, const Attr *A=0) const { return B ? B->h : 0; }
  virtual int  Advance    (const LFL::Box *B, const Attr *A=0) const { return B ? B->w : 0; }
  virtual int  Layout     (      LFL::Box *B, const Attr *A=0) const { return B ? B->w : 0; }
  virtual void Draw       (GraphicsContext*,  const LFL::Box&) const = 0;
  virtual string DebugString() const { return ""; } 
  void DrawGD(GraphicsDevice *gd, const LFL::Box &b) const;
};

struct GraphicsContext {
  GraphicsDevice *gd;
  const Drawable::Attr *attr;
  GraphicsContext(GraphicsDevice *d=0, const Drawable::Attr *a=0) : gd(d), attr(a) {}
  void DrawTexturedBox(const Box &b, const float *texcoord=0, int orientation=0) { return DrawTexturedBox1(gd, b, texcoord, orientation); }
  void DrawGradientBox(const Box &b, const Color *c) { return DrawGradientBox1(gd, b, c); }
  void DrawCrimpedBox(const Box &b, const float *tc, int o, float sx=0, float sy=0) { return DrawCrimpedBox1(gd, b, tc, o, sx, sy); }
  void DrawBox3(const Box3 &b, const point &p=point(), const Color *c=0) { return DrawTexturedBox3(gd, b, p, c); }
  void DrawBox1(const Box &b) { return DrawTexturedBox(b); }
  static void DrawTexturedBox1(GraphicsDevice*, const Box&, const float *texcoord=0, int orientation=0);
  static void DrawGradientBox1(GraphicsDevice*, const Box&, const Color*);
  static void DrawCrimpedBox1(GraphicsDevice*, const Box&, const float *texcoord, int orientation, float scrollX=0, float scrollY=0);
  static void DrawTexturedBox3(GraphicsDevice*, const Box3&, const point &p=point(), const Color *c=0);
};

struct DrawableNop : public Drawable {
  void Draw(GraphicsContext*, const LFL::Box &B) const override {}
};

struct GraphicsDeviceInterface {
  struct TextureRef  { unsigned v= 0, w=0, h=0, t=0, f=0; operator unsigned() const { return v; } };
  struct DepthRef    { unsigned v= 0, w=0, h=0;           operator unsigned() const { return v; } };
  struct ShaderRef   { unsigned v= 0;                     operator unsigned() const { return v; } };
  struct AttribRef   { int      v=-1;                     operator int     () const { return v; } };
  struct UniformRef  { int      v=-1;                     operator int     () const { return v; } };
  struct ProgramRef  { unsigned v= 0;                     operator unsigned() const { return v; } };
  struct FrameBufRef { unsigned v= 0, w=0, h=0;           operator unsigned() const { return v; } };
};

struct Texture : public Drawable {
  typedef GraphicsDeviceInterface::TextureRef TextureRef;
  static const int preferred_pf, updatesystemimage_pf;
  GraphicsDeviceHolder *parent;
  TextureRef ID;
  unsigned char *buf;
  bool owner, buf_owner;
  int width, height, pf, cubemap;
  float coord[4] = { unit_texcoord[0], unit_texcoord[1], unit_texcoord[2], unit_texcoord[3] };

  Texture(GraphicsDeviceHolder *P, int w=0, int h=0, int PF=preferred_pf, const TextureRef &id=TextureRef()) : parent(P), ID(id), buf(0), owner(1), buf_owner(1), width(w), height(h), pf(PF), cubemap(0) {}
  Texture(GraphicsDeviceHolder *P, int w,   int h,   int PF,              unsigned char *B)                  : parent(P),         buf(B), owner(1), buf_owner(0), width(w), height(h), pf(PF), cubemap(0) {}
  Texture(const Texture &t) : parent(t.parent), ID(t.ID), buf(t.buf), owner(ID?0:1), buf_owner(buf?0:1), width(t.width), height(t.height), pf(t.pf), cubemap(t.cubemap) { memcpy(&coord, t.coord, sizeof(coord)); }
  virtual ~Texture() { Clear(); }

  void Bind() const;
  int TexId() const override { return ID; }
  string CoordString() const { return StrCat("[", coord[0], ", ", coord[1], ", ", coord[2], ", ", coord[3], "]"); } 
  string DebugString() const override { return StrCat("Texture(", ID, ": ", width, ", ", height, ", ", Pixel::Name(pf), ", ", CoordString(), ")"); }
  string HexDump() const { string v; for (int ls=LineSize(), i=0; i<height; i++) StrAppend(&v, Vec<unsigned char>::Str(buf+i*ls, ls, "%02x"), "\n"); return v; }
  point Dimension() const { return point(width, height); }
  int PixelSize() const { return Pixel::Size(pf); }
  int LineSize() const { return width * PixelSize(); }
  int BufferSize() const { return height * LineSize(); }
  int GDBufferType(GraphicsDevice*) const;

  /// ClearGL() is thread-safe.
  void ClearGL();
  void ResetGL(int flag);
  void RenewGL() { ClearGL(); Create(width, height); }
  TextureRef ReleaseGL() { auto ret = ID; ID = TextureRef(); return ret; }

  void ClearBuffer() { if (buf_owner) delete [] buf; buf = 0; buf_owner = 1; }
  unsigned char *RenewBuffer() { ClearBuffer(); buf = NewBuffer().release(); return buf; }
  unique_ptr<unsigned char[]> NewBuffer() const { return make_unique<unsigned char[]>(BufferSize()); }
  unique_ptr<unsigned char[]> ReleaseBuffer() { unsigned char *ret=0; swap(ret, buf); ClearBuffer(); return unique_ptr<unsigned char[]>(ret); }

  struct Flag { enum { CreateGL=1, CreateBuf=2, FlipY=4, Resample=8, RepeatGL=16 }; };
  void Create      (int PF=0)               { Create(width, height, PF); }
  void Create      (int W, int H, int PF=0) { Resize(W, H, PF, Flag::CreateGL); }
  void CreateBacked(int W, int H, int PF=0) { Resize(W, H, PF, Flag::CreateGL | Flag::CreateBuf); }
  void Resize(int W, int H, int PF=0, int flag=0);
  void Clear() { if (buf_owner) ClearBuffer(); if (owner) ClearGL(); }

  void AssignBuffer(Texture *t, bool become_owner=0) { AssignBuffer(t->buf, point(t->width, t->height), t->pf, become_owner); if (become_owner) t->buf_owner=0; }
  void AssignBuffer(      unsigned char *B, const point &dim, int PF, bool become_owner=0) { buf=B; width=dim.x; height=dim.y; pf=PF; buf_owner=become_owner; }
  void LoadBuffer  (const unsigned char *B, const point &dim, int PF, int linesize, int flag=0);
  void UpdateBuffer(const unsigned char *B, const point &dim, int PF, int linesize, int flag=0);
  void UpdateBuffer(const unsigned char *B, const ::LFL::Box &box, int PF, int linesize, int blit_flag=0);
  void FlipBufferY() { Texture t(parent); t.LoadBuffer(buf, Dimension(), pf, LineSize(), Flag::FlipY); ClearBuffer(); swap(buf, t.buf); }

  void LoadGL  (const MultiProcessTextureResource&);
  void LoadGL  (const unsigned char *B, const point &dim, int PF, int linesize, int flag=0);
  void UpdateGL(const unsigned char *B, const ::LFL::Box &box, int PF=0, int flag=0);
  void UpdateGL(const ::LFL::Box &b, int PF=0, int flag=0) { return UpdateGL(buf ? (buf+(b.y*width+b.x)*PixelSize()) : 0, b, PF, flag); }
  void UpdateGL() { UpdateGL(LFL::Box(0, 0, width, height)); }
  void LoadGL(int flag=0) { LoadGL(buf, point(width, height), pf, LineSize(), flag); }

  int Id() const override { return 0; }
  void Draw(GraphicsContext *gc, const LFL::Box &b) const override { Bind(); gc->DrawTexturedBox(b, coord); }
  virtual void DrawCrimped(GraphicsDevice *d, const LFL::Box &b, int ort, float sx, float sy) const { Bind(); GraphicsContext::DrawCrimpedBox1(d, b, coord, ort, sx, sy); }
  virtual int LayoutAtPoint(const point &p, LFL::Box *out) const { *out = LFL::Box(p, width, height); return width; } 

#ifdef LFL_APPLE
  CGContextRef CGBitMap();
  CGContextRef CGBitMap(int X, int Y, int W, int H);
#endif
#ifdef LFL_WINDOWS
  HBITMAP CreateGDIBitMap(HDC dc);
#endif
  static void Coordinates(float *texcoord, int w, int h, int wd, int hd);
  static void Coordinates(float *texcoord, const Box &b, int wd, int hd);
  static const int minx_coord_ind, miny_coord_ind, maxx_coord_ind, maxy_coord_ind;
  static const float unit_texcoord[4];
};

struct TextureArray {
  vector<Texture> a;
  void ClearGL()         { for (auto &i : a) i.ClearGL(); }
  void ResetGL(int flag) { for (auto &i : a) i.ResetGL(flag); }
  void DrawSequence(Asset *out, Entity *e, int *ind);
};

struct DepthTexture {
  typedef GraphicsDeviceInterface::DepthRef DepthRef;
  GraphicsDeviceHolder *parent;
  DepthRef ID;
  int width, height, df;
  bool owner=true;
  DepthTexture(GraphicsDeviceHolder *p, int w=0, int h=0, int DF=Depth::_16, const DepthRef &id=DepthRef()) :
    parent(p), ID(id), width(w), height(h), df(DF) {}
  virtual ~DepthTexture() { if (owner) ClearGL(); }

  struct Flag { enum { CreateGL=1 }; };
  void Create(int W, int H, int DF=0) { Resize(W, H, DF, Flag::CreateGL); }
  void Resize(int W, int H, int DF=0, int flag=0);
  void ClearGL();
  void ResetGL(int flag);
};

struct FrameBuffer {
  typedef GraphicsDeviceInterface::FrameBufRef FrameBufRef;
  typedef GraphicsDeviceInterface::TextureRef TextureRef;
  typedef GraphicsDeviceInterface::DepthRef DepthRef;
  GraphicsDeviceHolder *parent;
  FrameBufRef ID;
  int width, height;
  Texture tex;
  DepthTexture depth;
  bool owner=true;

  FrameBuffer(GraphicsDeviceHolder *p, int w=0, int h=0, const FrameBufRef &id=FrameBufRef()) :
    parent(p), ID(id), width(w), height(h), tex(p), depth(p) {}
  virtual ~FrameBuffer() { if (owner) ClearGL(); }

  struct Flag { enum { CreateGL=1, CreateTexture=2, CreateDepthTexture=4, ReleaseFB=8, RepeatGL=16 }; };
  void Create(int W, int H, int flag=0) { Resize(W, H, Flag::CreateGL | flag); }
  void Resize(int W, int H, int flag=0);

  void AllocTexture(Texture *out);
  void AllocTexture(TextureRef *out);
  void AllocDepthTexture(DepthTexture *out);

  void Attach(const TextureRef &t=TextureRef(), const DepthRef &d=DepthRef(), bool update_viewport=1);
  void Release(bool update_viewport=1);
  void ClearGL();
  void ResetGL(int flag);
};

struct ShaderDefines {
  string text;
  bool vertex_color, normals, tex_2d, tex_cube;
  ShaderDefines(                 bool vc=0, bool n=0, bool t2D=0, bool tC=0) : ShaderDefines("", vc, n, t2D, tC) {}
  ShaderDefines(const string &t, bool vc=0, bool n=0, bool t2D=0, bool tC=0) : text(t), vertex_color(vc), normals(n), tex_2d(t2D), tex_cube(tC) {
    if (vertex_color) StrAppend(&text, "#define VERTEXCOLOR\r\n");
    if (normals)      StrAppend(&text, "#define NORMALS\r\n");
    if (tex_2d)       StrAppend(&text, "#define TEX2D\r\n");
    if (tex_cube)     StrAppend(&text, "#define TEXCUBE\r\n");
  }
};

struct Shader {
  static const int MaxVertexAttrib = 4;
  GraphicsDeviceHolder *parent;
  string name;
  float scale=0;
  int unused_attrib_slot[MaxVertexAttrib];
  bool dirty_material=0, dirty_light_pos[4], dirty_light_color[4], owner=true;
  GraphicsDeviceInterface::ProgramRef ID;
  GraphicsDeviceInterface::AttribRef slot_position, slot_normal, slot_tex, slot_color;
  GraphicsDeviceInterface::UniformRef uniform_model, uniform_invview, uniform_modelview, uniform_modelviewproj,
    uniform_campos, uniform_tex, uniform_cubetex, uniform_normalon, uniform_texon, uniform_coloron, uniform_cubeon,
    uniform_colordefault, uniform_material_ambient, uniform_material_diffuse, uniform_material_specular,
    uniform_material_emission, uniform_light0_pos, uniform_light0_ambient, uniform_light0_diffuse,
    uniform_light0_specular;
  Shader(GraphicsDeviceHolder *P) : parent(P) { memzeros(dirty_light_pos); memzeros(dirty_light_color); }
  virtual ~Shader() { if (owner) ClearGL(); }

  static int Create(GraphicsDeviceHolder*, const string &name, const string &vertex_shader, const string &fragment_shader, const ShaderDefines&, Shader *out);
  static int CreateShaderToy(GraphicsDeviceHolder*, const string &name, const string &fragment_shader, Shader *out);
  GraphicsDeviceInterface::UniformRef GetUniformIndex(const string &name);
  void SetUniform1i(const string &name, float v);
  void SetUniform1f(const string &name, float v);
  void SetUniform2f(const string &name, float v1, float v2);
  void SetUniform3f(const string &name, float v1, float v2, float v3);
  void SetUniform4f(const string &name, float v1, float v2, float v3, float v4);
  void SetUniform3fv(const string &name, const float *v);
  void SetUniform3fv(const string &name, int n, const float *v);
  void ClearGL();
};

struct Shaders {
  Shader shader_default, shader_normals, shader_cubemap, shader_cubenorm;
  Shaders(GraphicsDeviceHolder*);
  void SetGlobalUniform1f(GraphicsDevice*, const string &name, float v);
  void SetGlobalUniform2f(GraphicsDevice*, const string &name, float v1, float v2);
};

struct GraphicsDevice : public GraphicsDeviceInterface {
  struct Type { enum { OPENGL = 1, DIRECTX = 2, METAL = 3, BGFX = 4 }; };
  struct Constants {
    int Float, Points, Lines, LineLoop, Triangles, TriangleStrip, Polygon;
    int Texture2D, TextureCubeMap, UnsignedByte, UnsignedInt, FramebufferComplete, FramebufferBinding, FramebufferUndefined;
    int Ambient, Diffuse, Specular, Emission, Position, AmbientAndDiffuse;
    int One, SrcAlpha, OneMinusSrcAlpha, OneMinusDstColor, TextureWrapS, TextureWrapT, ClampToEdge;
    int VertexShader, FragmentShader, ShaderVersion, Extensions;
    int GLEWVersion, Version, Vendor, DepthBits, ScissorTest;
    int ActiveUniforms, ActiveAttributes, MaxVertexAttributes, MaxVertexUniformComp, MaxViewportDims, ViewportBox, ScissorBox;
    int Fill, Line, Point, GLPreferredBuffer, GLInternalFormat;
  };

  Constants c;
  Window *parent;
  const int type, version;
  FrameBufRef default_framebuffer;
  int default_draw_mode = DrawMode::_2D, draw_mode = 0;
  bool done_init = 0, have_framebuffer = 1, have_cubemap = 1, have_npot_textures = 1;
  bool blend_enabled = 0, invert_view_matrix = 0, track_model_matrix = 0, dont_clear_deferred = 0;
  CategoricalVariable<int> tex_mode, fill_mode;
  string vertex_shader, pixel_shader;
  Shader *shader = 0;
  Shaders *shaders;
  v3 camera_pos;
  m44 invview_matrix, model_matrix;
  Color clear_color = Color::black;
  vector<Color> default_color;
  vector<vector<Box>> scissor_stack;
  vector<int*> buffers;
  FrameBuffer *attached_framebuffer = 0;
  Void glew_context = 0;

  GraphicsDevice(int T, const Constants &cs, Window *W, int V, Shaders *S) : 
    c(cs), parent(W), type(T), version(V), tex_mode(2, 1, 0), fill_mode(3, c.Fill, c.Line, c.Point), shaders(S), scissor_stack(1) {}
  virtual ~GraphicsDevice() {}

  virtual void Init(AssetLoading*, const Box&) = 0;
  virtual void Finish() = 0;
  virtual void MarkDirty() = 0;
  virtual void Clear() = 0;
  virtual void ClearDepth() = 0;
  virtual void Flush() = 0;
  
  virtual bool GetEnabled(int) = 0;
  virtual bool GetShaderSupport() = 0;
  virtual void GetIntegerv(int t, int *out) = 0;
  virtual const char *GetString(int t) = 0;
  virtual const char *GetGLEWString(int t) = 0;
  virtual void CheckForError(const char *file, int line) = 0;

  virtual void EnableTexture() = 0;
  virtual void DisableTexture() = 0;
  virtual void EnableLighting() = 0;
  virtual void DisableLighting() = 0;
  virtual void EnableNormals() = 0;
  virtual void DisableNormals() = 0;
  virtual void EnableVertexColor() = 0;
  virtual void DisableVertexColor() = 0;
  virtual void EnableLight(int n) = 0;
  virtual void DisableLight(int n) = 0;
  virtual void DisableCubeMap() = 0;
  virtual void EnableScissor() = 0;
  virtual void DisableScissor() = 0;
  virtual void EnableDepthTest() = 0;
  virtual void DisableDepthTest() = 0;
  virtual void EnableBlend() = 0;
  virtual void DisableBlend() = 0;
  virtual void BlendMode(int sm, int tm) = 0;
  
  virtual void MatrixProjection() = 0;
  virtual void MatrixModelview() = 0;
  virtual void LoadIdentity() = 0;
  virtual void PushMatrix() = 0;
  virtual void PopMatrix() = 0;
  virtual void GetMatrix(m44 *out) = 0;
  virtual void PrintMatrix() = 0;
  virtual void Scalef(float x, float y, float z) = 0;
  virtual void Rotatef(float angle, float x, float y, float z) = 0;
  virtual void Ortho(float l, float r, float b, float t, float nv, float fv) = 0;
  virtual void Frustum(float l, float r, float b, float t, float nv, float fv) = 0;
  virtual void Mult(const float *m) = 0;
  virtual void Translate(float x, float y, float z) = 0;

  virtual void ViewPort(Box w) = 0;
  virtual void Scissor(Box w) = 0;
  virtual void ClearColor(const Color &c) = 0;
  virtual void Color4f(float r, float g, float b, float a) = 0;
  virtual void Light(int n, int t, float *color) = 0;
  virtual void Material(int t, float *color) = 0;
  virtual void PointSize(float n) = 0;
  virtual void LineWidth(float n) = 0;

  virtual TextureRef CreateTexture(int t, unsigned w, unsigned h, int f) = 0;
  virtual void DelTexture(TextureRef id) = 0;
  virtual void UpdateTexture(const TextureRef&, int targ, int l, int fi, int b, int t, const void*) = 0;
  virtual void UpdateSubTexture(const TextureRef&, int targ, int l, int xo, int yo, int w, int h, int t, const void*) = 0;
  virtual void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) = 0;
  virtual void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) = 0;
  virtual void BindTexture(const TextureRef &n) = 0;
  virtual void BindCubeMap(const TextureRef &n) = 0;
  virtual void ActiveTexture(int n) = 0;
  virtual void TextureGenLinear() = 0;
  virtual void TextureGenReflection() = 0;
  virtual void TexParameter(int, int, int) = 0;

  virtual FrameBufRef CreateFrameBuffer(unsigned w, unsigned h) = 0;
  virtual void DelFrameBuffer(FrameBufRef id) = 0;
  virtual void BindFrameBuffer(const FrameBufRef&) = 0;
  virtual void FrameBufferTexture(const TextureRef&) = 0;
  virtual void FrameBufferDepthTexture(const DepthRef&) = 0;
  virtual int CheckFrameBufferStatus() = 0;
  
  virtual DepthRef CreateDepthBuffer(unsigned w, unsigned h, int f) = 0;
  virtual void DelDepthBuffer(DepthRef id) = 0;
  virtual void BindDepthBuffer(const DepthRef &) = 0;

  virtual ProgramRef CreateProgram() = 0;
  virtual void DelProgram(ProgramRef) = 0;
  virtual ShaderRef CreateShader(int t) = 0;
  virtual void CompileShader(const ShaderRef&, vector<const char*> source) = 0;
  virtual void AttachShader(const ProgramRef&, const ShaderRef&) = 0;
  virtual void DelShader(const ShaderRef&) = 0;
  virtual void BindAttribLocation(const ProgramRef&, int loc, const string &name) = 0;
  virtual bool LinkProgram(const ProgramRef&) = 0;
  virtual void GetProgramiv(const ProgramRef&, int t, int *out) = 0;
  virtual AttribRef GetAttribLocation(const ProgramRef&, const string &name) = 0;
  virtual UniformRef GetUniformLocation(const ProgramRef&, const string &name) = 0;
  virtual void Uniform1i(const UniformRef&, int v) = 0;
  virtual void Uniform1f(const UniformRef&, float v) = 0;
  virtual void Uniform2f(const UniformRef&, float v1, float v2) = 0;
  virtual void Uniform3f(const UniformRef&, float v1, float v2, float v3) = 0;
  virtual void Uniform4f(const UniformRef&, float v1, float v2, float v3, float v4) = 0;
  virtual void Uniform3fv(const UniformRef&, int n, const float *v) = 0;
  virtual void Uniform4fv(const UniformRef&, int n, const float *v) = 0;
  virtual void UniformMatrix4fv(const UniformRef&, int n, const float *v) = 0;

  virtual void UseShader(Shader *shader) = 0;
  virtual void Screenshot(Texture *out) = 0;
  virtual void ScreenshotBox(Texture *out, const Box &b, int flag) = 0;
  virtual void DumpTexture(Texture *out, const TextureRef &t=TextureRef()) = 0;

  virtual bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty, int pt=0) = 0;
  virtual void TexPointer   (int m, int t, int w, int o, float *tex,   int l, int *out, bool dirty) = 0;
  virtual void ColorPointer (int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) = 0;
  virtual void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) = 0;
  virtual void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) = 0;
  virtual void DrawArrays(int t, int o, int n) = 0;
  virtual void DeferDrawArrays(int t, int o, int n) = 0;
  virtual void SetDontClearDeferred(bool v) {}
  virtual void ClearDeferred() {}

  int TextureDim(int x) { return have_npot_textures ? x : NextPowerOfTwo(x); }
  int RegisterBuffer(int *b) { buffers.push_back(b); return -1; }
  void FillColor(const Color &c) { DisableTexture(); SetColor(c); };
  void SetColor(const Color &c) { Color4f(c.r(), c.g(), c.b(), c.a()); }
  void PushColor(const Color &c) { PushColor(); SetColor(c); }
  void PushColor();
  void PopColor();
  void RestoreViewport(int drawmode);
  void TranslateRotateTranslate(float a, const Box&);
  void DrawMode(int drawmode, bool flush=1);
  void DrawMode(int drawmode, const Box&, bool flush=1);
  void EnableLayering() { DisableDepthTest(); DisableLighting(); EnableBlend(); EnableTexture(); }
  void LookAt(const v3 &pos, const v3 &targ, const v3 &up);
  void PushScissorOffset(const Box &t, const Box &w) { PushScissor(Box(w.x-t.x, w.y-t.y, w.w, w.h)); }
  void PushScissor(Box w);
  void PopScissor();
  void PushScissorStack();
  void PopScissorStack();
  Box GetViewport();
  Box GetScissorBox();
  int GetPrimitive(int geometry_primtype);
  void DrawPixels(const Box &b, const Texture &tex);
  void InitDefaultLight();

  static unique_ptr<GraphicsDevice> Create(Window*, Shaders*);
};

struct ShaderBasedGraphicsDevice : public GraphicsDevice {
  using GraphicsDevice::GraphicsDevice;
  struct BoundTexture {
    unsigned t, n; int l;
    bool operator!=(const BoundTexture &x) const { return t != x.t || n != x.n || l != x.l; };
  };
  struct VertexAttribute {
    int m, t, w, o;
    bool operator!=(const VertexAttribute &x) const { return m != x.m || t != x.t || w != x.w || o != x.o; };
  };
  struct Deferred {
    int prim_type=0, vertex_size=0, vertexbuffer=-1, vertexbuffer_size=1024*4*4, vertexbuffer_len=0, vertexbuffer_appended=0, draw_calls=0;
  };

  vector<m44> modelview_matrix, projection_matrix;
  bool dirty_matrix=1, dirty_color=1, cubemap_on=0, normals_on=0, texture_on=0, colorverts_on=0, lighting_on=0;
  int matrix_target=-1, bound_vertexbuffer=-1, bound_indexbuffer=-1;
  VertexAttribute vertex_attr, tex_attr, color_attr, normal_attr;
  BoundTexture bound_texture;
  LFL::Material material;
  LFL::Light light[4];
  Deferred deferred;

  virtual void UpdateTexture()    = 0;
  virtual void UpdateNormals()    = 0;
  virtual void UpdateColorVerts() = 0;

  void MarkDirty() override { dirty_matrix = dirty_color = 1; }
  bool GetShaderSupport() override { return true; }

  void EnableTexture()      override;
  void DisableTexture()     override;
  void EnableLighting()     override;
  void DisableLighting()    override;
  void EnableNormals()      override;
  void DisableNormals()     override;
  void EnableVertexColor()  override;
  void DisableVertexColor() override;
  void EnableLight(int n)   override;
  void DisableLight(int n)  override;
  void DisableCubeMap()     override;
  
  void MatrixProjection()                                              override;
  void MatrixModelview()                                               override;
  void LoadIdentity()                                                  override;
  void PushMatrix()                                                    override;
  void PopMatrix()                                                     override;
  void GetMatrix(m44 *out)                                             override;
  void PrintMatrix()                                                   override;
  void Scalef(float x, float y, float z)                               override;
  void Rotatef(float angle, float x, float y, float z)                 override;
  void Ortho  (float l, float r, float b, float t, float nv, float fv) override;
  void Frustum(float l, float r, float b, float t, float nv, float fv) override;
  void Mult(const float *m)                                            override;
  void Translate(float x, float y, float z)                            override;

  void Color4f(float r, float g, float b, float a) override;
  void Light(int n, int t, float *v)               override;
  void Material(int t, float *v)                   override;

  void SetDontClearDeferred(bool v) override;
  int AddDeferredVertexSpace(int l);

  vector<m44> *TargetMatrix();
  void UpdateShader();
  void UpdateMatrix();
  void UpdateColor();
  void UpdateMaterial();
  void PushDirtyState();
};

struct Scissor {
  GraphicsDevice *gd;
  Scissor(GraphicsDevice *d, int x, int y, int w, int h) : gd(d) { gd->PushScissor(Box(x, y, w, h)); }
  Scissor(GraphicsDevice *d, const Box &w)               : gd(d) { gd->PushScissor(w); }
  ~Scissor()                                                     { gd->PopScissor(); }
};

struct ScissorStack {
  GraphicsDevice *gd;
  ScissorStack(GraphicsDevice *d) : gd(d) { gd->PushScissorStack(); }
  ~ScissorStack()                         { gd->PopScissorStack(); }
};

struct ScopedDrawMode {
  GraphicsDevice *gd;
  int prev_mode;
  bool nop;
  ScopedDrawMode(GraphicsDevice *d, int dm) : gd(d), prev_mode(gd->draw_mode), nop(dm == DrawMode::NullOp) { if (!nop) gd->DrawMode(dm,        0); }
  ~ScopedDrawMode()                                                                                        { if (!nop) gd->DrawMode(prev_mode, 0); }
};

struct ScopedColor {
  GraphicsDevice *gd;
  ~ScopedColor()                                         { gd->PopColor(); }
  ScopedColor(GraphicsDevice *d, const Color &c) : gd(d) { gd->PushColor(c); }
};

struct ScopedFillColor {
  GraphicsDevice *gd;
  ~ScopedFillColor()                                         { gd->PopColor(); }
  ScopedFillColor(GraphicsDevice *d, const Color &c) : gd(d) { gd->PushColor(); gd->FillColor(c); }
};

struct ScopedClearColor {
  GraphicsDevice *gd;
  bool enabled;
  Color prev_color;
  ~ScopedClearColor()                                                                                  { if (enabled) gd->ClearColor(prev_color); }
  ScopedClearColor(GraphicsDevice *d, const Color *c) : gd(d), enabled(c), prev_color(gd->clear_color) { if (enabled) gd->ClearColor(*c); }
  ScopedClearColor(GraphicsDevice *d, const Color &c) : gd(d), enabled(1), prev_color(gd->clear_color) {              gd->ClearColor(c); }
};

struct ScopedDontClearDeferred {
  GraphicsDevice *gd;
  bool previous_dont_clear_deferred;
  ~ScopedDontClearDeferred()                                                                                { if (!previous_dont_clear_deferred) gd->SetDontClearDeferred(false); }
  ScopedDontClearDeferred(GraphicsDevice *d) : gd(d), previous_dont_clear_deferred(gd->dont_clear_deferred) { if (!previous_dont_clear_deferred) gd->SetDontClearDeferred(true);  }
};

}; // namespace LFL
#endif // LFL_CORE_APP_VIDEO_H__
