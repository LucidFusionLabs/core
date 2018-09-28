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

#ifndef LFL_CORE_APP_FONT_H__
#define LFL_CORE_APP_FONT_H__

namespace LFL {
DECLARE_string(font_engine);
DECLARE_string(font);
DECLARE_string(font_family);
DECLARE_int(font_size);
DECLARE_int(font_flag);
DECLARE_int(missing_glyph);
DECLARE_bool(atlas_dump);
DECLARE_string(atlas_font_sizes);
DECLARE_int(scale_font_height);
DECLARE_int(add_font_size);
DECLARE_int(glyph_table_size);
DECLARE_int(glyph_table_start);
DECLARE_bool(subpixel_fonts);

struct FontEngine {
  struct Resource { virtual ~Resource() {} };
  Fonts *parent;
  FontEngine(Fonts *p) : parent(p) {}
  virtual ~FontEngine() {}
  virtual const char*      Name() const = 0;
  virtual void             Shutdown() {}
  virtual void             SetDefault() {}
  virtual bool             Init(const FontDesc&) { return true; }
  virtual unique_ptr<Font> Open(const FontDesc&) = 0;
  virtual bool             HaveGlyph (Font *f, char16_t) { return true; }
  virtual int              InitGlyphs(Font *f,       Glyph *g, int n) = 0;
  virtual int              LoadGlyphs(Font *f, const Glyph *g, int n) = 0;
  virtual string           DebugString(Font *f) const = 0;
};

struct Glyph : public Drawable {
  char16_t id=0;
  short bearing_x=0, bearing_y=0, advance=0;
  bool wide=0, space=0;
  union Internal {
    struct IPCClient { int id; }                                                   ipcclient;
    struct FreeType  { int id; Font *substitute; }                                 freetype;
    struct CoreText  { int id; float origin_x, origin_y, width, height, advance; } coretext;
    struct FillColor { unsigned color;                                           } fillcolor;
  } internal;
  mutable Texture tex;
  mutable bool ready=0;
  Glyph(GraphicsDeviceHolder *p, int i=0) : id(i), tex(p) { memzero(internal); tex.owner=0; }

  bool operator<(const Glyph &y) const { return id < y.id; }
  void FromMetrics(const GlyphMetrics &m);
  void FromArray(const double *in,  int l);
  int  ToArray  (      double *out, int l);

  int  Id()    const override { return id; }
  int  TexId() const override { return tex.ID; }
  bool Wide()  const override { return wide; }

  int  Baseline   (const LFL::Box *b, const Drawable::Attr *a=0) const override;
  int  Ascender   (const LFL::Box *b, const Drawable::Attr *a=0) const override;
  int  Advance    (const LFL::Box *b, const Drawable::Attr *a=0) const override;
  int  LeftBearing(                   const Drawable::Attr *a=0) const override;
  int  Layout     (      LFL::Box *b, const Drawable::Attr *a=0) const override;
  void Draw       (GraphicsContext*,  const LFL::Box&)           const override;
};

struct FillColor : public Glyph {
  FillColor(GraphicsDeviceHolder *p, ColorDesc c) : Glyph(p) { internal.fillcolor.color = c; }
  virtual void Draw(GraphicsContext*, const LFL::Box&) const override;
};

struct GlyphMetrics {
  char16_t id=0;
  short width=0, height=0, bearing_x=0, bearing_y=0, advance=0;
  bool wide=0, space=0;
  int tex_id=0;
  GlyphMetrics() {}
  GlyphMetrics(const Glyph &g) : id(g.id), width(g.tex.width), height(g.tex.height), bearing_x(g.bearing_x),
    bearing_y(g.bearing_y), advance(g.advance), wide(g.wide), space(g.space), tex_id(g.tex.ID) {}
};

struct GlyphCache {
  typedef function<void(const Box&, unsigned char *buf, int linesize, int pf)> FilterCB;
  Box dim;
  Texture tex;
  unique_ptr<Flow> flow;
#if defined(LFL_APPLE)
  CGContextRef cgcontext=0;
#elif defined(LFL_WINDOWS)
  HDC hdc=0;
#endif
  vector<const Glyph*> glyph;
  int max_width=128, max_height=128;
  GlyphCache(GraphicsDeviceHolder*, const GraphicsDevice::TextureRef&, int W, int H=0);
  ~GlyphCache();

  bool ShouldCacheGlyph(const Texture &t) const {
    CHECK_LT(t.width,  1024*1024);
    CHECK_LT(t.height, 1024*1024);
    return t.width < max_width && t.height < max_height;
  }

  void Clear(bool reopen=true);
  bool Add(point *out, float *out_texcoord, int w, int h, int max_height=0);
  void Load(const Font*, const Glyph*, const unsigned char *buf, int linesize, int pf, const FilterCB &f=FilterCB());
#if defined(LFL_APPLE)
  void Load(const Font*, const Glyph*, CGFontRef cgfont, int size);
#elif defined(LFL_WINDOWS)
  void Load(const Font*, const Glyph*, HFONT hfont, int size, HDC dc);
#endif
};

struct GlyphMap {
  int table_start;
  vector<Glyph>             table;
  unordered_map<int, Glyph> index;
  shared_ptr<GlyphCache>    cache;
  GlyphMap(GraphicsDeviceHolder *p, shared_ptr<GlyphCache> C = shared_ptr<GlyphCache>()) :
    table_start(FLAGS_glyph_table_start), cache(move(C))
  { for (int i=0; i<FLAGS_glyph_table_size; ++i) table.emplace_back(p, table_start + i); }
};

struct Font {
  struct DrawFlag {
    enum {
      NoWrap=1<<6, GlyphBreak=1<<7, AlignCenter=1<<8, AlignRight=1<<9, 
      Underline=1<<10, Overline=1<<11, Midline=1<<12, Blink=1<<13,
      Uppercase=1<<14, Lowercase=1<<15, Capitalize=1<<16, Clipped=1<<17,
      DontAssignFlowP=1<<18, DontCompleteFlow=1<<19
    };
    static int Orientation(int f) { return f & 0xf; };
  };

  short size=0, ascender=0, descender=0, max_width=0, fixed_width=-1, missing_glyph=0;
  bool mono=0, mix_fg=0, has_bg=0, fix_metrics=0;
  float scale=0;
  int flag=0;
  Color fg, bg;
  RefCounter ref;
  const FontDesc *desc=0;
  FontEngine* engine;
  shared_ptr<GlyphMap> glyph;
  shared_ptr<FontEngine::Resource> resource;

  virtual ~Font() {}
  Font(FontEngine *E, const FontDesc &D, shared_ptr<FontEngine::Resource> R) :
    size(D.size), missing_glyph(FLAGS_missing_glyph), mono(D.flag & FontDesc::Mono),
    flag(D.flag), fg(D.fg), bg(D.bg), engine(E), resource(move(R)) {}

  short Height() const { return ascender + descender; }
  short FixedWidth() const { return X_or_Y(fixed_width, mono ? max_width : 0); }
  string DebugString() { return engine ? engine->DebugString(this) : string(); }

  Glyph *FindGlyph        (char16_t gind);
  Glyph *FindOrInsertGlyph(char16_t gind);

  void Select(GraphicsDevice*);
  void UpdateMetrics(Glyph *g);
  void SetMetrics(short a, short d, short mw, short fw, short mg, bool mf, bool hb, bool fm, float f)
  { ascender=a; descender=d; max_width=mw; fixed_width=fw; missing_glyph=mg; mix_fg=mf; has_bg=hb; fix_metrics=fm; scale=f; }
  int GetGlyphWidth(int g) { return RoundXY_or_Y(scale, FindGlyph(g)->advance); }
  void DrawGlyph(GraphicsDevice*, int g, const Box &w);
  void DrawGlyphWithAttr(GraphicsContext*, int g, const Box &w);

  template <class X> void Size(const StringPieceT<X> &text, Box *out, int width=0, int flag=0, int *lines_out=0);
  /**/               void Size(const string          &text, Box *out, int width=0, int flag=0, int *lines_out=0) { return Size(StringPiece           (text), out, width, flag, lines_out); }
  /**/               void Size(const String16        &text, Box *out, int width=0, int flag=0, int *lines_out=0) { return Size(String16Piece         (text), out, width, flag, lines_out); }
  template <class X> void Size(const X               *text, Box *out, int width=0, int flag=0, int *lines_out=0) { return Size(StringPiece::Unbounded(text), out, width, flag, lines_out); }

  template <class X> int Lines(const StringPieceT<X> &text, int width, bool word_wrap=1) { if (!width) return 1; Box b; Size(text, &b, width, word_wrap?0:DrawFlag::GlyphBreak); return b.h / Height(); }
  /**/               int Lines(const string          &text, int width, bool word_wrap=1) { return Lines(StringPiece           (text), width, word_wrap); }
  /**/               int Lines(const String16        &text, int width, bool word_wrap=1) { return Lines(String16Piece         (text), width, word_wrap); }
  template <class X> int Lines(const X               *text, int width, bool word_wrap=1) { return Lines(StringPiece::Unbounded(text), width, word_wrap); }

  template <class X> int Width(const StringPieceT<X> &text) { Box b; Size(text, &b); if (b.w) CHECK_EQ(b.h, Height()); return b.w; }
  /**/               int Width(const string          &text) { return Width(StringPiece           (text)); }
  /**/               int Width(const String16        &text) { return Width(String16Piece         (text)); }
  template <class X> int Width(const X               *text) { return Width(StringPiece::Unbounded(text)); }

  template <class X> void Shape(const StringPieceT<X> &text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0);
  /**/               void Shape(const string          &text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0) { return Shape(StringPiece           (text), box, out, draw_flag, attr_id); }
  /**/               void Shape(const String16        &text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0) { return Shape(String16Piece         (text), box, out, draw_flag, attr_id); }
  template <class X> void Shape(const X               *text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0) { return Shape(StringPiece::Unbounded(text), box, out, draw_flag, attr_id); }

  template <class X> int Draw(GraphicsDevice *gd, const StringPieceT<X> &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw<X>(gd,                        text,  Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }
  /**/               int Draw(GraphicsDevice *gd, const string          &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw   (gd, StringPiece           (text), Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }
  /**/               int Draw(GraphicsDevice *gd, const String16        &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw   (gd, String16Piece         (text), Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }
  template <class X> int Draw(GraphicsDevice *gd, const X               *text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw   (gd, StringPiece::Unbounded(text), Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }

  template <class X> int Draw(GraphicsDevice *gd, const StringPieceT<X> &text, const Box &box, vector<Box> *lb=0, int draw_flag=0);
  /**/               int Draw(GraphicsDevice *gd, const string          &text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw(gd, StringPiece           (text), box, lb, draw_flag); }
  /**/               int Draw(GraphicsDevice *gd, const String16        &text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw(gd, String16Piece         (text), box, lb, draw_flag); }
  template <class X> int Draw(GraphicsDevice *gd, const X               *text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw(gd, StringPiece::Unbounded(text), box, lb, draw_flag); }
};

struct FakeFontEngine : public FontEngine {
  static const int size = 10, fixed_width = 8, ascender = 9, descender = 5;
  static const char16_t wide_glyph_begin = 0x1FC, wide_glyph_end = 0x1FD;

  // U+1FC = 'AE'; encoded: c7bc,  U+1FD = 'ae', encoded: c7bd
  FontDesc fake_font_desc;
  Font fake_font;
  FakeFontEngine(Fonts*);

  const char *Name() const override { return "FakeFontEngine"; }
  unique_ptr<Font> Open(const FontDesc&) override { return make_unique<Font>(fake_font); }
  int LoadGlyphs(Font *f, const Glyph *g, int n) override { return n; }
  int InitGlyphs(Font *f,       Glyph *g, int n) override;
  string DebugString(Font *f) const override { return "FakeFontEngineFont"; }

  static const char *Filename() { return "__FakeFontFilename__"; }
};

struct AtlasFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource { Font *primary=0; };
  typedef map<size_t, Font*> FontSizeMap;
  typedef map<size_t, FontSizeMap> FontFlagMap;
  typedef map<size_t, FontFlagMap> FontColorMap;
  typedef map<string, FontColorMap> FontMap;
  FontMap font_map;
  bool in_init=0;
  AtlasFontEngine(Fonts *f);

  const char*      Name() const override { return "AtlasFontEngine"; }
  void             SetDefault() override;
  bool             Init(const FontDesc&) override;
  unique_ptr<Font> Open(const FontDesc&) override;
  bool             HaveGlyph (Font *f, char16_t) override { return false; }
  int              InitGlyphs(Font *f,       Glyph *g, int n) override { return n; }
  int              LoadGlyphs(Font *f, const Glyph *g, int n) override { return n; }
  string           DebugString(Font *f) const override;

  static unique_ptr<Font> OpenAtlas(Fonts*, const FontDesc&);
  static void WriteAtlas(ApplicationInfo*, const string &name, Font *glyphs, Texture *t);
  static void WriteAtlas(ApplicationInfo*, const string &name, Font *glyphs);
  static void WriteGlyphFile(ApplicationInfo*, const string &name, Font *glyphs);
  static unique_ptr<Font> MakeFromPNGFiles(Fonts *fonts, const string &name, const vector<string> &png, const point &atlas_dim);
  static void SplitIntoPNGFiles(GraphicsDeviceHolder*, const string &input_png_fn, const map<int, v4> &glyphs, const string &dir_out);
  static int Dimension(int n, int w, int h) { return 1 << max(8,FloorLog2(sqrt((w+4)*(h+4)*n))); }
};

struct FreeTypeFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    string name, content;
    FT_FaceRec_ *face=0;
    virtual ~Resource();
    Resource(FT_FaceRec_ *FR=0, const string &N="", string *C=0) : name(N), face(FR) { if (C) swap(*C, content); }
  };
  unordered_map<FontDesc, Font*, FontDesc::Hasher, FontDesc::Equal> font_map;
  unordered_map<string, shared_ptr<Resource>> resource;
  GlyphCache::FilterCB subpixel_filter = &FreeTypeFontEngine::SubPixelFilter;
  FreeTypeFontEngine(Fonts *p) : FontEngine(p) {}

  const char*      Name() const override { return "FreeTypeFontEngine"; }
  void             SetDefault() override;
  bool             Init(const FontDesc&) override;
  unique_ptr<Font> Open(const FontDesc&) override;
  int              InitGlyphs(Font *f,       Glyph *g, int n) override;
  int              LoadGlyphs(Font *f, const Glyph *g, int n) override;
  string           DebugString(Font *f) const override;

  static void Init();
  static void SubPixelFilter(const Box &b, unsigned char *buf, int linesize, int pf);
  static unique_ptr<Resource> OpenFile  (const FontDesc&);
  static unique_ptr<Resource> OpenBuffer(const FontDesc&, string *content);
};

#ifdef LFL_APPLE
struct CoreTextFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    string name;
    CGFontRef cgfont;
    hb_face_t *hb_face=0;
    int flag;
    virtual ~Resource();
    Resource(const char *N=0, CGFontRef CGF=0, int F=0) : name(BlankNull(N)), cgfont(CGF), flag(F) {}
  };
  unordered_map<string, shared_ptr<Resource>> resource;
  CoreTextFontEngine(Fonts *p) : FontEngine(p) {}

  const char*      Name() const override { return "CoreTextFontEngine"; }
  void             SetDefault() override;
  unique_ptr<Font> Open(const FontDesc&) override;
  int              InitGlyphs(Font *f,       Glyph *g, int n) override;
  int              LoadGlyphs(Font *f, const Glyph *g, int n) override;
  string           DebugString(Font *f) const override;

  struct Flag { enum { WriteAtlas=1 }; };
  static Font *Open(const string &name, int size, Color c, int flag, int ct_flag);
  static void GetSubstitutedFont(Font*, CTFontRef, char16_t gid, CGFontRef *cgout, CTFontRef *ctout, int *id_out);
  static void AssignGlyph(Glyph *out, const CGRect &bounds, struct CGSize &advance);
  static v2 GetAdvanceBounds(Font*);
};
#else
struct CoreTextFontEngine {};
#endif

#ifdef LFL_WINDOWS
struct GDIFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    string name;
    HFONT hfont;
    int flag;
    virtual ~Resource();
    Resource(const char *N = 0, HFONT H = 0, int F = 0) : name(BlankNull(N)), hfont(H), flag(F) {}
  };
  unordered_map<string, shared_ptr<Resource>> resource;
  HDC hdc=0;
  GDIFontEngine(Fonts*);
  ~GDIFontEngine();

  const char*      Name() const override { return "GDIFontEngine"; }
  void             Shutdown() override;
  void             SetDefault() override;
  unique_ptr<Font> Open(const FontDesc&) override;
  int              InitGlyphs(Font *f, Glyph *g, int n) override;
  int              LoadGlyphs(Font *f, const Glyph *g, int n) override;
  string           DebugString(Font *f) const override;

  struct Flag { enum { WriteAtlas = 1 }; };
  static Font *Open(const string &name, int size, Color c, int flag, int ct_flag);
  static bool GetSubstitutedFont(Font *f, HFONT hfont, char16_t glyph_id, HDC hdc, HFONT *hfontout);
  static void AssignGlyph(Glyph *out, const ::SIZE &bounds, const ::SIZE &advance);
};
#else
struct GDIFontEngine {};
#endif

#if defined(LFL_LINUX)
struct FCFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    unique_ptr<Font> font;
    Resource(unique_ptr<Font> f) : font(move(f)) {}
  };
  unordered_map<string, shared_ptr<Resource>> resource;
  FCFontEngine(Fonts *p) : FontEngine(p) {}

  const char*      Name() const override { return "FCFontEngine"; }
  void             Shutdown() override;
  void             SetDefault() override;
  unique_ptr<Font> Open(const FontDesc&) override;
  int              InitGlyphs(Font *f, Glyph *g, int n) override;
  int              LoadGlyphs(Font *f, const Glyph *g, int n) override;
  string           DebugString(Font *f) const override;

  unique_ptr<Font> OpenTTF(const FontDesc&);
  static void Init();
};
#else
struct FCFontEngine {};
#endif

#if defined(LFL_ANDROID)
struct AndroidFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    unique_ptr<Font> font;
    Resource(unique_ptr<Font> f) : font(move(f)) {}
  };
  unordered_map<string, shared_ptr<Resource>> resource;
  AndroidFontEngine(Fonts *p) : FontEngine(p) {}

  const char*      Name() const override { return "AndroidFontEngine"; }
  void             Shutdown() override;
  void             SetDefault() override;
  unique_ptr<Font> Open(const FontDesc&) override;
  int              InitGlyphs(Font *f, Glyph *g, int n) override;
  int              LoadGlyphs(Font *f, const Glyph *g, int n) override;
  string           DebugString(Font *f) const override;

  unique_ptr<Font> OpenTTF(const FontDesc&);
  static void Init();
};
#else
struct AndroidFontEngine {};
#endif

struct IPCClientFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource { int id; Resource(int I=0) : id(I) {} };
  ThreadDispatcher *dispatch;
  IPCClientFontEngine(ThreadDispatcher *d, Fonts *p) : FontEngine(p), dispatch(d) {}

  const char *Name() const override { return "IPCClientFontEngine"; }
  unique_ptr<Font> Open(const FontDesc&) override;
  int InitGlyphs(Font *f, Glyph *g, int n) override;
  int LoadGlyphs(Font *f, const Glyph *g, int n) override;
  string DebugString(Font *f) const override;
  int OpenSystemFontResponse(Font *f, const IPC::OpenSystemFontResponse*, const MultiProcessBuffer&);
  static int GetId(Font *f);
};

struct IPCServerFontEngine : public FontEngine {
  IPCServerFontEngine(Fonts *p) : FontEngine(p) {}
  const char *Name() const override { return "IPCServerFontEngine"; }
  unique_ptr<Font> Open(const FontDesc&) override;
  int InitGlyphs(Font *f, Glyph *g, int n) override;
  int LoadGlyphs(Font *f, const Glyph *g, int n) override;
  string DebugString(Font *f) const override;
};

struct Fonts {
  static int InitFontWidth();
  static int InitFontHeight();
  struct Family {
    vector<string> normal, bold, italic, bold_italic;
    void Add(const string &n, int flag) {
      bool is_bold = flag & FontDesc::Bold, is_italic = flag & FontDesc::Italic;
      if (is_bold && is_italic) bold_italic.push_back(n);
      else if (is_bold)         bold       .push_back(n);
      else if (is_italic)       italic     .push_back(n);
      else                      normal     .push_back(n);
    }
  };

  ApplicationInfo *appinfo;
  GraphicsDeviceHolder *parent;
  AssetLoading *loader;
  shared_ptr<GlyphCache> rgba_glyph_cache, a_glyph_cache;
  FontEngine *default_font_engine=0;
  LazyInitializedPtr<FakeFontEngine> fake_engine;
  LazyInitializedPtr<AtlasFontEngine> atlas_engine;
  LazyInitializedPtr<FreeTypeFontEngine> freetype_engine;
  LazyInitializedPtr<CoreTextFontEngine> coretext_engine;
  LazyInitializedPtr<GDIFontEngine> gdi_engine;
  LazyInitializedPtr<FCFontEngine> fc_engine;
  LazyInitializedPtr<AndroidFontEngine> android_engine;
  LazyInitializedPtr<IPCClientFontEngine> ipc_client_engine;
  LazyInitializedPtr<IPCServerFontEngine> ipc_server_engine;
  unordered_map<FontDesc, unique_ptr<Font>, FontDesc::ColoredHasher, FontDesc::ColoredEqual> desc_map;
  unordered_map<string, Family> family_map;
  unordered_map<unsigned, FillColor> color_map;
  Font *default_font=0;
  Fonts(ApplicationInfo *A, GraphicsDeviceHolder *P, AssetLoading *L) : appinfo(A), parent(P), loader(L) {}
  virtual ~Fonts();

  void SelectFillColor(GraphicsDevice*);
  FillColor *GetFillColor(const Color&);

  Font *Find        (                    const FontDesc &d);
  Font *Insert      (FontEngine *engine, const FontDesc &d);
  Font *FindOrInsert(FontEngine *engine, const FontDesc &d);

  FontEngine *GetFontEngine(int engine_type);
  FontEngine *DefaultFontEngine();
  Font *Fake();
  Font *GetByDesc(FontDesc, int scale_window_height=0);
  template <class... Args> Font *Get(int scale_window_height, Args&&... args) { return GetByDesc(FontDesc(forward<Args>(args)...), scale_window_height); }
  Font *Change(Font*, int new_size, const Color &new_fg, const Color &new_bg, int new_flag=0);
  int ScaledFontSize(int pointsize, int scale_window_height);
  void ResetGL(GraphicsDeviceHolder *p, int flag);
  void LoadDefaultFonts();
  void LoadConsoleFont(const string &name, const vector<int> &sizes = vector<int>(1, 32));
  shared_ptr<GlyphCache> GetGlyphCache() {
    if (!rgba_glyph_cache) rgba_glyph_cache = make_shared<GlyphCache>(parent, GraphicsDevice::TextureRef(), 512);
    if (!rgba_glyph_cache->tex.ID) rgba_glyph_cache->tex.Create(rgba_glyph_cache->tex.width, rgba_glyph_cache->tex.height);
    return rgba_glyph_cache;
  }
};

struct FontRef {
  Font *ptr=0;
  FontDesc desc;
  FontRef(Font *F) { SetFont(F); }
  FontRef(Window *W=nullptr, const FontDesc &d=FontDesc()) : desc(d) { if (W) Load(W); }

  Font *Load(Window*);
  Font *Load(Window *w, FontDesc d) { desc=move(d); return Load(w); }
  operator Font* () const { return ptr; }
  Font* operator->() const { return ptr; }
  void SetFont(Font *F) { if ((ptr = F)) desc = *ptr->desc; }
};

struct DejaVuSansFreetype {
  static void SetDefault();
  static void Load(Fonts*);
};

}; // namespace LFL
#endif // LFL_CORE_APP_FONT_H__
